"""
Script Name: ctd_detector.py
Description: Reimplementation of CTD (Ding et al., arXiv:1902.00615v6, 2024:
             "Confidence Trigger Detection: Accelerating Real-time Tracking-by-
             detection Systems"). Uses the Mahalanobis distance between the
             Kalman filter's current-frame prediction and the last detection
             box of each track to derive a per-track confidence via the
             chi-squared survival function. Skips the detector only if every
             active track is sufficiently confident AND a max-consecutive-skip
             counter has not been reached. On skip, Kalman-predicted positions
             are emitted (same as Albireo and FixedSkip).

Author: Amir Taherin
Email: taherin.a@northeastern.edu
Date Created: 2026-04-24
Last Modified: 2026-04-24
Version: 1.0

License: MIT License

Usage:
    Imported as a module by run_experiment.py. Not intended to be run directly.

Notes:
    - CTD's "confidence threshold" semantics follow the paper's convention:
        confidence = P(X >= M^2)   with X ~ chi-squared(df=4)
      Skip if confidence >= threshold. Paper sweeps 10%, 30%, 50%, 70%, 90%
      (Figure 3) and recommends ~30%. We expose this as `confidence_threshold`.
    - Paper sets max_consecutive_skips = 8 (Table II).
    - Uses the Albireo Track class (10D CA KF). CTD itself is agnostic to which
      KF is used — the paper uses DeepSORT's 8D CV. Using the same Track as
      Albireo controls for KF dimensionality in the comparison.
    - Matching reuses the two-stage IoU + Mahalanobis matcher from FixedSkip.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2


def _iou(box_a, box_b):
    ax1 = box_a[0] - box_a[2] / 2; ay1 = box_a[1] - box_a[3] / 2
    ax2 = box_a[0] + box_a[2] / 2; ay2 = box_a[1] + box_a[3] / 2
    bx1 = box_b[0] - box_b[2] / 2; by1 = box_b[1] - box_b[3] / 2
    bx2 = box_b[0] + box_b[2] / 2; by2 = box_b[1] + box_b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter
    return inter / union if union > 0 else 0.0


class CTDDetector:
    """
    Confidence Trigger Detection baseline.

    For each frame:
      1. Kalman predict all tracks.
      2. Boundary prune.
      3. Compute per-track Mahalanobis^2 between KF-predicted box (current) and
         last-known detection box for that track.
      4. Convert to per-track confidence via chi-squared survival function.
      5. Skip detection iff all active tracks confident AND consecutive skip
         counter < cap. Otherwise run detector, match, update.
    """

    # chi-squared df=4 (the paper fixes v=4 for bbox components cx,cy,w,h).
    _CHI2_DF = 4

    def __init__(
        self,
        confidence_threshold: float = 0.30,
        max_consecutive_skips: int = 8,
        frame_width: int = 1920,
        frame_height: int = 1080,
        sigma_meas_px: float = 5.0,
        max_lost_frames: int = 10,
        class_filter: set = None,
        yolo_conf: float = 0.25,
        Track=None,
        iou_threshold: float = 0.25,
        mahal_threshold: float = 9.4877,
        boundary_margin: float = 0.05,
        uncertainty_threshold: float = 3e-4,
    ):
        if Track is None:
            raise ValueError("CTDDetector requires a Track class (e.g., from Albireo)")
        self.confidence_threshold = confidence_threshold
        self.max_consecutive_skips = max_consecutive_skips
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_lost_frames = max_lost_frames
        self.class_filter = class_filter
        self.yolo_conf = yolo_conf
        self.Track = Track
        self.iou_threshold = iou_threshold
        self.mahal_threshold = mahal_threshold
        self.boundary_margin = boundary_margin
        self.uncertainty_threshold = uncertainty_threshold

        self.tracks: list = []
        self.frame_idx: int = 0
        self.consecutive_skips: int = 0

        # Per-track last-known detection box (in cxcywh pixel coordinates).
        # Keyed by Track.track_id.
        self._last_det_box: dict = {}

        self._imgsz = (
            ((frame_height + 31) // 32) * 32,
            ((frame_width + 31) // 32) * 32,
        )
        self._sigma_meas_norm = sigma_meas_px / max(self._imgsz)

        # Precompute the Mahalanobis^2 threshold corresponding to the
        # confidence threshold. Skip when M^2 <= this value.
        # confidence = P(X >= M^2) = 1 - chi2.cdf(M^2, df)
        # confidence >= T  <=>  M^2 <= chi2.ppf(1 - T, df)
        self._m2_threshold = float(
            chi2.ppf(1.0 - self.confidence_threshold, df=self._CHI2_DF)
        )

    def step(self, frame, model, device: str) -> dict:
        """Process one video frame."""

        # Step 1: Kalman predict for all tracks
        for track in self.tracks:
            track.predict()

        # Step 2: Boundary pruning
        for track in self.tracks:
            if track.is_out_of_frame(self.boundary_margin):
                track.state = "lost"
        self.tracks = [t for t in self.tracks if t.state != "lost"]
        # Drop last-detection records for any track we just lost.
        self._last_det_box = {
            tid: box for tid, box in self._last_det_box.items()
            if tid in {t.track_id for t in self.tracks}
        }

        # Step 3: CTD skip decision
        active = [t for t in self.tracks if t.state != "lost"]
        run_detector = self._should_run_detector(active)

        if not run_detector:
            # Skip frame: emit KF-predicted detections.
            self.consecutive_skips += 1
            detections_out = self._emit_predicted_detections(active)
            self.frame_idx += 1
            return {
                "frame_idx":     self.frame_idx - 1,
                "ran_inference": False,
                "detections":    detections_out,
                "num_tracks":    len(self.tracks),
                "num_confirmed": sum(1 for t in self.tracks if t.state == "confirmed"),
            }

        # Detector frame.
        self.consecutive_skips = 0
        results = model.predict(
            frame, imgsz=self._imgsz, conf=self.yolo_conf, verbose=False
        )
        raw_boxes = results[0].boxes

        detections = []
        if raw_boxes is not None and len(raw_boxes) > 0:
            for i in range(len(raw_boxes)):
                cls_id = int(raw_boxes.cls[i].item())
                if self.class_filter is not None and cls_id not in self.class_filter:
                    continue
                xywh = raw_boxes.xywh[i].cpu().numpy()
                cls_name = results[0].names[cls_id]
                conf = float(raw_boxes.conf[i].item())
                detections.append({
                    "box":        xywh,
                    "class_id":   cls_id,
                    "class_name": cls_name,
                    "conf":       conf,
                })

        matched_track_ids, matched_det_idxs = self._match(active, detections)
        matched_track_set = set(matched_track_ids)

        for t_idx, d_idx in zip(matched_track_ids, matched_det_idxs):
            d = detections[d_idx]
            active[t_idx].update(
                d["box"], d["class_id"], d["class_name"], self.frame_idx,
                conf=d["conf"],
            )
            # Record the fresh detection for future Mahalanobis comparisons.
            self._last_det_box[active[t_idx].track_id] = np.asarray(
                d["box"], dtype=np.float64
            )

        # Unmatched tracks on a detection frame: follow DeepSORT-style lifecycle
        # (mark_missed, lost after max_lost_frames). This is more faithful to
        # CTD (which uses DeepSORT) than FixedSkip's immediate-kill policy.
        for i, track in enumerate(active):
            if i not in matched_track_set:
                track.mark_missed()

        # Spawn new tracks from unmatched detections.
        matched_det_set = set(matched_det_idxs)
        for d_idx, d in enumerate(detections):
            if d_idx not in matched_det_set:
                new_track = self.Track(
                    d["box"],
                    d["class_id"],
                    d["class_name"],
                    self.frame_idx,
                    frame_width=self.frame_width,
                    frame_height=self.frame_height,
                    sigma_meas=self._sigma_meas_norm,
                    max_lost_frames=self.max_lost_frames,
                    conf=d["conf"],
                )
                self.tracks.append(new_track)
                self._last_det_box[new_track.track_id] = np.asarray(
                    d["box"], dtype=np.float64
                )

        # Prune lost tracks and their cached last-detection boxes.
        self.tracks = [t for t in self.tracks if t.state != "lost"]
        self._last_det_box = {
            tid: box for tid, box in self._last_det_box.items()
            if tid in {t.track_id for t in self.tracks}
        }

        # Emit track states as detections (mirrors FixedSkipDetector output).
        detections_out = self._emit_predicted_detections(self.tracks)

        self.frame_idx += 1
        return {
            "frame_idx":     self.frame_idx - 1,
            "ran_inference": True,
            "detections":    detections_out,
            "num_tracks":    len(self.tracks),
            "num_confirmed": sum(1 for t in self.tracks if t.state == "confirmed"),
        }

    def _should_run_detector(self, active: list) -> bool:
        """CTD skip decision."""
        # First frame (or no tracks yet) must run the detector to bootstrap.
        if self.frame_idx == 0 or not active:
            return True

        # Max-skip counter cap.
        if self.consecutive_skips >= self.max_consecutive_skips:
            return True

        # Any track without a recorded last detection forces a run. Should be
        # rare (newly-spawned tracks always get their last_det recorded).
        for t in active:
            if t.track_id not in self._last_det_box:
                return True

        # Per-track Mahalanobis^2 vs last-known detection. If any track's M^2
        # exceeds the confidence threshold (i.e., confidence < threshold),
        # fall back to running the detector.
        for t in active:
            last_box = self._last_det_box[t.track_id]
            m2 = t.mahalanobis_sq(last_box)
            if m2 > self._m2_threshold:
                return True

        return False

    def _emit_predicted_detections(self, tracks: list) -> list:
        """Format track state as detection dicts (same schema as VanillaDetector)."""
        out = []
        for t in tracks:
            if t.state == "lost":
                continue
            out.append({
                "track_id":   t.track_id,
                "class_name": t.class_name,
                "box_xyxy":   t.get_box_xyxy().tolist(),
                "conf":       t.get_confidence(self.uncertainty_threshold),
                "state":      t.state,
            })
        return out

    def _match(self, active_tracks, detections):
        # Identical to FixedSkipDetector._match — two-stage IoU then Mahalanobis.
        if not active_tracks or not detections:
            return [], []

        n_tracks = len(active_tracks)
        n_dets = len(detections)
        det_boxes = np.array([d["box"] for d in detections])

        recent_mask = [i for i, t in enumerate(active_tracks)
                       if t.frames_since_update <= 2]
        matched_track_ids = []
        matched_det_idxs = []
        used_tracks = set()
        used_dets = set()

        if recent_mask:
            recent_tracks = [active_tracks[i] for i in recent_mask]
            iou_cost = np.zeros((len(recent_tracks), n_dets), dtype=np.float64)
            for i, t in enumerate(recent_tracks):
                for j in range(n_dets):
                    iou_cost[i, j] = 1.0 - _iou(t.get_box_cxcywh(), det_boxes[j])
            rows, cols = linear_sum_assignment(iou_cost)
            for r, c in zip(rows, cols):
                if (1.0 - iou_cost[r, c]) >= self.iou_threshold:
                    matched_track_ids.append(recent_mask[r])
                    matched_det_idxs.append(c)
                    used_tracks.add(recent_mask[r])
                    used_dets.add(c)

        remaining_track_mask = [i for i in range(n_tracks) if i not in used_tracks]
        remaining_det_mask = [j for j in range(n_dets) if j not in used_dets]

        if remaining_track_mask and remaining_det_mask:
            rem_tracks = [active_tracks[i] for i in remaining_track_mask]
            rem_dets = [det_boxes[j] for j in remaining_det_mask]
            mah_cost = np.full((len(rem_tracks), len(rem_dets)), 1e9,
                                dtype=np.float64)
            for i, t in enumerate(rem_tracks):
                for j, db in enumerate(rem_dets):
                    md = t.mahalanobis_sq(db)
                    if md < self.mahal_threshold:
                        mah_cost[i, j] = md
            rows, cols = linear_sum_assignment(mah_cost)
            for r, c in zip(rows, cols):
                if mah_cost[r, c] < self.mahal_threshold:
                    matched_track_ids.append(remaining_track_mask[r])
                    matched_det_idxs.append(remaining_det_mask[c])

        return matched_track_ids, matched_det_idxs
