"""
Script Name: fixed_skip_detector.py
Description: Fixed-interval skip baseline. Runs the detector every N frames and
             uses 10D Kalman prediction on skipped frames. No ERD, no rescue,
             no uncertainty-driven skip — purely periodic inference.

Author: Amir Taherin
Email: taherin.a@northeastern.edu
Date Created: 2026-04-18
Last Modified: 2026-04-18
Version: 1.0

License: MIT License
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou(box_a, box_b):
    ax1 = box_a[0] - box_a[2] / 2;  ay1 = box_a[1] - box_a[3] / 2
    ax2 = box_a[0] + box_a[2] / 2;  ay2 = box_a[1] + box_a[3] / 2
    bx1 = box_b[0] - box_b[2] / 2;  by1 = box_b[1] - box_b[3] / 2
    bx2 = box_b[0] + box_b[2] / 2;  by2 = box_b[1] + box_b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter
    return inter / union if union > 0 else 0.0


class FixedSkipDetector:
    """
    Baseline detector that runs inference every N frames.
    On skipped frames, all tracks are propagated via Kalman prediction.
    Uses the same 10D Track class and two-stage matcher as Albireo,
    but the skip decision is fixed (every N) rather than adaptive.
    """

    def __init__(
        self,
        skip_n: int,
        frame_width: int = 1920,
        frame_height: int = 1080,
        sigma_meas_px: float = 5.0,
        max_lost_frames: int = 3,
        class_filter: set = None,
        yolo_conf: float = 0.25,
        Track=None,
        iou_threshold: float = 0.25,
        mahal_threshold: float = 9.4877,
        boundary_margin: float = 0.05,
        uncertainty_threshold: float = 3e-4,
    ):
        self.skip_n = skip_n
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

        self._imgsz = (
            ((frame_height + 31) // 32) * 32,
            ((frame_width + 31) // 32) * 32,
        )
        self._sigma_meas_norm = sigma_meas_px / max(self._imgsz)

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

        # Step 3: Fixed skip decision — run detector every N frames
        run_yolo = (self.frame_idx % self.skip_n == 0)

        # Step 4: Detector inference + matching
        if run_yolo:
            results = model.predict(frame, imgsz=self._imgsz, conf=self.yolo_conf, verbose=False)
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
                        "box": xywh,
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "conf": conf,
                    })

            active = [t for t in self.tracks if t.state != "lost"]
            matched_track_ids, matched_det_idxs = self._match(active, detections)
            matched_track_set = set(matched_track_ids)

            for t_idx, d_idx in zip(matched_track_ids, matched_det_idxs):
                d = detections[d_idx]
                active[t_idx].update(
                    d["box"], d["class_id"], d["class_name"], self.frame_idx,
                    conf=d["conf"],
                )

            # Immediate kill for unmatched tracks on inference frames
            for i, track in enumerate(active):
                if i not in matched_track_set:
                    track.state = "lost"

            # Spawn new tracks from unmatched detections
            matched_det_set = set(matched_det_idxs)
            for d_idx, d in enumerate(detections):
                if d_idx not in matched_det_set:
                    self.tracks.append(self.Track(
                        d["box"],
                        d["class_id"],
                        d["class_name"],
                        self.frame_idx,
                        frame_width=self.frame_width,
                        frame_height=self.frame_height,
                        sigma_meas=self._sigma_meas_norm,
                        max_lost_frames=self.max_lost_frames,
                        conf=d["conf"],
                    ))

        # Step 5: Prune lost tracks
        self.tracks = [t for t in self.tracks if t.state != "lost"]

        self.frame_idx += 1
        return {
            "frame_idx": self.frame_idx - 1,
            "ran_inference": run_yolo,
            "ran_erd": False,
            "erd_result": None,
            "num_tracks": len(self.tracks),
        }

    def _match(self, active_tracks, detections):
        if not active_tracks or not detections:
            return [], []

        n_tracks = len(active_tracks)
        n_dets = len(detections)
        det_boxes = np.array([d["box"] for d in detections])

        # Stage 1: IoU for recently-seen tracks
        recent_mask = [i for i, t in enumerate(active_tracks) if t.frames_since_update <= 2]
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

        # Stage 2: Mahalanobis for remaining tracks
        remaining_track_mask = [i for i in range(n_tracks) if i not in used_tracks]
        remaining_det_mask = [j for j in range(n_dets) if j not in used_dets]

        if remaining_track_mask and remaining_det_mask:
            rem_tracks = [active_tracks[i] for i in remaining_track_mask]
            rem_dets = [det_boxes[j] for j in remaining_det_mask]
            mah_cost = np.full((len(rem_tracks), len(rem_dets)), 1e9, dtype=np.float64)
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
