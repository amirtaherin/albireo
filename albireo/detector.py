"""
Albireo adaptive detector: the core frame-skipping system.

Wraps an off-the-shelf object detector and decides, per frame, whether
detector invocation can be safely skipped. A 10-dimensional per-object
Kalman filter [cx, cy, w, h, vx, vy, vw, vh, ax, ay] supplies a
forward-looking uncertainty trigger (tr(H P H^T) vs. threshold u);
a rescue rule preserves confirmed objects through brief detector misses
by injecting the KF-predicted box as a synthetic detection; and a
lightweight empty-scene screen (ERD) gates the detector when no active
object states exist.

Author: Amir Taherin (taherin.a@northeastern.edu)
License: MIT

Usage:
    Imported by experiments/run_experiment.py, or directly:
        from albireo.detector import AdaptiveDetector
"""

import pathlib
import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy.linalg
import torch
import torch.nn as nn
import cv2
import warnings
warnings.filterwarnings("ignore")

# --- Module-level constants ---
MAX_SKIP          = 15
MIN_HITS_CONFIRM  = 3
MAX_LOST_FRAMES   = 3
IOU_THRESHOLD     = 0.25
MAHAL_THRESHOLD   = 9.4877    # chi2(0.95, df=4)
MIN_INFER_EVERY   = 30
BOUNDARY_MARGIN   = 0.05

# Normalized Kalman parameters
SIGMA_P       = 0.003
SIGMA_S       = 0.002
SIGMA_V       = 0.002
SIGMA_A       = 0.001
SIGMA_MEAS_PX = 5.0

UNCERTAINTY_THRES = 3e-4

# KF rescue defaults
RESCUE_CONF_DEFAULT    = 0.5   # min last-detection conf to trust KF prediction
MAX_RESCUE_DEFAULT     = 2     # max consecutive I-frame misses rescued by KF

# ERD constants
ERD_INPUT_W = 640
ERD_INPUT_H = 360


# ==============================================================================
# ERD (Empty Road Detection) model
# ==============================================================================

class _ERD_NET(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = self._make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def _make_layers():
        config = [8, 8, 'M', 16, 16, 'M', 16, 16, 'M', 32, 32, 'M', 32, 32, 'M']
        layers = []
        in_channels = 3
        for c in config:
            if c == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2))
            else:
                layers.extend([
                    nn.Conv2d(in_channels, c, kernel_size=3, padding=1),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                ])
                in_channels = c
        return nn.Sequential(*layers)


def _load_erd(device="cpu"):
    weights_path = (pathlib.Path(__file__).resolve().parent.parent
                    / "erd_finetune" / "weights" / "erd_bdd100k.pt")
    if not weights_path.exists():
        raise FileNotFoundError(f"ERD weights not found: {weights_path}")
    model = _ERD_NET()
    model.load_state_dict(torch.load(str(weights_path), map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def _erd_predict(erd_model, frame_bgr, device="cpu"):
    resized = cv2.resize(frame_bgr, (ERD_INPUT_W, ERD_INPUT_H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = erd_model(tensor)
    return int(output.argmax(dim=1).item())


# ==============================================================================
# Track class — 10D Kalman filter with rescue counter
# ==============================================================================

class Track:
    """
    Single object track with a 10D Kalman filter.

    State: [cx, cy, w, h, vx, vy, vw, vh, ax, ay]

    Albireo addition: consecutive_rescues counts how many consecutive I-frames
    this track was rescued by the KF (detector missed, KF prediction used).
    Resets to 0 on any real detector match.
    """

    _id_counter: int = 0

    def __init__(
        self,
        box_cxcywh_px: np.ndarray,
        class_id:      int,
        class_name:    str,
        frame_idx:     int,
        frame_width:   int   = 1920,
        frame_height:  int   = 1080,
        sigma_p:       float = SIGMA_P,
        sigma_s:       float = SIGMA_S,
        sigma_v:       float = SIGMA_V,
        sigma_a:       float = SIGMA_A,
        sigma_meas:    float = SIGMA_MEAS_PX / 640,
        max_lost_frames: int = MAX_LOST_FRAMES,
        conf:          float = 1.0,
    ):
        self.track_id:    int = Track._id_counter
        Track._id_counter += 1

        self.class_id:    int   = class_id
        self.class_name:  str   = class_name
        self.state:       str   = "tentative"
        self.hits:        int   = 1
        self.frames_since_update: int = 0
        self.frames_since_infer:  int = 0
        self.age:         int   = 0
        self.last_seen_frame: int = frame_idx
        self.last_conf:   float = conf
        self.frame_width:  int  = frame_width
        self.frame_height: int  = frame_height
        self.max_lost_frames: int = max_lost_frames

        # KF rescue tracking
        self.consecutive_rescues: int = 0

        self._sx = float(frame_width)
        self._sy = float(frame_height)

        cx, cy, w, h = box_cxcywh_px
        self.x: np.ndarray = np.array(
            [cx / self._sx, cy / self._sy,
             w  / self._sx, h  / self._sy,
             0.0, 0.0,    # vx, vy
             0.0, 0.0,    # vw, vh
             0.0, 0.0],   # ax, ay
            dtype=np.float64,
        )

        self.F: np.ndarray = np.eye(10, dtype=np.float64)
        self.F[0, 4] = 1.0   # cx += vx
        self.F[1, 5] = 1.0   # cy += vy
        self.F[2, 6] = 1.0   # w  += vw
        self.F[3, 7] = 1.0   # h  += vh
        self.F[0, 8] = 0.5   # cx += 0.5*ax
        self.F[1, 9] = 0.5   # cy += 0.5*ay
        self.F[4, 8] = 1.0   # vx += ax
        self.F[5, 9] = 1.0   # vy += ay

        self.H: np.ndarray = np.zeros((4, 10), dtype=np.float64)
        self.H[:4, :4] = np.eye(4)

        self.Q: np.ndarray = np.diag([
            sigma_p ** 2, sigma_p ** 2,
            sigma_s ** 2, sigma_s ** 2,
            sigma_v ** 2, sigma_v ** 2,
            sigma_v ** 2, sigma_v ** 2,
            sigma_a ** 2, sigma_a ** 2,
        ]).astype(np.float64)

        self.R: np.ndarray = np.diag(
            [sigma_meas ** 2] * 4
        ).astype(np.float64)

        self.P: np.ndarray = np.diag([
            (1e-2) ** 2, (1e-2) ** 2,
            (5e-3) ** 2, (5e-3) ** 2,
            (5e-3) ** 2, (5e-3) ** 2,
            (5e-3) ** 2, (5e-3) ** 2,
            (1e-3) ** 2, (1e-3) ** 2,
        ]).astype(np.float64)

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.frames_since_update += 1
        self.frames_since_infer  += 1
        self.x[2] = max(self.x[2], 1.0 / self._sx)
        self.x[3] = max(self.x[3], 1.0 / self._sy)
        return self.get_box_cxcywh()

    def update(self, box_cxcywh_px, class_id, class_name, frame_idx, conf=1.0):
        cx, cy, w, h = box_cxcywh_px
        z = np.array(
            [cx / self._sx, cy / self._sy, w / self._sx, h / self._sy],
            dtype=np.float64,
        )
        innov = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        I_KH = np.eye(10) - K @ self.H
        self.P = I_KH @ self.P

        self.x[2] = max(self.x[2], 1.0 / self._sx)
        self.x[3] = max(self.x[3], 1.0 / self._sy)

        self.hits += 1
        self.frames_since_update = 0
        self.frames_since_infer  = 0
        self.last_seen_frame = frame_idx
        self.last_conf = conf
        self.class_id = class_id
        self.class_name = class_name

        if self.state == "tentative" and self.hits >= MIN_HITS_CONFIRM:
            self.state = "confirmed"

    def mark_missed(self):
        self.frames_since_update += 1
        if self.state == "confirmed" and self.frames_since_update > self.max_lost_frames:
            self.state = "lost"
        elif self.state == "tentative":
            self.state = "lost"

    def is_uncertain(self, threshold=UNCERTAINTY_THRES):
        return float(np.trace(self.P[0:4, 0:4])) > threshold

    def needs_forced_infer(self, max_skip=MAX_SKIP):
        return self.frames_since_infer >= max_skip

    def is_out_of_frame(self, margin=BOUNDARY_MARGIN):
        cx_n, cy_n = self.x[0], self.x[1]
        return (
            cx_n < -margin or cx_n > 1.0 + margin or
            cy_n < -margin or cy_n > 1.0 + margin
        )

    def get_box_cxcywh(self):
        cx_n, cy_n, w_n, h_n = self.x[0:4]
        return np.array(
            [cx_n * self._sx, cy_n * self._sy, w_n * self._sx, h_n * self._sy],
            dtype=np.float64,
        )

    def get_box_xyxy(self):
        cx, cy, w, h = self.get_box_cxcywh()
        x1 = max(0, min(int(cx - w / 2), self.frame_width  - 1))
        y1 = max(0, min(int(cy - h / 2), self.frame_height - 1))
        x2 = max(0, min(int(cx + w / 2), self.frame_width  - 1))
        y2 = max(0, min(int(cy + h / 2), self.frame_height - 1))
        return np.array([x1, y1, x2, y2], dtype=np.int32)

    def mahalanobis_sq(self, box_cxcywh_px):
        cx, cy, w, h = box_cxcywh_px
        z = np.array(
            [cx / self._sx, cy / self._sy, w / self._sx, h / self._sy],
            dtype=np.float64,
        )
        innov = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            S_inv_innov = scipy.linalg.solve(S, innov, assume_a='pos')
            return float(innov @ S_inv_innov)
        except scipy.linalg.LinAlgError:
            return 1e9

    def get_confidence(self, uncertainty_threshold=UNCERTAINTY_THRES):
        uncertainty_ratio = np.trace(self.P[0:4, 0:4]) / uncertainty_threshold
        decay = max(0.1, 1.0 - uncertainty_ratio)
        return self.last_conf * decay

    @classmethod
    def reset_id_counter(cls):
        cls._id_counter = 0


# ==============================================================================


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


class AdaptiveDetector:
    """
    Albireo tracker: Albireo + KF-augmented inference.

    On inference frames, confirmed tracks that the detector missed but had
    high confidence on their last real detection are rescued by injecting
    their KF predicted box as a synthetic detection. Capped at max_rescue
    consecutive I-frame misses.
    """

    def __init__(
        self,
        uncertainty_threshold: float = UNCERTAINTY_THRES,
        max_skip:              int   = MAX_SKIP,
        min_infer_every:       int   = MIN_INFER_EVERY,
        iou_threshold:         float = IOU_THRESHOLD,
        mahal_threshold:       float = MAHAL_THRESHOLD,
        frame_width:           int   = 1920,
        frame_height:          int   = 1080,
        sigma_meas_px:         float = SIGMA_MEAS_PX,
        max_lost_frames:       int   = MAX_LOST_FRAMES,
        boundary_margin:       float = BOUNDARY_MARGIN,
        class_filter:          set   = None,
        yolo_conf:             float = 0.25,
        rescue_conf:           float = RESCUE_CONF_DEFAULT,
        max_rescue:            int   = MAX_RESCUE_DEFAULT,
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.max_skip        = max_skip
        self.min_infer_every = min_infer_every
        self.iou_threshold   = iou_threshold
        self.mahal_threshold = mahal_threshold
        self.frame_width     = frame_width
        self.frame_height    = frame_height
        self.max_lost_frames = max_lost_frames
        self.boundary_margin = boundary_margin
        self.class_filter    = class_filter
        self.yolo_conf       = yolo_conf
        self.rescue_conf     = rescue_conf
        self.max_rescue      = max_rescue

        self.tracks:    list = []
        self.frame_idx: int  = 0

        self._imgsz = (
            ((frame_height + 31) // 32) * 32,
            ((frame_width  + 31) // 32) * 32,
        )
        self._sigma_meas_norm = sigma_meas_px / max(self._imgsz)

        self._erd_device = "cpu"
        self._erd_model = _load_erd(self._erd_device)

    def step(self, frame, model, device: str) -> dict:
        """Process one video frame."""

        ran_erd = False
        erd_result = None
        n_rescued = 0

        # Step 1: Kalman predict for all tracks
        for track in self.tracks:
            track.predict()

        # Step 2: Boundary pruning
        for track in self.tracks:
            if track.is_out_of_frame(self.boundary_margin):
                track.state = "lost"
        self.tracks = [t for t in self.tracks if t.state != "lost"]

        # Step 3: Decide whether to run detector
        active   = [t for t in self.tracks if t.state != "lost"]
        run_yolo = False

        if not active:
            erd_class = _erd_predict(self._erd_model, frame, self._erd_device)
            ran_erd = True
            if erd_class == 1:  # non-empty
                erd_result = "non-empty"
                run_yolo = True
            else:
                erd_result = "empty"
                run_yolo = False
        elif any(t.is_uncertain(self.uncertainty_threshold) for t in active):
            run_yolo = True
        elif any(t.needs_forced_infer(self.max_skip) for t in active):
            run_yolo = True
        elif self.frame_idx % self.min_infer_every == 0:
            run_yolo = True

        # Step 4: Detector inference + matching
        if run_yolo:
            results   = model.predict(frame, imgsz=self._imgsz, conf=self.yolo_conf, verbose=False)
            raw_boxes = results[0].boxes

            detections = []
            if raw_boxes is not None and len(raw_boxes) > 0:
                for i in range(len(raw_boxes)):
                    cls_id = int(raw_boxes.cls[i].item())
                    if self.class_filter is not None and cls_id not in self.class_filter:
                        continue
                    xywh     = raw_boxes.xywh[i].cpu().numpy()
                    cls_name = results[0].names[cls_id]
                    conf     = float(raw_boxes.conf[i].item())
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
                active[t_idx].consecutive_rescues = 0

            # Handle unmatched tracks: immediate kill OR KF rescue
            for i, track in enumerate(active):
                if i not in matched_track_set:
                    if (track.state == "confirmed"
                            and track.last_conf >= self.rescue_conf
                            and track.consecutive_rescues < self.max_rescue):
                        # Rescue: keep the track alive using KF prediction
                        track.consecutive_rescues += 1
                        track.frames_since_infer = 0
                        n_rescued += 1
                    else:
                        track.state = "lost"

            matched_det_set = set(matched_det_idxs)
            for d_idx, d in enumerate(detections):
                if d_idx not in matched_det_set:
                    self.tracks.append(Track(
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

        # Step 6: Build result
        confirmed = [t for t in self.tracks if t.state == "confirmed"]
        detections_out = [
            {
                "track_id":   t.track_id,
                "class_name": t.class_name,
                "box_xyxy":   t.get_box_xyxy().tolist(),
                "conf":       t.get_confidence(self.uncertainty_threshold),
                "state":      t.state,
            }
            for t in confirmed
        ]

        self.frame_idx += 1
        return {
            "frame_idx":     self.frame_idx - 1,
            "ran_inference": run_yolo,
            "ran_erd":       ran_erd,
            "erd_result":    erd_result,
            "detections":    detections_out,
            "num_tracks":    len(self.tracks),
            "num_confirmed": len(confirmed),
            "num_rescued":   n_rescued,
        }

    def _match(self, active_tracks, detections):
        if not active_tracks or not detections:
            return [], []

        n_tracks = len(active_tracks)
        n_dets   = len(detections)
        det_boxes = np.array([d["box"] for d in detections])

        # Stage 1: IoU for recently-seen tracks
        recent_mask = [i for i, t in enumerate(active_tracks) if t.frames_since_update <= 2]
        matched_track_ids = []
        matched_det_idxs  = []
        used_tracks = set()
        used_dets   = set()

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
        remaining_det_mask   = [j for j in range(n_dets)   if j not in used_dets]

        if remaining_track_mask and remaining_det_mask:
            rem_tracks = [active_tracks[i] for i in remaining_track_mask]
            rem_dets   = [det_boxes[j]     for j in remaining_det_mask]
            mah_cost   = np.full((len(rem_tracks), len(rem_dets)), 1e9, dtype=np.float64)
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
