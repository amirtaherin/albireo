"""
Script Name: erd_only_detector.py
Description: ERD-only ablation baseline. Runs the Empty Road Detector on every
             frame; if ERD says empty, emit zero detections; otherwise run the
             full YOLO detector. No Kalman filter, no tracking, no rescue. Used
             to isolate the contribution of scene-level ERD screening versus
             the full Albireo (KF + ERD + rescue) stack.

Author: Amir Taherin
Email: taherin.a@northeastern.edu
Date Created: 2026-04-24
Last Modified: 2026-04-24
Version: 1.0

License: MIT License

Usage:
    Imported as a module by run_experiment.py. Not intended to be run directly.

Notes:
    - ERD weights and preprocessing match Albireo.
    - On ERD-empty frames, returns an empty detection list (no object state to
      propagate since there are no tracks).
    - frame_state bookkeeping mirrors the adaptive output so plots can reuse
      the same machinery.
"""

import warnings

warnings.filterwarnings("ignore")

import pathlib
import sys

# Reuse the ERD loader/predictor from the Albireo package.
sys.path.insert(0, str((pathlib.Path(__file__).resolve().parent / "..").resolve()))
from albireo.detector import _load_erd, _erd_predict


class ERDOnlyDetector:
    """Run ERD on every frame; detector fires only on non-empty frames."""

    def __init__(
        self,
        frame_width: int = 1920,
        frame_height: int = 1080,
        class_filter: set = None,
        yolo_conf: float = 0.25,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.class_filter = class_filter
        self.yolo_conf = yolo_conf
        self._imgsz = (
            ((frame_height + 31) // 32) * 32,
            ((frame_width + 31) // 32) * 32,
        )
        self.frame_idx = 0

        self._erd_device = "cpu"
        self._erd_model = _load_erd(self._erd_device)

    def step(self, frame, model, device: str) -> dict:
        erd_class = _erd_predict(self._erd_model, frame, self._erd_device)
        non_empty = (erd_class == 1)

        detections = []
        if non_empty:
            results = model.predict(
                frame, imgsz=self._imgsz, conf=self.yolo_conf, verbose=False
            )
            raw_boxes = results[0].boxes
            if raw_boxes is not None and len(raw_boxes) > 0:
                for i in range(len(raw_boxes)):
                    cls_id = int(raw_boxes.cls[i].item())
                    if self.class_filter is not None and cls_id not in self.class_filter:
                        continue
                    xyxy = raw_boxes.xyxy[i].cpu().numpy().astype(int)
                    cls_name = results[0].names[cls_id]
                    conf = float(raw_boxes.conf[i].item())
                    detections.append({
                        "track_id":   None,
                        "class_name": cls_name,
                        "box_xyxy":   xyxy.tolist(),
                        "conf":       conf,
                        "state":      "detected",
                    })

        self.frame_idx += 1
        return {
            "frame_idx":     self.frame_idx - 1,
            "ran_inference": non_empty,
            "ran_erd":       True,
            "erd_result":    "non-empty" if non_empty else "empty",
            "detections":    detections,
            "num_tracks":    len(detections),
            "num_confirmed": len(detections),
        }

    def reset(self) -> None:
        self.frame_idx = 0
