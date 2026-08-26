"""
Script Name: vanilla_detector.py
Description: Baseline YOLO detector that runs inference on every frame with no
             tracking or skip logic. Used as the comparison baseline for Experiments 1 & 2.

Author: Amir Taherin
Email: amirtaherin@gmail.com
Email: taherin.a@northeastern.edu
Date Created: 2026-03-27
Last Modified: 2026-03-27
Version: 1.0

License: MIT License

Usage:
    Imported as a module by run_experiment.py. Not intended to be run directly.

Notes:
    - Returns the same result dict schema as AdaptiveTracker.step() for fair comparison
    - ran_inference is always True
    - No state between frames
"""

import warnings
warnings.filterwarnings("ignore")


class VanillaDetector:
    """
    Simple frame-by-frame YOLO detector. Runs inference on every frame.
    Returns the same dict schema as AdaptiveTracker.step() for direct comparison.
    """

    def __init__(self, frame_width: int = 1920, frame_height: int = 1080, class_filter: set = None, yolo_conf: float = 0.25):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.class_filter = class_filter
        self.yolo_conf = yolo_conf
        # Round up to nearest multiple of 32 (YOLO max stride requirement)
        self._imgsz = (
            ((frame_height + 31) // 32) * 32,
            ((frame_width  + 31) // 32) * 32,
        )
        self.frame_idx = 0

    def step(self, frame, model, device: str) -> dict:
        """
        Run YOLO on this frame. Always runs inference.

        Returns:
            {
                'frame_idx': int,
                'ran_inference': True,
                'detections': list[dict],
                'num_tracks': int,
                'num_confirmed': int,
            }
        """
        results = model.predict(
            frame,
            imgsz=self._imgsz,
            conf=self.yolo_conf,
            verbose=False,
        )
        raw_boxes = results[0].boxes

        detections = []
        if raw_boxes is not None and len(raw_boxes) > 0:
            for i in range(len(raw_boxes)):
                cls_id = int(raw_boxes.cls[i].item())
                if self.class_filter is not None and cls_id not in self.class_filter:
                    continue
                xyxy = raw_boxes.xyxy[i].cpu().numpy().astype(int)
                cls_name = results[0].names[cls_id]
                conf = float(raw_boxes.conf[i].item())
                detections.append({
                    "track_id": None,
                    "class_name": cls_name,
                    "box_xyxy": xyxy.tolist(),
                    "conf": conf,
                    "state": "detected",
                })

        self.frame_idx += 1
        return {
            "frame_idx": self.frame_idx - 1,
            "ran_inference": True,
            "detections": detections,
            "num_tracks": len(detections),
            "num_confirmed": len(detections),
        }

    def reset(self) -> None:
        """Reset frame counter."""
        self.frame_idx = 0
