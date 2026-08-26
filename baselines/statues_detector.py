"""
Script Name: statues_detector.py
Description: Reimplementation of the Statues baseline (Kim, Kim, Ryu, ISLPED
             2024: "Statues: Energy-Efficient Video Object Detection on Edge
             Security Devices with Computational Skipping"). Pixel-level frame
             differencing with connected-component scoring to decide whether to
             skip the detector. On skip: reuse previous detection boxes verbatim
             (no tracking, no position update). This is fundamentally different
             from Albireo's object-level Kalman filter skipping and from SORT-
             style tracking — it has no state per object.

Author: Amir Taherin
Email: taherin.a@northeastern.edu
Date Created: 2026-04-24
Last Modified: 2026-04-24
Version: 1.0

License: MIT License

Usage:
    Imported as a module by run_experiment.py. Not intended to be run directly.

Notes:
    - Algorithm 1 from the Statues paper. Parameter defaults: s=2 (downscaling),
      pixel_threshold=30 (paper does not specify a value; 30 is standard for
      pixel-diff binarization), component_score_threshold=75 (paper's value).
    - No Track class needed; state is just the last downscaled grayscale frame
      plus the last emitted detection list.
    - Expected failure mode on dashcam: ego-motion changes every pixel, so even
      at loose thresholds the largest connected component exceeds the threshold
      on essentially every frame.
"""

import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
from scipy.ndimage import label as _cc_label


class StatuesDetector:
    """
    Pixel-level frame differencing baseline. On skip, reuses the previous
    frame's detections verbatim (no position update).
    """

    def __init__(
        self,
        pixel_threshold: int = 30,
        component_score_threshold: int = 75,
        downscale_s: int = 2,
        frame_width: int = 1920,
        frame_height: int = 1080,
        class_filter: set = None,
        yolo_conf: float = 0.25,
    ):
        self.pixel_threshold = pixel_threshold
        self.component_score_threshold = component_score_threshold
        self.s = downscale_s
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.class_filter = class_filter
        self.yolo_conf = yolo_conf

        self._imgsz = (
            ((frame_height + 31) // 32) * 32,
            ((frame_width + 31) // 32) * 32,
        )
        self.frame_idx = 0

        # Statues state: last downscaled grayscale frame + last detection list
        self._prev_gray_down: np.ndarray = None
        self._last_detections: list = []

    def step(self, frame, model, device: str) -> dict:
        """Process one video frame. Matches the dict schema of the other detectors."""

        # Downscale via average pooling (s x s blocks), then grayscale.
        # cv2.resize with INTER_AREA is equivalent to average pooling over the
        # downscaled region. Matches Statues' FDM exactly.
        h, w = frame.shape[:2]
        new_h, new_w = h // self.s, w // self.s
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_down = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # First frame: always run the detector (Statues Algorithm 1 line 7-8).
        if self._prev_gray_down is None:
            run_detector = True
        else:
            diff = np.abs(
                gray_down.astype(np.int16) - self._prev_gray_down.astype(np.int16)
            )
            binary = (diff >= self.pixel_threshold).astype(np.uint8)
            # 4-connectivity connected components (Statues uses the standard
            # labeling algorithm; connectivity choice is a minor detail).
            labeled, n_components = _cc_label(binary)
            run_detector = False
            if n_components > 0:
                # Largest component size (Statues' CBSM picks the max score).
                sizes = np.bincount(labeled.ravel())
                # bincount[0] is the background count; skip it.
                max_component = int(sizes[1:].max()) if len(sizes) > 1 else 0
                if max_component > self.component_score_threshold:
                    run_detector = True

        self._prev_gray_down = gray_down

        if run_detector:
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
            self._last_detections = detections
            out = detections
        else:
            # Statues' core behaviour: reuse the previous detection boxes verbatim.
            # Copy so later callers can mutate without cross-frame aliasing.
            out = [dict(d) for d in self._last_detections]

        self.frame_idx += 1
        return {
            "frame_idx":     self.frame_idx - 1,
            "ran_inference": run_detector,
            "detections":    out,
            "num_tracks":    len(out),
            "num_confirmed": len(out),
        }

    def reset(self) -> None:
        self.frame_idx = 0
        self._prev_gray_down = None
        self._last_detections = []
