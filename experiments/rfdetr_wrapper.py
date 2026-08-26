"""
Script Name: rfdetr_wrapper.py
Description: Wraps RF-DETR to expose the same interface as Ultralytics YOLO,
             so the Albireo adaptive detector and vanilla baseline can use it without
             code changes.

             Key differences handled:
             - RF-DETR returns supervision.Detections (xyxy, confidence, class_id)
             - RF-DETR class_id is 1-indexed (COCO original); YOLO is 0-indexed
             - RF-DETR expects RGB; OpenCV loads BGR
             - RF-DETR uses `threshold` instead of `conf`

Author: Amir Taherin
Email: amirtaherin@gmail.com
Email: taherin.a@northeastern.edu
Date Created: 2026-04-12
Last Modified: 2026-04-12
Version: 1.0

License: MIT License

Usage:
    from rfdetr_wrapper import RFDETRModel
    model = RFDETRModel("rfdetr-base")  # or "rfdetr-large"
    # Now usable anywhere YOLO model is expected:
    results = model.predict(frame_bgr, imgsz=..., conf=0.5, verbose=False)
    boxes = results[0].boxes
"""

import cv2
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")


# COCO 0-indexed class names (same as YOLO)
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}


class BoxesWrapper:
    """Mimics ultralytics results[0].boxes interface."""

    def __init__(self, xyxy, xywh, cls, conf):
        self.xyxy = xyxy    # (N, 4) tensor
        self.xywh = xywh    # (N, 4) tensor
        self.cls = cls      # (N,) tensor
        self.conf = conf    # (N,) tensor

    def __len__(self):
        return self.xyxy.shape[0]


class ResultWrapper:
    """Mimics ultralytics results[0] interface."""

    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class RFDETRModel:
    """
    Wraps RF-DETR to match the Ultralytics YOLO model interface.

    Usage:
        model = RFDETRModel("rfdetr-base")
        results = model.predict(bgr_frame, conf=0.5, verbose=False)
        boxes = results[0].boxes  # .xyxy, .xywh, .cls, .conf
    """

    def __init__(self, model_name="rfdetr-base"):
        import rfdetr
        model_map = {
            "rfdetr-nano": rfdetr.RFDETRNano,
            "rfdetr-small": rfdetr.RFDETRSmall,
            "rfdetr-base": rfdetr.RFDETRBase,
            "rfdetr-medium": rfdetr.RFDETRMedium,
            "rfdetr-large": rfdetr.RFDETRLarge,
            "rfdetr-seg-xlarge": rfdetr.RFDETRSegXLarge,
            "rfdetr-seg-2xlarge": rfdetr.RFDETRSeg2XLarge,
        }
        if model_name not in model_map:
            raise ValueError(f"Unknown RF-DETR model: {model_name}. "
                             f"Available: {list(model_map.keys())}")
        self._model = model_map[model_name]()
        try:
            self._model.optimize_for_inference()
        except Exception:
            pass
        self.names = COCO_NAMES
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

    def to(self, device):
        """No-op for API compatibility — RF-DETR handles device internally."""
        return self

    def predict(self, frame_bgr, imgsz=None, conf=0.25, verbose=False, **kwargs):
        """
        Run RF-DETR inference on a BGR frame.

        Args:
            frame_bgr: BGR numpy array (H, W, 3) — standard OpenCV format.
            imgsz: Ignored (RF-DETR uses its own resolution).
            conf: Confidence threshold.
            verbose: Ignored.

        Returns:
            List with one ResultWrapper, matching ultralytics format.
        """
        # RF-DETR expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        dets = self._model.predict(frame_rgb, threshold=conf)

        if len(dets) == 0:
            boxes = BoxesWrapper(
                xyxy=torch.empty((0, 4)),
                xywh=torch.empty((0, 4)),
                cls=torch.empty((0,)),
                conf=torch.empty((0,)),
            )
            return [ResultWrapper(boxes, self.names)]

        # RF-DETR class_id is 1-indexed; convert to 0-indexed (YOLO convention)
        xyxy = torch.from_numpy(dets.xyxy.astype(np.float32))
        confidence = torch.from_numpy(dets.confidence.astype(np.float32))
        class_ids = torch.from_numpy((dets.class_id - 1).astype(np.int64))

        # Convert xyxy to xywh (center format)
        x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        xywh = torch.stack([cx, cy, w, h], dim=1)

        boxes = BoxesWrapper(xyxy=xyxy, xywh=xywh, cls=class_ids, conf=confidence)
        return [ResultWrapper(boxes, self.names)]
