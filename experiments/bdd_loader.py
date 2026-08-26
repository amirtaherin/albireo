"""
Script Name: bdd_loader.py
Description: BDD100K MOT dataset loader for Experiment 3. Loads per-video
             annotations from a single flat JSON (submission/GT format) and
             the corresponding JPEG frames.

Author: Amir Taherin
Email: amirtaherin@gmail.com
Email: taherin.a@northeastern.edu
Date Created: 2026-04-10
Last Modified: 2026-04-10
Version: 1.1

License: MIT License

Usage:
    Imported as a module by run_experiment.py. Not intended to be run directly.

Notes:
    - GT labels are in the flat BDD100K submission/tracking format:
        a single JSON file where each element is:
          {"name": "{video_name}-{frame:07d}.jpg", "labels": [...]}
      This is the format used by bdd100k_tracking20_val_example_submission.json.
    - Frames are stored as individual JPEGs:
        images/track/{split}/{video_name}/{video_name}-{frame:07d}.jpg
    - Frame numbers are 1-indexed and sequential (1 to N_frames).
    - BDD categories are mapped to COCO class IDs so YOLO detections can be
      compared directly to ground truth.

BDD100K → COCO class mapping used for evaluation:
    pedestrian  → 0  (person)
    rider       → 0  (person — person riding a vehicle)
    car         → 2  (car)
    truck       → 7  (truck)
    bus         → 5  (bus)
    bicycle     → 1  (bicycle)
    motorcycle  → 3  (motorcycle)
    train       → 6  (train)
    other vehicle / other person / trailer → skipped (not in COCO)
"""

import json
import pathlib
from typing import Dict, Iterator, List, Optional, Set, Tuple

import cv2
import numpy as np


# BDD100K category name → COCO class ID (YOLO uses COCO IDs)
BDD_TO_COCO: Dict[str, int] = {
    "pedestrian":   0,   # COCO: person
    "rider":        0,   # COCO: person
    "car":          2,   # COCO: car
    "truck":        7,   # COCO: truck
    "bus":          5,   # COCO: bus
    "bicycle":      1,   # COCO: bicycle
    "motorcycle":   3,   # COCO: motorcycle
    "train":        6,   # COCO: train
    # Skipped: "other vehicle", "other person", "trailer"
}

# Default classes to include in evaluation (subset of BDD_TO_COCO)
DEFAULT_EVAL_CLASSES: Set[str] = {
    "pedestrian", "rider", "car", "truck", "bus", "bicycle", "motorcycle",
}


class BDDSequence:
    """
    One BDD100K MOT video sequence — frames + per-frame ground-truth annotations.

    Frames are loaded lazily (not all in memory at once).
    """

    def __init__(
        self,
        video_name: str,
        frames_dir: pathlib.Path,
        frame_records: List[dict],          # sorted list of per-frame label dicts
        eval_classes: Optional[Set[str]] = None,
    ):
        self.video_name = video_name
        self._frames_dir = frames_dir
        self._eval_classes = eval_classes or DEFAULT_EVAL_CLASSES
        self._frame_records = frame_records  # already sorted by frame index

        # Infer frame dimensions from first available frame
        if self._frame_records:
            first_frame_path = self._get_frame_path(self._frame_records[0]["_frame_idx"])
            if first_frame_path.exists():
                img = cv2.imread(str(first_frame_path))
                if img is not None:
                    self.frame_height, self.frame_width = img.shape[:2]
                else:
                    self.frame_height, self.frame_width = 720, 1280
            else:
                self.frame_height, self.frame_width = 720, 1280
        else:
            self.frame_height, self.frame_width = 720, 1280

    def _get_frame_path(self, frame_idx: int) -> pathlib.Path:
        """Return path to the JPEG frame for a given 1-indexed frame number."""
        # BDD100K MOT frames are named {video_name}-{frame:07d}.jpg
        return self._frames_dir / f"{self.video_name}-{frame_idx:07d}.jpg"

    def __len__(self) -> int:
        return len(self._frame_records)

    def get_frame(self, idx: int) -> Tuple[np.ndarray, List[dict]]:
        """
        Returns (image_bgr, gt_boxes) for the sequence position `idx` (0-based).

        gt_boxes: list of dicts with keys:
            "box_xyxy":  [x1, y1, x2, y2] in pixels (float)
            "class_id":  COCO class ID (int)
            "class_name": BDD category string
            "track_id":  str — track ID within this video
        """
        record = self._frame_records[idx]
        frame_path = self._get_frame_path(record["_frame_idx"])

        img = cv2.imread(str(frame_path))
        if img is None:
            raise FileNotFoundError(f"Frame not found: {frame_path}")

        gt_boxes = []
        for label in record.get("labels", []):
            cat = label.get("category", "")
            if cat not in self._eval_classes:
                continue
            coco_id = BDD_TO_COCO.get(cat)
            if coco_id is None:
                continue

            box2d = label.get("box2d")
            if box2d is None:
                continue

            x1 = float(box2d["x1"])
            y1 = float(box2d["y1"])
            x2 = float(box2d["x2"])
            y2 = float(box2d["y2"])

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            gt_boxes.append({
                "box_xyxy":   [x1, y1, x2, y2],
                "class_id":   coco_id,
                "class_name": cat,
                "track_id":   str(label.get("id", "")),
            })

        return img, gt_boxes

    def iter_frames(self) -> Iterator[Tuple[int, np.ndarray, List[dict]]]:
        """Iterate over all frames: yields (frame_idx, image_bgr, gt_boxes)."""
        for i in range(len(self)):
            img, gt = self.get_frame(i)
            yield i, img, gt


class BDDDataset:
    """
    BDD100K MOT dataset — loads all (or selected) validation sequences from a
    flat submission-format JSON label file.

    Expected directory layout:
        {bdd_root}/
            images/
                track/
                    val/
                        {video_name}/
                            {video_name}-0000001.jpg
                            {video_name}-0000002.jpg
                            ...
            labels_submission/
                val_gt.json   ← flat list: [{"name": "{video}-XXXXXXX.jpg",
                                             "labels": [{box2d, category, id}, ...]}, ...]
    """

    def __init__(
        self,
        bdd_root: str,
        split: str = "val",
        eval_classes: Optional[Set[str]] = None,
        label_file: Optional[str] = None,
    ):
        self._root = pathlib.Path(bdd_root)
        self._split = split
        self._eval_classes = eval_classes or DEFAULT_EVAL_CLASSES

        self._images_dir = self._root / "images" / "track" / split

        # Label file: default location
        if label_file is None:
            label_file = str(self._root / "labels_submission" / "val_gt.json")
        self._label_path = pathlib.Path(label_file)

        if not self._images_dir.exists():
            raise FileNotFoundError(
                f"BDD100K image directory not found: {self._images_dir}\n"
                f"Expected layout: {{bdd_root}}/images/track/{split}/"
            )
        if not self._label_path.exists():
            raise FileNotFoundError(
                f"BDD100K label file not found: {self._label_path}\n"
                f"Expected: {{bdd_root}}/labels_submission/val_gt.json"
            )

        # Parse flat JSON and build per-video index
        with open(self._label_path) as f:
            raw = json.load(f)

        # Group by video name, extract frame index from filename
        import re
        _video_data: Dict[str, list] = {}
        for entry in raw:
            name = entry.get("name", "")
            # name format: "{video_name}-{frame:07d}.jpg"
            m = re.match(r"^(.*)-(\d{7})\.jpg$", name)
            if not m:
                continue
            video_name, frame_str = m.group(1), m.group(2)
            frame_idx = int(frame_str)
            if video_name not in _video_data:
                _video_data[video_name] = []
            record = dict(entry)
            record["_frame_idx"] = frame_idx
            _video_data[video_name].append(record)

        # Sort each video's frames by frame index
        for v in _video_data:
            _video_data[v].sort(key=lambda x: x["_frame_idx"])

        self._video_data = _video_data
        self._video_names: List[str] = sorted(_video_data.keys())

        if not self._video_names:
            raise RuntimeError(f"No valid video entries found in {self._label_path}")

    def get_sequence_names(self) -> List[str]:
        return list(self._video_names)

    def get_sequence(self, video_name: str) -> BDDSequence:
        if video_name not in self._video_data:
            raise KeyError(f"Video '{video_name}' not found in label file")

        frames_dir = self._images_dir / video_name
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

        return BDDSequence(
            video_name=video_name,
            frames_dir=frames_dir,
            frame_records=self._video_data[video_name],
            eval_classes=self._eval_classes,
        )

    def __len__(self) -> int:
        return len(self._video_names)
