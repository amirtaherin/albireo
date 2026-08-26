"""
Script Name: generate_erd_labels.py
Description: Generate ERD training labels by running the detector on every
             BDD100K frame. A frame is labeled "empty" if the detector returns
             zero detections for tracked classes, "non-empty" otherwise.

             This produces detector-aware labels rather than GT-based labels,
             ensuring ERD learns to predict whether running the detector would
             be useful — not whether the GT has annotations.

Author: Amir Taherin
Email: amirtaherin@gmail.com
Email: taherin.a@northeastern.edu
Date Created: 2026-04-14
Last Modified: 2026-04-14
Version: 1.0

License: MIT License

Usage:
    python generate_erd_labels.py --bdd-root ../../dataset/bdd100k \
        --model yolo11x --conf 0.25 --output erd_labels_yolo11x.json
"""

import argparse
import json
import os
import pathlib
import sys
import time

import cv2
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# Tracked classes (COCO IDs)
TRACKED_CLASSES = {0, 1, 2, 3, 5, 6, 7}  # person, bicycle, car, motorcycle, bus, train, truck


def main():
    parser = argparse.ArgumentParser(
        description="Generate ERD labels from detector output"
    )
    parser.add_argument("--bdd-root", required=True)
    parser.add_argument("--model", default="yolo11x")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", default="erd_labels_yolo11x.json")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    bdd_root = pathlib.Path(args.bdd_root)
    images_dir = bdd_root / "images" / "track" / args.split

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from ultralytics import YOLO
    model = YOLO(f"{args.model}.pt").to(device)
    print(f"Model: {args.model} on {device}, conf={args.conf}")

    # Compute imgsz for BDD100K (1280x720)
    frame_h, frame_w = 720, 1280
    imgsz = (
        ((frame_h + 31) // 32) * 32,
        ((frame_w + 31) // 32) * 32,
    )
    print(f"Image size: {imgsz}")

    # Scan all video directories
    video_dirs = sorted([
        d for d in images_dir.iterdir()
        if d.is_dir()
    ])
    print(f"Found {len(video_dirs)} videos")

    labels = {}  # filename -> 0 (empty) or 1 (non-empty)
    total_frames = 0
    empty_count = 0
    nonempty_count = 0
    t0 = time.time()

    for vi, vdir in enumerate(video_dirs):
        frame_files = sorted([
            f for f in vdir.iterdir()
            if f.suffix == ".jpg"
        ])

        for fpath in frame_files:
            img = cv2.imread(str(fpath))
            if img is None:
                continue

            results = model.predict(img, imgsz=imgsz, conf=args.conf, verbose=False)
            raw_boxes = results[0].boxes

            has_detection = False
            if raw_boxes is not None and len(raw_boxes) > 0:
                for i in range(len(raw_boxes)):
                    cls_id = int(raw_boxes.cls[i].item())
                    if cls_id in TRACKED_CLASSES:
                        has_detection = True
                        break

            label = 1 if has_detection else 0
            labels[fpath.name] = label
            total_frames += 1

            if label == 0:
                empty_count += 1
            else:
                nonempty_count += 1

        elapsed = time.time() - t0
        fps = total_frames / elapsed if elapsed > 0 else 0
        print(f"  [{vi+1}/{len(video_dirs)}] {vdir.name}: "
              f"total={total_frames}  empty={empty_count}  "
              f"non-empty={nonempty_count}  "
              f"empty%={100*empty_count/total_frames:.1f}%  "
              f"fps={fps:.1f}")

    # Save labels
    output = {
        "model": args.model,
        "conf": args.conf,
        "split": args.split,
        "total_frames": total_frames,
        "empty_count": empty_count,
        "nonempty_count": nonempty_count,
        "labels": labels,
    }

    with open(args.output, "w") as f:
        json.dump(output, f)

    print(f"\nDone. Saved {total_frames} labels to {args.output}")
    print(f"Empty:     {empty_count} ({100*empty_count/total_frames:.1f}%)")
    print(f"Non-empty: {nonempty_count} ({100*nonempty_count/total_frames:.1f}%)")


if __name__ == "__main__":
    main()
