"""
Script Name: train_erd_bdd100k.py
Description: Retrain the ERD (Empty Road Detection) CNN on BDD100K data.
             The original ERD was trained on Moscow surveillance footage and
             does not generalize to dashcam scenes. This script extracts
             empty/non-empty labels from BDD100K ground truth annotations and
             fine-tunes the ERD CNN.

             Empty = no tracked-class objects (person, car, truck, bus, etc.)
             Non-empty = at least one tracked-class object present

Author: Amir Taherin
Email: amirtaherin@gmail.com
Email: taherin.a@northeastern.edu
Date Created: 2026-04-14
Last Modified: 2026-04-14
Version: 1.0

License: MIT License

Usage:
    python train_erd_bdd100k.py --bdd-root ../../../dataset/bdd100k \
        --epochs 30 --batch-size 32 --output erd_bdd100k.pt
"""

import argparse
import json
import os
import pathlib
import re
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import ERD model definition from adaptive_detector
from adaptive_detector import _ERD_NET

# BDD100K tracked classes (same as bdd_loader.py)
EVAL_CLASSES = {
    "pedestrian", "rider", "car", "truck", "bus",
    "bicycle", "motorcycle", "train",
}

ERD_INPUT_W = 640
ERD_INPUT_H = 360


class BDD100K_ERD_Dataset(Dataset):
    """
    Binary classification dataset: empty (0) vs non-empty (1) road frames.

    Empty frames are identified as images present on disk but absent from the
    GT label JSON (which only contains frames with tracked objects), or frames
    in the GT with no eval-class labels.
    """

    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        if img is None:
            # Return a black frame if image can't be loaded
            img = np.zeros((ERD_INPUT_H, ERD_INPUT_W, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (ERD_INPUT_W, ERD_INPUT_H))

        # BGR → RGB, HWC → CHW, [0,1]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(
            rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        )
        label = self.labels[idx]
        return tensor, label


def build_dataset(bdd_root, split="val", detector_labels=None):
    """
    Scan BDD100K images and build empty/non-empty frame lists.

    If detector_labels is provided (path to JSON from generate_erd_labels.py),
    uses detector output as ground truth. Otherwise falls back to GT annotations.

    Returns: (image_paths, labels) where label is 0 (empty) or 1 (non-empty).
    """
    bdd_root = pathlib.Path(bdd_root)
    images_dir = bdd_root / "images" / "track" / split

    if detector_labels:
        # Detector-based labels (Option A)
        print(f"Loading detector-based labels from {detector_labels} ...")
        with open(detector_labels) as f:
            label_data = json.load(f)
        det_labels = label_data["labels"]  # filename -> 0 or 1
        print(f"  Detector: {label_data['model']}, conf={label_data['conf']}")

        image_paths = []
        labels = []
        for vdir in sorted(os.listdir(images_dir)):
            vpath = images_dir / vdir
            if not vpath.is_dir():
                continue
            for fname in sorted(os.listdir(vpath)):
                if not fname.endswith(".jpg"):
                    continue
                fpath = str(vpath / fname)
                label = det_labels.get(fname, 0)
                image_paths.append(fpath)
                labels.append(label)
    else:
        # GT-based labels (original approach)
        label_path = bdd_root / "labels_submission" / "val_gt.json"
        print(f"Loading GT labels from {label_path} ...")
        with open(label_path) as f:
            raw = json.load(f)

        gt_names = set()
        for entry in raw:
            name = entry.get("name", "")
            entry_labels = entry.get("labels", [])
            has_obj = any(l.get("category", "") in EVAL_CLASSES for l in entry_labels)
            if has_obj:
                gt_names.add(name)

        image_paths = []
        labels = []
        for vdir in sorted(os.listdir(images_dir)):
            vpath = images_dir / vdir
            if not vpath.is_dir():
                continue
            for fname in sorted(os.listdir(vpath)):
                if not fname.endswith(".jpg"):
                    continue
                fpath = str(vpath / fname)
                if fname in gt_names:
                    image_paths.append(fpath)
                    labels.append(1)
                else:
                    image_paths.append(fpath)
                    labels.append(0)

    n_empty = labels.count(0)
    n_nonempty = labels.count(1)
    print(f"Total frames: {len(labels)}")
    print(f"  Empty:     {n_empty} ({100*n_empty/len(labels):.1f}%)")
    print(f"  Non-empty: {n_nonempty} ({100*n_nonempty/len(labels):.1f}%)")

    return image_paths, labels


def balanced_split(image_paths, labels, val_ratio=0.2, seed=42):
    """
    Split data into train/val sets with balanced classes.
    Undersamples the majority class (non-empty) to match the minority (empty).
    """
    rng = random.Random(seed)

    empty_idxs = [i for i, l in enumerate(labels) if l == 0]
    nonempty_idxs = [i for i, l in enumerate(labels) if l == 1]

    rng.shuffle(empty_idxs)
    rng.shuffle(nonempty_idxs)

    # Undersample non-empty to 3x the empty count for training
    # This keeps some class imbalance (realistic) but prevents extreme skew
    max_nonempty = min(len(nonempty_idxs), 3 * len(empty_idxs))
    nonempty_idxs = nonempty_idxs[:max_nonempty]

    # Split each class proportionally
    n_empty_val = max(1, int(len(empty_idxs) * val_ratio))
    n_nonempty_val = max(1, int(len(nonempty_idxs) * val_ratio))

    val_idxs = empty_idxs[:n_empty_val] + nonempty_idxs[:n_nonempty_val]
    train_idxs = empty_idxs[n_empty_val:] + nonempty_idxs[n_nonempty_val:]

    rng.shuffle(train_idxs)
    rng.shuffle(val_idxs)

    train_paths = [image_paths[i] for i in train_idxs]
    train_labels = [labels[i] for i in train_idxs]
    val_paths = [image_paths[i] for i in val_idxs]
    val_labels = [labels[i] for i in val_idxs]

    print(f"\nTrain set: {len(train_paths)} ({train_labels.count(0)} empty, {train_labels.count(1)} non-empty)")
    print(f"Val set:   {len(val_paths)} ({val_labels.count(0)} empty, {val_labels.count(1)} non-empty)")

    return train_paths, train_labels, val_paths, val_labels


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build dataset
    image_paths, labels = build_dataset(
        args.bdd_root, detector_labels=args.detector_labels
    )
    train_paths, train_labels, val_paths, val_labels = balanced_split(
        image_paths, labels, val_ratio=0.2
    )

    train_dataset = BDD100K_ERD_Dataset(train_paths, train_labels)
    val_dataset = BDD100K_ERD_Dataset(val_paths, val_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Compute class weights for imbalanced data
    n_empty = train_labels.count(0)
    n_nonempty = train_labels.count(1)
    total = n_empty + n_nonempty
    weight_empty = total / (2 * n_empty)
    weight_nonempty = total / (2 * n_nonempty)
    class_weights = torch.tensor([weight_empty, weight_nonempty], dtype=torch.float32).to(device)
    print(f"\nClass weights: empty={weight_empty:.2f}, non-empty={weight_nonempty:.2f}")

    # Model
    if args.pretrained:
        print(f"Loading pretrained ERD weights from {args.pretrained}")
        model = _ERD_NET()
        model.load_state_dict(torch.load(args.pretrained, map_location="cpu", weights_only=True))
    else:
        print("Training from scratch")
        model = _ERD_NET()
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam([
        {"params": model.features.parameters(), "lr": args.lr_features},
        {"params": model.classifier.parameters(), "lr": args.lr_classifier},
    ])

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_imgs, batch_labels in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_imgs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == batch_labels).sum().item()
            train_total += batch_imgs.size(0)

        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_tp = val_fp = val_tn = val_fn = 0

        with torch.no_grad():
            for batch_imgs, batch_labels in val_loader:
                batch_imgs = batch_imgs.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == batch_labels).sum().item()
                val_total += batch_imgs.size(0)

                # Confusion matrix (class 0 = empty, class 1 = non-empty)
                for p, t in zip(preds, batch_labels):
                    p, t = p.item(), t.item()
                    if t == 0 and p == 0:
                        val_tn += 1  # correctly identified empty
                    elif t == 0 and p == 1:
                        val_fp += 1  # empty classified as non-empty (harmless)
                    elif t == 1 and p == 0:
                        val_fn += 1  # non-empty classified as empty (dangerous)
                    elif t == 1 and p == 1:
                        val_tp += 1  # correctly identified non-empty

        val_acc = val_correct / val_total
        precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        empty_recall = val_tn / (val_tn + val_fp) if (val_tn + val_fp) > 0 else 0

        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"train_loss={train_loss/train_total:.4f}  train_acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f}  "
              f"empty_recall={empty_recall:.3f}  nonempty_recall={recall:.3f}  "
              f"TP={val_tp} FP={val_fp} TN={val_tn} FN={val_fn}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.output)
            print(f"  → Saved best model (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Retrain ERD on BDD100K")
    parser.add_argument("--bdd-root", required=True, help="Path to BDD100K dataset root")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-features", type=float, default=2e-4)
    parser.add_argument("--lr-classifier", type=float, default=2e-3)
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained ERD weights for fine-tuning")
    parser.add_argument("--detector-labels", type=str, default=None,
                        help="Path to detector-generated labels JSON (from generate_erd_labels.py)")
    parser.add_argument("--output", type=str, default="erd_bdd100k.pt",
                        help="Output path for trained weights")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
