"""
Script Name: map_evaluator.py
Description: VOC-style mAP evaluator for per-frame object detection accuracy.
             Computes AP@0.5 and mAP@0.5:0.95 (COCO-style) by comparing predicted
             bounding boxes against ground truth annotations.

Author: Amir Taherin
Email: amirtaherin@gmail.com
Email: taherin.a@northeastern.edu
Date Created: 2026-03-29
Last Modified: 2026-03-29
Version: 1.0

License: MIT License

Usage:
    Imported as a module by run_experiment.py. Not intended to be run directly.

Notes:
    - VOC 2010+ all-point interpolation for AP
    - Each GT box is matched at most once (highest-confidence prediction wins)
    - Predictions must include "box_xyxy" and "conf"; GT must include "box_xyxy"
"""

import numpy as np


def compute_iou(box_a, box_b):
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def compute_iou_matrix(preds, gts):
    """
    Compute IoU matrix between predicted and ground truth boxes.

    Args:
        preds: list of dicts with "box_xyxy"
        gts: list of dicts with "box_xyxy"

    Returns:
        np.ndarray of shape (len(preds), len(gts))
    """
    n_preds = len(preds)
    n_gts = len(gts)
    iou_mat = np.zeros((n_preds, n_gts), dtype=np.float64)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            iou_mat[i, j] = compute_iou(p["box_xyxy"], g["box_xyxy"])
    return iou_mat


class SequenceEvaluator:
    """
    Accumulates per-frame detections and ground truth over a sequence,
    then computes AP and mAP.
    """

    def __init__(self):
        # Accumulated across all frames: (conf, is_tp, iou_with_match) per prediction
        self._predictions = []  # list of (conf, frame_idx)
        self._pred_boxes = []   # list of box_xyxy arrays
        self._gt_boxes = []     # list of (frame_idx, box_xyxy)
        self._frame_data = []   # list of (preds_for_frame, gts_for_frame)
        self.total_gt = 0
        self._per_frame_stats = []  # (frame_idx, n_preds, n_gt, n_tp, n_fp, n_fn)

    def add_frame(self, frame_idx, predictions, ground_truth):
        """
        Add one frame of predictions and ground truth.

        Args:
            frame_idx: int
            predictions: list of {"box_xyxy": array, "conf": float}
            ground_truth: list of {"box_xyxy": array, ...}
        """
        self._frame_data.append((frame_idx, predictions, ground_truth))
        self.total_gt += len(ground_truth)

    def compute_ap(self, iou_threshold=0.5):
        """
        Compute Average Precision at a given IoU threshold.
        Uses VOC 2010+ all-point interpolation.

        Returns:
            (ap, precision_array, recall_array)
        """
        # Collect all predictions with their TP/FP labels
        all_preds = []  # (conf, is_tp)

        for frame_idx, preds, gts in self._frame_data:
            if len(preds) == 0:
                continue

            if len(gts) == 0:
                # All predictions are FP
                for p in preds:
                    all_preds.append((p["conf"], False))
                continue

            # Compute IoU matrix
            iou_mat = compute_iou_matrix(preds, gts)

            # Sort predictions by confidence (descending) for this frame
            sorted_indices = sorted(range(len(preds)), key=lambda i: preds[i]["conf"], reverse=True)

            gt_matched = [False] * len(gts)

            for pred_idx in sorted_indices:
                # Find best matching GT
                best_iou = 0.0
                best_gt = -1
                for gt_idx in range(len(gts)):
                    if gt_matched[gt_idx]:
                        continue
                    if iou_mat[pred_idx, gt_idx] > best_iou:
                        best_iou = iou_mat[pred_idx, gt_idx]
                        best_gt = gt_idx

                if best_iou >= iou_threshold and best_gt >= 0:
                    all_preds.append((preds[pred_idx]["conf"], True))
                    gt_matched[best_gt] = True
                else:
                    all_preds.append((preds[pred_idx]["conf"], False))

        if not all_preds:
            return 0.0, np.array([]), np.array([])

        # Sort all predictions globally by confidence (descending)
        all_preds.sort(key=lambda x: x[0], reverse=True)

        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        for conf, is_tp in all_preds:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / max(self.total_gt, 1)
            precisions.append(precision)
            recalls.append(recall)

        precisions = np.array(precisions)
        recalls = np.array(recalls)

        # VOC 2010+ all-point interpolation
        # For each recall level, precision = max precision at recall >= current
        ap = 0.0
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])

        # Find points where recall changes
        recall_diff = np.diff(recalls, prepend=0.0)
        ap = np.sum(recall_diff * precisions)

        return float(ap), precisions, recalls

    def compute_map(self, iou_thresholds=None):
        """
        Compute COCO-style mAP averaged over multiple IoU thresholds.

        Args:
            iou_thresholds: list/array of IoU thresholds. Default: [0.50, 0.55, ..., 0.95]

        Returns:
            (map_value, {iou_threshold: ap_value})
        """
        if iou_thresholds is None:
            iou_thresholds = np.arange(0.50, 1.00, 0.05)

        ap_per_threshold = {}
        for t in iou_thresholds:
            ap, _, _ = self.compute_ap(iou_threshold=t)
            ap_per_threshold[round(float(t), 2)] = ap

        map_value = np.mean(list(ap_per_threshold.values()))
        return float(map_value), ap_per_threshold

    def compute_per_frame_stats(self, iou_threshold=0.5):
        """
        Compute per-frame TP, FP, FN counts at a given IoU threshold.

        Returns:
            list of (frame_idx, n_preds, n_gt, n_tp, n_fp, n_fn)
        """
        stats = []
        for frame_idx, preds, gts in self._frame_data:
            if len(preds) == 0:
                stats.append((frame_idx, 0, len(gts), 0, 0, len(gts)))
                continue
            if len(gts) == 0:
                stats.append((frame_idx, len(preds), 0, 0, len(preds), 0))
                continue

            iou_mat = compute_iou_matrix(preds, gts)
            sorted_indices = sorted(range(len(preds)), key=lambda i: preds[i]["conf"], reverse=True)
            gt_matched = [False] * len(gts)
            tp = 0
            fp = 0

            for pred_idx in sorted_indices:
                best_iou = 0.0
                best_gt = -1
                for gt_idx in range(len(gts)):
                    if gt_matched[gt_idx]:
                        continue
                    if iou_mat[pred_idx, gt_idx] > best_iou:
                        best_iou = iou_mat[pred_idx, gt_idx]
                        best_gt = gt_idx
                if best_iou >= iou_threshold and best_gt >= 0:
                    tp += 1
                    gt_matched[best_gt] = True
                else:
                    fp += 1

            fn = sum(1 for m in gt_matched if not m)
            stats.append((frame_idx, len(preds), len(gts), tp, fp, fn))

        return stats

    def get_pr_curve(self, iou_threshold=0.5):
        """Returns (precision_array, recall_array) for plotting."""
        _, precisions, recalls = self.compute_ap(iou_threshold)
        return precisions, recalls
