"""
Script Name: run_experiment.py
Description: Experiment 3 / 4 orchestrator. Runs vanilla YOLO and the adaptive
             Kalman tracker (plus FixedSkip, CTD, Statues, ERD-only baselines)
             on BDD100K MOT or VIRAT-Ground clips and compares detection
             accuracy (mAP) plus energy/timing.

Author: Amir Taherin
Email: amirtaherin@gmail.com
Email: taherin.a@northeastern.edu
Date Created: 2026-04-10
Last Modified: 2026-04-27
Version: 1.1

License: MIT License

Usage:
    python run_experiment.py [options]

    # BDD100K (default)
    python run_experiment.py --bdd-root /path/to/bdd100k \
        --model yolo11x --platform xavier --num-clips 30

    # VIRAT-Ground
    python run_experiment.py --dataset virat --virat-root /path/to/virat \
        --model yolo26x --platform thor --num-clips 50

Notes:
    - BDD100K MOT val split has 200 clips, ~200 frames each at 5fps.
    - VIRAT-Ground evaluates: person, car, "other vehicle"->truck, bike->bicycle.
    - By default runs on a random sample of --num-clips clips for speed.
    - On skipped frames, confidence is decayed by sqrt(trace(P)/threshold) to
      reflect Kalman uncertainty in the mAP computation.
"""

import argparse
import csv
import gzip
import importlib.util
import json
import os
import pathlib
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import torch

# --- Jetson Workaround: torchvision C++ NMS ops may not be compiled ---
def _pure_torch_nms(boxes, scores, iou_threshold):
    """Pure-PyTorch NMS fallback for Jetson where torchvision C++ ops are unavailable."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ix1 = torch.clamp(x1[rest], min=x1[i].item())
        iy1 = torch.clamp(y1[rest], min=y1[i].item())
        ix2 = torch.clamp(x2[rest], max=x2[i].item())
        iy2 = torch.clamp(y2[rest], max=y2[i].item())
        inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
        order = rest[iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)

try:
    import torchvision
    torchvision.ops.nms = _pure_torch_nms
    import torchvision.ops.boxes as _tvb
    _tvb.nms = _pure_torch_nms
except ImportError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _sub in ("experiments", "baselines", "telemetry"):
    sys.path.insert(0, str(_ROOT / _sub))
sys.path.insert(0, str(_ROOT))
from albireo.detector import AdaptiveDetector, Track
from bdd_loader import BDDDataset, BDD_TO_COCO, DEFAULT_EVAL_CLASSES
# virat_loader is imported lazily in main() only when --dataset virat is selected,
# to avoid requiring opencv-python for BDD-only runs.
from vanilla_detector import VanillaDetector
from fixed_skip_detector import FixedSkipDetector
from ctd_detector import CTDDetector
from statues_detector import StatuesDetector
from erd_only_detector import ERDOnlyDetector
from map_evaluator import SequenceEvaluator
from clip_tegrastats import ClipTegrastats, _SUMMARY_KEYS


# COCO class IDs that correspond to the BDD classes we evaluate (default).
# Replaced in main() with VIRAT-specific IDs when --dataset virat.
EVAL_COCO_IDS = set(BDD_TO_COCO[c] for c in DEFAULT_EVAL_CLASSES)


# ==============================================================================
# Run functions
# ==============================================================================

def run_vanilla_on_sequence(seq, model, device, yolo_conf=0.25,
                            tegrastats_log=None):
    """Run vanilla YOLO on every frame. Returns per-frame detections and timing."""
    detector = VanillaDetector(
        frame_width=seq.frame_width,
        frame_height=seq.frame_height,
        class_filter=EVAL_COCO_IDS,
        yolo_conf=yolo_conf,
    )
    tgs = ClipTegrastats(log_path=tegrastats_log)
    per_frame_dets = []
    frame_times_ms = []

    tgs.start()
    wall_start = time.perf_counter()

    for i in range(len(seq)):
        img, _ = seq.get_frame(i)
        t0 = time.perf_counter()
        result = detector.step(img, model, device)
        t1 = time.perf_counter()
        per_frame_dets.append(result["detections"])
        frame_times_ms.append((t1 - t0) * 1000.0)

    wall_duration = time.perf_counter() - wall_start
    tgs.stop()

    return {
        "per_frame_dets": per_frame_dets,
        "frame_times_ms": frame_times_ms,
        "wall_duration_s": wall_duration,
        "tegrastats": tgs.get_summary(),
        "infer_count": len(seq),
        "skip_count": 0,
    }


def run_adaptive_on_sequence(seq, model, device, args, AdaptiveDetector, Track,
                             tegrastats_log=None):
    """Run the Albireo adaptive detector on a BDD sequence. Returns per-frame detections and timing."""
    import inspect
    Track.reset_id_counter()
    _all_kwargs = dict(
        uncertainty_threshold=args.uncertainty_threshold,
        max_skip=args.max_skip,
        min_infer_every=args.min_infer_every,
        frame_width=seq.frame_width,
        frame_height=seq.frame_height,
        sigma_meas_px=args.sigma_meas_px,
        max_lost_frames=args.max_lost_frames,
        class_filter=EVAL_COCO_IDS,
        yolo_conf=args.yolo_conf,
        rescue_conf=args.rescue_conf,
        max_rescue=args.max_rescue,
    )
    _valid = set(inspect.signature(AdaptiveDetector.__init__).parameters) - {"self"}
    tracker = AdaptiveDetector(**{k: v for k, v in _all_kwargs.items() if k in _valid})

    tgs = ClipTegrastats(log_path=tegrastats_log)
    per_frame_dets = []
    frame_times_ms = []
    infer_count = 0
    skip_count = 0
    erd_empty_count = 0
    predict_count = 0
    erd_infer_count = 0
    augmented_infer_count = 0

    tgs.start()
    wall_start = time.perf_counter()

    for i in range(len(seq)):
        img, _ = seq.get_frame(i)
        t0 = time.perf_counter()
        result = tracker.step(img, model, device)
        t1 = time.perf_counter()

        # Use ALL non-lost tracks for detection output (not just confirmed).
        # We are evaluating detection AP, not tracking — track identity and
        # lifecycle state don't matter; we want every predicted box we have.
        frame_dets = [
            {
                "box_xyxy": t.get_box_xyxy().tolist(),
                "conf":     t.get_confidence(tracker.uncertainty_threshold),
                "class_id": t.class_id,
                "class_name": t.class_name,
            }
            for t in tracker.tracks
            if t.state != "lost"
        ]
        per_frame_dets.append(frame_dets)
        frame_times_ms.append((t1 - t0) * 1000.0)
        if result["ran_inference"]:
            infer_count += 1
            if result.get("ran_erd") and result.get("erd_result") == "non-empty":
                erd_infer_count += 1
            elif result.get("num_rescued", 0) >= 1:
                augmented_infer_count += 1
        else:
            skip_count += 1
            if result.get("ran_erd") and result.get("erd_result") == "empty":
                erd_empty_count += 1
            else:
                predict_count += 1

    wall_duration = time.perf_counter() - wall_start
    tgs.stop()

    return {
        "per_frame_dets": per_frame_dets,
        "frame_times_ms": frame_times_ms,
        "wall_duration_s": wall_duration,
        "tegrastats": tgs.get_summary(),
        "infer_count": infer_count,
        "skip_count": skip_count,
        "erd_empty_count": erd_empty_count,
        "predict_count": predict_count,
        "erd_infer_count": erd_infer_count,
        "augmented_infer_count": augmented_infer_count,
    }


def run_fixedskip_on_sequence(seq, model, device, args, Track, skip_n,
                              tegrastats_log=None):
    """Run fixed-interval skip baseline. Returns per-frame detections and timing."""
    Track.reset_id_counter()
    detector = FixedSkipDetector(
        skip_n=skip_n,
        frame_width=seq.frame_width,
        frame_height=seq.frame_height,
        sigma_meas_px=args.sigma_meas_px,
        max_lost_frames=args.max_lost_frames,
        class_filter=EVAL_COCO_IDS,
        yolo_conf=args.yolo_conf,
        Track=Track,
        uncertainty_threshold=args.uncertainty_threshold,
    )

    tgs = ClipTegrastats(log_path=tegrastats_log)
    per_frame_dets = []
    frame_times_ms = []
    infer_count = 0
    skip_count = 0

    tgs.start()
    wall_start = time.perf_counter()

    for i in range(len(seq)):
        img, _ = seq.get_frame(i)
        t0 = time.perf_counter()
        result = detector.step(img, model, device)
        t1 = time.perf_counter()

        frame_dets = [
            {
                "box_xyxy": t.get_box_xyxy().tolist(),
                "conf":     t.get_confidence(detector.uncertainty_threshold),
                "class_id": t.class_id,
                "class_name": t.class_name,
            }
            for t in detector.tracks
            if t.state != "lost"
        ]
        per_frame_dets.append(frame_dets)
        frame_times_ms.append((t1 - t0) * 1000.0)
        if result["ran_inference"]:
            infer_count += 1
        else:
            skip_count += 1

    wall_duration = time.perf_counter() - wall_start
    tgs.stop()

    return {
        "per_frame_dets": per_frame_dets,
        "frame_times_ms": frame_times_ms,
        "wall_duration_s": wall_duration,
        "tegrastats": tgs.get_summary(),
        "infer_count": infer_count,
        "skip_count": skip_count,
        "erd_empty_count": 0,
        "predict_count": skip_count,
    }


def run_ctd_on_sequence(seq, model, device, args, Track, confidence_threshold,
                         tegrastats_log=None):
    """Run CTD (confidence-triggered) baseline for a single threshold value."""
    Track.reset_id_counter()
    detector = CTDDetector(
        confidence_threshold=confidence_threshold,
        max_consecutive_skips=args.ctd_max_skip,
        frame_width=seq.frame_width,
        frame_height=seq.frame_height,
        sigma_meas_px=args.sigma_meas_px,
        max_lost_frames=args.max_lost_frames,
        class_filter=EVAL_COCO_IDS,
        yolo_conf=args.yolo_conf,
        Track=Track,
        uncertainty_threshold=args.uncertainty_threshold,
    )

    tgs = ClipTegrastats(log_path=tegrastats_log)
    per_frame_dets = []
    frame_times_ms = []
    infer_count = 0
    skip_count = 0

    tgs.start()
    wall_start = time.perf_counter()

    for i in range(len(seq)):
        img, _ = seq.get_frame(i)
        t0 = time.perf_counter()
        result = detector.step(img, model, device)
        t1 = time.perf_counter()
        per_frame_dets.append(result["detections"])
        frame_times_ms.append((t1 - t0) * 1000.0)
        if result["ran_inference"]:
            infer_count += 1
        else:
            skip_count += 1

    wall_duration = time.perf_counter() - wall_start
    tgs.stop()

    return {
        "per_frame_dets": per_frame_dets,
        "frame_times_ms": frame_times_ms,
        "wall_duration_s": wall_duration,
        "tegrastats": tgs.get_summary(),
        "infer_count": infer_count,
        "skip_count": skip_count,
        "erd_empty_count": 0,
        "predict_count": skip_count,
    }


def run_statues_on_sequence(seq, model, device, args, component_threshold,
                             tegrastats_log=None):
    """Run Statues baseline (pixel-level frame differencing) for one threshold."""
    detector = StatuesDetector(
        pixel_threshold=args.statues_pixel_threshold,
        component_score_threshold=component_threshold,
        downscale_s=args.statues_downscale_s,
        frame_width=seq.frame_width,
        frame_height=seq.frame_height,
        class_filter=EVAL_COCO_IDS,
        yolo_conf=args.yolo_conf,
    )

    tgs = ClipTegrastats(log_path=tegrastats_log)
    per_frame_dets = []
    frame_times_ms = []
    infer_count = 0
    skip_count = 0

    tgs.start()
    wall_start = time.perf_counter()

    for i in range(len(seq)):
        img, _ = seq.get_frame(i)
        t0 = time.perf_counter()
        result = detector.step(img, model, device)
        t1 = time.perf_counter()
        per_frame_dets.append(result["detections"])
        frame_times_ms.append((t1 - t0) * 1000.0)
        if result["ran_inference"]:
            infer_count += 1
        else:
            skip_count += 1

    wall_duration = time.perf_counter() - wall_start
    tgs.stop()

    return {
        "per_frame_dets": per_frame_dets,
        "frame_times_ms": frame_times_ms,
        "wall_duration_s": wall_duration,
        "tegrastats": tgs.get_summary(),
        "infer_count": infer_count,
        "skip_count": skip_count,
        "erd_empty_count": 0,
        "predict_count": skip_count,
    }


def run_erd_only_on_sequence(seq, model, device, args, tegrastats_log=None):
    """Run ERD-only ablation: scene-level screening without KF tracking."""
    detector = ERDOnlyDetector(
        frame_width=seq.frame_width,
        frame_height=seq.frame_height,
        class_filter=EVAL_COCO_IDS,
        yolo_conf=args.yolo_conf,
    )

    tgs = ClipTegrastats(log_path=tegrastats_log)
    per_frame_dets = []
    frame_times_ms = []
    infer_count = 0
    skip_count = 0
    erd_empty_count = 0
    erd_infer_count = 0

    tgs.start()
    wall_start = time.perf_counter()

    for i in range(len(seq)):
        img, _ = seq.get_frame(i)
        t0 = time.perf_counter()
        result = detector.step(img, model, device)
        t1 = time.perf_counter()
        per_frame_dets.append(result["detections"])
        frame_times_ms.append((t1 - t0) * 1000.0)
        if result["ran_inference"]:
            infer_count += 1
            erd_infer_count += 1
        else:
            skip_count += 1
            erd_empty_count += 1

    wall_duration = time.perf_counter() - wall_start
    tgs.stop()

    return {
        "per_frame_dets": per_frame_dets,
        "frame_times_ms": frame_times_ms,
        "wall_duration_s": wall_duration,
        "tegrastats": tgs.get_summary(),
        "infer_count": infer_count,
        "skip_count": skip_count,
        "erd_empty_count": erd_empty_count,
        "predict_count": 0,
        "erd_infer_count": erd_infer_count,
    }


def evaluate_sequence(seq, detections):
    """
    Compare per-frame detections against BDD GT annotations using mAP evaluator.
    Returns a SequenceEvaluator with all frames added.
    """
    evaluator = SequenceEvaluator()
    for i in range(len(seq)):
        _, gt_boxes = seq.get_frame(i)
        preds = []
        for d in detections[i]:
            preds.append({
                "box_xyxy": np.array(d["box_xyxy"], dtype=np.float64),
                "conf": d.get("conf", 1.0),
            })
        evaluator.add_frame(i, preds, gt_boxes)
    return evaluator


def collect_metrics(evaluator):
    """Extract all needed metrics from a SequenceEvaluator."""
    ap_50, prec_arr, rec_arr = evaluator.compute_ap(iou_threshold=0.5)
    ap_50_95, _ = evaluator.compute_map()

    # Aggregate TP/FP/FN from per-frame stats
    frame_stats = evaluator.compute_per_frame_stats(iou_threshold=0.5)
    total_preds = sum(s[1] for s in frame_stats)
    total_tp    = sum(s[3] for s in frame_stats)
    total_fp    = sum(s[4] for s in frame_stats)
    total_fn    = sum(s[5] for s in frame_stats)

    avg_precision = float(np.mean(prec_arr)) if len(prec_arr) else 0.0
    avg_recall    = float(np.mean(rec_arr))  if len(rec_arr)  else 0.0

    return {
        "ap_50":        ap_50,
        "ap_50_95":     ap_50_95,
        "avg_precision": avg_precision,
        "avg_recall":   avg_recall,
        "total_gt":     evaluator.total_gt,
        "total_preds":  total_preds,
        "total_tp":     total_tp,
        "total_fp":     total_fp,
        "total_fn":     total_fn,
        "pr_precision": prec_arr.tolist() if hasattr(prec_arr, "tolist") else list(prec_arr),
        "pr_recall":    rec_arr.tolist()  if hasattr(rec_arr,  "tolist") else list(rec_arr),
        "per_frame_stats": frame_stats,
    }


# ==============================================================================
# Output: CSVs
# ==============================================================================

def write_csvs(all_results, results_dir):
    """Write per-sequence and summary CSVs."""
    summary_path = results_dir / "summary.csv"
    fields = [
        "sequence", "system", "ap_50", "ap_50_95",
        "avg_precision", "avg_recall",
        "total_gt", "total_preds", "total_tp", "total_fp", "total_fn",
        "infer_count", "skip_count", "skip_rate_pct",
        "erd_empty_count", "predict_count",
        "erd_empty_rate_pct", "predict_rate_pct",
        "erd_infer_count", "erd_infer_rate_pct",
        "augmented_infer_count", "augmented_infer_rate_pct",
        "pure_infer_count", "pure_infer_rate_pct",
        "wall_duration_s", "avg_frame_time_ms",
        "avg_power_w", "max_power_w", "energy_j",
        "avg_power_gpu_w", "energy_gpu_j",
        "avg_power_cpu_w", "energy_cpu_j",
        "avg_power_io_w", "energy_io_j",
        "avg_gpu_temp_c", "max_gpu_temp_c",
        "avg_cpu_temp_c", "max_cpu_temp_c", "max_tj_temp_c",
        "avg_gpu_util_pct", "avg_cpu_util_pct", "avg_emc_util_pct",
        "avg_ram_used_mb", "max_ram_used_mb",
        "n_tegrastats_samples",
    ]
    rows = []
    for res in all_results:
        rows.append({k: res[k] for k in fields})

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {summary_path}")

    for res in all_results:
        if "per_frame_stats" in res:
            path = (results_dir
                    / f"{res['sequence']}_{res['system'].lower()}_per_frame.csv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_idx", "n_preds", "n_gt", "tp", "fp", "fn"])
                for row in res["per_frame_stats"]:
                    writer.writerow(row)

        # Save per-frame predictions with class info for per-class analysis
        if "per_frame_dets_raw" in res:
            preds_path = (results_dir
                          / f"{res['sequence']}_{res['system'].lower()}_preds.json.gz")
            with gzip.open(preds_path, "wt") as f:
                json.dump(res["per_frame_dets_raw"], f)


# ==============================================================================
# Output: Plots
# ==============================================================================

def produce_plots(all_results, output_dir):
    """Generate PR curves and summary bar charts."""
    _COLORS = ["red", "blue", "green", "orange", "purple", "brown", "cyan", "magenta"]

    sequences = {}
    for r in all_results:
        sn = r["sequence"]
        if sn not in sequences:
            sequences[sn] = {}
        sequences[sn][r["system"]] = r

    all_systems = list(dict.fromkeys(r["system"] for r in all_results))
    sys_colors = {s: _COLORS[i % len(_COLORS)] for i, s in enumerate(all_systems)}

    # PR curves per sequence
    for seq_name, sys_data in sequences.items():
        if len(sys_data) < 2:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        for sname in all_systems:
            if sname not in sys_data:
                continue
            r = sys_data[sname]
            if len(r.get("pr_precision", [])) > 0:
                ax.plot(r["pr_recall"], r["pr_precision"], color=sys_colors[sname],
                        linewidth=1.5, label=f"{sname} (AP@50={r['ap_50']:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve: {seq_name}")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = output_dir / f"pr_curve_{seq_name}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  Saved {path}")

    # AP@50 comparison bar
    seq_names = sorted(sequences.keys())
    n_sys = len(all_systems)
    x = np.arange(len(seq_names))
    w = 0.8 / max(n_sys, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, len(seq_names) * 1.5), 5))

    for si, sname in enumerate(all_systems):
        ap50_vals = [sequences[s].get(sname, {}).get("ap_50", 0) for s in seq_names]
        map_vals  = [sequences[s].get(sname, {}).get("ap_50_95", 0) for s in seq_names]
        offset = (si - n_sys / 2 + 0.5) * w
        ax1.bar(x + offset, ap50_vals, w, color=sys_colors[sname], label=sname)
        ax2.bar(x + offset, map_vals, w, color=sys_colors[sname], label=sname)

    ax1.set_xticks(x)
    ax1.set_xticklabels(seq_names, rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("AP@0.50")
    ax1.set_title("AP@0.50 Comparison")
    ax1.legend(fontsize=7); ax1.set_ylim([0, 1])

    ax2.set_xticks(x)
    ax2.set_xticklabels(seq_names, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("mAP@0.50:0.95")
    ax2.set_title("mAP@0.50:0.95 Comparison")
    ax2.legend(fontsize=7); ax2.set_ylim([0, 1])

    fig.tight_layout()
    path = output_dir / "map_comparison.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved {path}")

    # Skip rate + energy summary
    skip_systems = [s for s in all_systems if s != "Vanilla"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    for sname in skip_systems:
        rates = [r["skip_rate_pct"] for r in all_results if r["system"] == sname]
        if rates:
            ax1.hist(rates, bins=20, color=sys_colors[sname], alpha=0.5, label=sname)
            ax1.axvline(np.mean(rates), color=sys_colors[sname], linestyle="--",
                        label=f"{sname} mean: {np.mean(rates):.1f}%")
    ax1.set_xlabel("Skip rate (%)")
    ax1.set_ylabel("Clip count")
    ax1.set_title("Skip Rate Distribution")
    ax1.legend(fontsize=7)

    energies_by_sys = {}
    for sname in all_systems:
        vals = [r["energy_j"] for r in all_results
                if r["system"] == sname and r["energy_j"] is not None]
        if vals:
            energies_by_sys[sname] = np.mean(vals)
    if energies_by_sys:
        names = list(energies_by_sys.keys())
        vals = [energies_by_sys[n] for n in names]
        colors = [sys_colors[n] for n in names]
        ax2.bar(names, vals, color=colors)
        ax2.set_ylabel("Avg energy per clip (J)")
        ax2.set_title("Energy per Clip")
        ax2.tick_params(axis="x", rotation=30)
    else:
        ax2.text(0.5, 0.5, "No power data", ha="center", va="center",
                 transform=ax2.transAxes)

    fig.tight_layout()
    path = output_dir / "summary_bar.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved {path}")


# ==============================================================================
# Console summary
# ==============================================================================

def print_summary(all_results, args):
    systems = list(dict.fromkeys(r["system"] for r in all_results))
    by_sys = {s: [r for r in all_results if r["system"] == s] for s in systems}
    n_clips = max(len(v) for v in by_sys.values())

    def safe_mean(lst, key, default=None):
        vals = [r[key] for r in lst if r[key] is not None]
        return np.mean(vals) if vals else default

    col_w = 14
    header_line = f"{'Metric':<40}" + "".join(f"{s:>{col_w}}" for s in systems)
    sep = "-" * (40 + col_w * len(systems))

    print("\n")
    print("=" * len(sep))
    print(f"  Albireo Evaluation Summary  (model: {args.model}, "
          f"clips: {n_clips}, platform: {args.platform})")
    print("=" * len(sep))
    print(header_line)
    print(sep)

    def _row(label, key, fmt=".4f", default=0):
        vals = []
        for s in systems:
            v = safe_mean(by_sys[s], key, default)
            vals.append(f"{v:{fmt}}" if v is not None else "N/A")
        print(f"{label:<40}" + "".join(f"{v:>{col_w}}" for v in vals))

    counts_line = f"{'Clips evaluated':<40}" + "".join(
        f"{len(by_sys[s]):>{col_w}}" for s in systems)
    print(counts_line)

    _row("Avg AP@0.50", "ap_50", fmt=".4f")
    _row("Avg mAP@0.50:0.95", "ap_50_95", fmt=".4f")
    _row("Avg skip rate (%)", "skip_rate_pct", fmt=".1f")

    adp = by_sys.get("Adaptive", [])
    if adp:
        erd_rates = [r.get("erd_empty_rate_pct", 0) for r in adp]
        pred_rates = [r.get("predict_rate_pct", 0) for r in adp]
        erd_infer_rates = [r.get("erd_infer_rate_pct", 0) for r in adp]
        aug_infer_rates = [r.get("augmented_infer_rate_pct", 0) for r in adp]
        pure_infer_rates = [r.get("pure_infer_rate_pct", 0) for r in adp]
        adp_idx = systems.index("Adaptive")
        pad_before = "".join(f"{'---':>{col_w}}" for _ in range(adp_idx))
        pad_after = "".join(f"{'---':>{col_w}}" for _ in range(adp_idx + 1, len(systems)))
        def _adp_row(label, rates):
            s = f"{np.mean(rates):.1f}+/-{np.std(rates):.1f}"
            print(f"{label:<40}{pad_before}{s:>{col_w}}{pad_after}")
        _adp_row("  Skip: E  — ERD empty (%)", erd_rates)
        _adp_row("  Skip: P  — KF predict (%)", pred_rates)
        _adp_row("  Infer: I  — pure inference (%)", pure_infer_rates)
        _adp_row("  Infer: E+I — ERD→inference (%)", erd_infer_rates)
        _adp_row("  Infer: AI — augmented/rescue (%)", aug_infer_rates)

    _row("Avg frame time (ms)", "avg_frame_time_ms", fmt=".2f")
    print(sep)
    _row("Avg power (W)", "avg_power_w", fmt=".2f")
    _row("  GPU power (W)", "avg_power_gpu_w", fmt=".2f")
    _row("  CPU power (W)", "avg_power_cpu_w", fmt=".2f")
    _row("  IO power (W)", "avg_power_io_w", fmt=".2f")
    _row("Avg energy/clip (J)", "energy_j", fmt=".1f")
    _row("  GPU energy (J)", "energy_gpu_j", fmt=".1f")
    _row("  CPU energy (J)", "energy_cpu_j", fmt=".1f")
    _row("  IO energy (J)", "energy_io_j", fmt=".1f")

    van_e = safe_mean(by_sys.get("Vanilla", []), "energy_j")
    if van_e is not None and van_e > 0:
        savings_line = f"{'Energy saving vs Vanilla (%)':<40}"
        for s in systems:
            e = safe_mean(by_sys[s], "energy_j")
            if e is not None and s != "Vanilla":
                saving = 100.0 * (van_e - e) / van_e
                savings_line += f"{saving:>{col_w}.1f}"
            else:
                savings_line += f"{'---':>{col_w}}"
        print(savings_line)

    _row("Avg GPU temp (°C)", "avg_gpu_temp_c", fmt=".1f")
    _row("Max GPU temp (°C)", "max_gpu_temp_c", fmt=".1f")
    _row("Max Tj temp (°C)", "max_tj_temp_c", fmt=".1f")
    _row("Avg GPU util (%)", "avg_gpu_util_pct", fmt=".1f")
    _row("Avg CPU util (%)", "avg_cpu_util_pct", fmt=".1f")
    _row("Avg EMC util (%)", "avg_emc_util_pct", fmt=".1f")

    print("=" * len(sep))
    print()


# ==============================================================================
# Main
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Albireo Experiment 3: Detection accuracy (mAP) on BDD100K MOT"
    )
    parser.add_argument(
        "--dataset", default="bdd", choices=["bdd", "virat"],
        help="Which dataset to evaluate (default: bdd). VIRAT requires --virat-root.",
    )
    parser.add_argument(
        "--bdd-root", default=None,
        help="Path to BDD100K root directory (contains images/ and labels/). "
             "Required when --dataset bdd.",
    )
    parser.add_argument(
        "--virat-root", default=None,
        help="Path to VIRAT root directory (contains videos/ and annotations/). "
             "Required when --dataset virat.",
    )
    parser.add_argument(
        "--sequences", default=None,
        help="Comma-separated video names to run (default: random sample of --num-clips)",
    )
    parser.add_argument(
        "--num-clips", type=int, default=30,
        help="Number of clips to evaluate (random sample from val set, default: 30)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for clip sampling",
    )
    parser.add_argument(
        "--model", default="yolo11x",
        help="YOLO model stem (e.g. yolo11n, yolo11x)",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Results directory. Defaults to results/{platform}/albireo/{model}/",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output plots directory. Defaults to output/{platform}/albireo/{model}/",
    )
    parser.add_argument(
        "--platform", default="xavier", choices=["xavier", "orin", "thor", "polaris"],
        help="Target platform (for results organization)",
    )
    parser.add_argument(
        "--uncertainty-threshold", type=float, default=3e-4,
        help="Kalman covariance trace threshold (normalised fraction²) to trigger YOLO.",
    )
    parser.add_argument(
        "--sigma-meas-px", type=float, default=5.0,
        help="Measurement noise in pixels; normalised by YOLO imgsz at runtime.",
    )
    parser.add_argument(
        "--max-skip", type=int, default=15,
        help="Hard cap on frames between YOLO runs per track (default: 15)",
    )
    parser.add_argument(
        "--max-lost-frames", type=int, default=10,
        help="Frames unmatched before a confirmed track is marked lost (default: 10)",
    )
    parser.add_argument(
        "--min-infer-every", type=int, default=30,
        help="Global floor: always run YOLO at least every N frames",
    )
    parser.add_argument(
        "--split", default="val", choices=["val", "train"],
        help="BDD100K split to evaluate (default: val)",
    )
    parser.add_argument(
        "--yolo-conf", type=float, default=0.25,
        help="YOLO confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--rescue-conf", type=float, default=0.5,
        help="Idea 7: min last-detection conf for KF rescue (default: 0.5)",
    )
    parser.add_argument(
        "--max-rescue", type=int, default=2,
        help="Idea 7: max consecutive I-frame misses rescued by KF (default: 2)",
    )
    parser.add_argument(
        "--mode", default="vanilla+adaptive",
        choices=["vanilla+adaptive", "adaptive", "fixedskip", "all",
                 "ctd", "statues", "erd_only", "baselines"],
        help="Which systems to run. 'adaptive' skips vanilla (use when vanilla "
             "results for this model already exist). 'baselines' = ctd + "
             "statues + erd_only.",
    )
    parser.add_argument(
        "--skip-n", type=int, nargs="+", default=[2, 3, 5],
        help="Fixed-skip intervals to test (default: 2 3 5)",
    )
    # --- CTD (Confidence Trigger Detection, arXiv:1902.00615) ---
    parser.add_argument(
        "--ctd-thresholds", type=float, nargs="+",
        default=[0.30, 0.50, 0.70, 0.90],
        help="CTD confidence thresholds to sweep. Paper recommends ~0.30 and "
             "evaluates 0.10/0.90 in Table II. Default: 0.30 0.50 0.70 0.90.",
    )
    parser.add_argument(
        "--ctd-max-skip", type=int, default=8,
        help="CTD max consecutive skips cap (paper Table II: 8).",
    )
    # --- Statues (ISLPED 2024) ---
    parser.add_argument(
        "--statues-component-thresholds", type=int, nargs="+",
        default=[25, 75, 200, 500],
        help="Statues component-score thresholds to sweep (paper default: 75).",
    )
    parser.add_argument(
        "--statues-pixel-threshold", type=int, default=30,
        help="Statues pixel-difference binarization threshold. Paper does not "
             "specify a value; 30 is standard for 8-bit grayscale frame diff.",
    )
    parser.add_argument(
        "--statues-downscale-s", type=int, default=2,
        help="Statues s x s average-pool downscaling (paper: 2).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve default paths using platform/<system>/model hierarchy.
    # Each mode writes under its own subdirectory so nothing collides.
    exp_dir = pathlib.Path(__file__).parent.parent
    if args.mode == "fixedskip":
        default_subdir = "fixedskip"
    elif args.mode == "ctd":
        default_subdir = "ctd"
    elif args.mode == "statues":
        default_subdir = "statues"
    elif args.mode == "erd_only":
        default_subdir = "erd_only"
    elif args.mode == "baselines":
        default_subdir = "baselines"
    else:
        default_subdir = "albireo"
    if args.results_dir is None:
        args.results_dir = str(exp_dir / "results" / args.platform
                                / default_subdir / args.model)
    if args.output_dir is None:
        args.output_dir = str(exp_dir / "output" / args.platform
                               / default_subdir / args.model)

    results_dir = pathlib.Path(args.results_dir)
    output_dir  = pathlib.Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} | Platform: {args.platform}")

    # Load model
    if args.model.startswith("rfdetr"):
        from rfdetr_wrapper import RFDETRModel
        print(f"Loading model: {args.model} (RF-DETR)")
        try:
            model = RFDETRModel(args.model)
        except Exception as e:
            print(f"[ERROR] Failed to load RF-DETR model: {e}")
            sys.exit(1)
        print(f"Model loaded ({args.model})")
    else:
        print(f"Loading model: {args.model}.pt")
        try:
            model = YOLO(f"{args.model}.pt").to(device)
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            sys.exit(1)
        print(f"Model loaded on {model.device}")

    # Load dataset (BDD100K or VIRAT-Ground)
    global EVAL_COCO_IDS
    if args.dataset == "virat":
        if not args.virat_root:
            print("[ERROR] --virat-root is required when --dataset virat")
            sys.exit(1)
        from virat_loader import VIRATDataset, DEFAULT_EVAL_COCO_IDS as VIRAT_EVAL_IDS
        EVAL_COCO_IDS = VIRAT_EVAL_IDS
        print(f"\nVIRAT root: {args.virat_root}")
        try:
            dataset = VIRATDataset(args.virat_root)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        all_videos = dataset.get_sequence_names()
        print(f"Discovered {len(all_videos)} VIRAT clips")
    else:
        if not args.bdd_root:
            print("[ERROR] --bdd-root is required when --dataset bdd")
            sys.exit(1)
        print(f"\nBDD100K root: {args.bdd_root}")
        try:
            dataset = BDDDataset(
                args.bdd_root,
                split=args.split,
            )
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            sys.exit(1)
        all_videos = dataset.get_sequence_names()
        print(f"Discovered {len(all_videos)} {args.split} clips")

    # Select sequences
    if args.sequences:
        selected = [s.strip() for s in args.sequences.split(",")]
        missing = [s for s in selected if s not in all_videos]
        if missing:
            print(f"[WARNING] Sequences not found and will be skipped: {missing}")
        selected = [s for s in selected if s in all_videos]
    else:
        rng = random.Random(args.seed)
        n = min(args.num_clips, len(all_videos))
        selected = rng.sample(all_videos, n)
        selected.sort()

    print(f"Running on {len(selected)} clips: {selected[:5]}{'...' if len(selected)>5 else ''}")

    # Evaluate classes
    if args.dataset == "bdd":
        print(f"Eval classes: {sorted(DEFAULT_EVAL_CLASSES)}")
    print(f"Eval COCO IDs: {sorted(EVAL_COCO_IDS)}")

    run_vanilla = args.mode in ("vanilla+adaptive", "all")
    run_adaptive = args.mode in ("vanilla+adaptive", "adaptive", "all")
    run_fixedskip = args.mode in ("fixedskip", "all")
    run_ctd = args.mode in ("ctd", "baselines")
    run_statues = args.mode in ("statues", "baselines")
    run_erd_only = args.mode in ("erd_only", "baselines")

    all_results = []
    tgs_dir = results_dir / "tegrastats"

    for clip_i, video_name in enumerate(selected):
        print(f"\n[{clip_i+1}/{len(selected)}] {video_name}")

        try:
            seq = dataset.get_sequence(video_name)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        print(f"  Frames: {len(seq)} | Size: {seq.frame_width}×{seq.frame_height}")
        n_frames = max(len(seq), 1)

        # --- Vanilla ---
        if run_vanilla:
            print(f"  Running vanilla YOLO...")
            van_run = run_vanilla_on_sequence(
                seq, model, device, yolo_conf=args.yolo_conf,
                tegrastats_log=tgs_dir / f"{video_name}_vanilla.log")
            van_eval = evaluate_sequence(seq, van_run["per_frame_dets"])
            van_metrics = collect_metrics(van_eval)

            van_tgs = van_run["tegrastats"]
            van_ft = float(np.mean(van_run["frame_times_ms"]))

            print(f"  Vanilla: AP@50={van_metrics['ap_50']:.4f}  "
                  f"mAP={van_metrics['ap_50_95']:.4f}  "
                  f"t={van_ft:.1f}ms/frame"
                  + (f"  E={van_tgs['energy_j']:.1f}J" if van_tgs["energy_j"] else ""))

            van_result = {
                "sequence":         video_name,
                "system":           "Vanilla",
                "ap_50":            van_metrics["ap_50"],
                "ap_50_95":         van_metrics["ap_50_95"],
                "avg_precision":    van_metrics["avg_precision"],
                "avg_recall":       van_metrics["avg_recall"],
                "total_gt":         van_metrics["total_gt"],
                "total_preds":      van_metrics["total_preds"],
                "total_tp":         van_metrics["total_tp"],
                "total_fp":         van_metrics["total_fp"],
                "total_fn":         van_metrics["total_fn"],
                "infer_count":      van_run["infer_count"],
                "skip_count":       0,
                "skip_rate_pct":    0.0,
                "erd_empty_count":  0,
                "predict_count":    0,
                "erd_empty_rate_pct": 0.0,
                "predict_rate_pct": 0.0,
                "erd_infer_count":  0,
                "erd_infer_rate_pct": 0.0,
                "augmented_infer_count": 0,
                "augmented_infer_rate_pct": 0.0,
                "pure_infer_count": van_run["infer_count"],
                "pure_infer_rate_pct": 100.0,
                "wall_duration_s":  round(van_run["wall_duration_s"], 2),
                "avg_frame_time_ms": round(van_ft, 3),
                "pr_precision":     van_metrics.get("pr_precision", []),
                "pr_recall":        van_metrics.get("pr_recall", []),
                "per_frame_stats":  van_metrics.get("per_frame_stats", []),
                "per_frame_dets_raw": van_run["per_frame_dets"],
            }
            van_result.update(van_tgs)
            all_results.append(van_result)

        # --- Albireo adaptive detector ---
        if run_adaptive:
            print("  Running Albireo adaptive detector...")
            adp_run = run_adaptive_on_sequence(
                seq, model, device, args, AdaptiveDetector, Track,
                tegrastats_log=tgs_dir / f"{video_name}_adaptive.log")
            adp_eval = evaluate_sequence(seq, adp_run["per_frame_dets"])
            adp_metrics = collect_metrics(adp_eval)

            adp_tgs = adp_run["tegrastats"]
            skip_rate = 100.0 * adp_run["skip_count"] / n_frames
            adp_ft = float(np.mean(adp_run["frame_times_ms"]))

            erd_empty = adp_run.get("erd_empty_count", 0)
            predict   = adp_run.get("predict_count", 0)
            erd_infer = adp_run.get("erd_infer_count", 0)
            aug_infer = adp_run.get("augmented_infer_count", 0)
            pure_infer = adp_run["infer_count"] - erd_infer - aug_infer
            erd_empty_rate = 100.0 * erd_empty / n_frames
            predict_rate   = 100.0 * predict / n_frames
            erd_infer_rate = 100.0 * erd_infer / n_frames
            aug_infer_rate = 100.0 * aug_infer / n_frames
            pure_infer_rate = 100.0 * pure_infer / n_frames

            print(f"  Adaptive: AP@50={adp_metrics['ap_50']:.4f}  "
                  f"mAP={adp_metrics['ap_50_95']:.4f}  "
                  f"skip={skip_rate:.1f}% (E={erd_empty_rate:.1f}% P={predict_rate:.1f}%)  "
                  f"infer (I={pure_infer_rate:.1f}% E+I={erd_infer_rate:.1f}% AI={aug_infer_rate:.1f}%)  "
                  f"t={adp_ft:.1f}ms/frame"
                  + (f"  E={adp_tgs['energy_j']:.1f}J" if adp_tgs["energy_j"] else ""))

            adp_result = {
                "sequence":         video_name,
                "system":           "Adaptive",
                "ap_50":            adp_metrics["ap_50"],
                "ap_50_95":         adp_metrics["ap_50_95"],
                "avg_precision":    adp_metrics["avg_precision"],
                "avg_recall":       adp_metrics["avg_recall"],
                "total_gt":         adp_metrics["total_gt"],
                "total_preds":      adp_metrics["total_preds"],
                "total_tp":         adp_metrics["total_tp"],
                "total_fp":         adp_metrics["total_fp"],
                "total_fn":         adp_metrics["total_fn"],
                "infer_count":      adp_run["infer_count"],
                "skip_count":       adp_run["skip_count"],
                "skip_rate_pct":    round(skip_rate, 2),
                "erd_empty_count":  erd_empty,
                "predict_count":    predict,
                "erd_empty_rate_pct": round(erd_empty_rate, 2),
                "predict_rate_pct": round(predict_rate, 2),
                "erd_infer_count":  erd_infer,
                "erd_infer_rate_pct": round(erd_infer_rate, 2),
                "augmented_infer_count": aug_infer,
                "augmented_infer_rate_pct": round(aug_infer_rate, 2),
                "pure_infer_count": pure_infer,
                "pure_infer_rate_pct": round(pure_infer_rate, 2),
                "wall_duration_s":  round(adp_run["wall_duration_s"], 2),
                "avg_frame_time_ms": round(adp_ft, 3),
                "pr_precision":     adp_metrics.get("pr_precision", []),
                "pr_recall":        adp_metrics.get("pr_recall", []),
                "per_frame_stats":  adp_metrics.get("per_frame_stats", []),
                "per_frame_dets_raw": adp_run["per_frame_dets"],
            }
            adp_result.update(adp_tgs)
            all_results.append(adp_result)

        # --- Fixed-skip baselines ---
        if run_fixedskip:
            for skip_n in args.skip_n:
                system_label = f"FixedSkip-{skip_n}"
                print(f"  Running {system_label}...")
                fs_run = run_fixedskip_on_sequence(
                    seq, model, device, args, Track, skip_n,
                    tegrastats_log=tgs_dir / f"{video_name}_fixedskip{skip_n}.log")
                fs_eval = evaluate_sequence(seq, fs_run["per_frame_dets"])
                fs_metrics = collect_metrics(fs_eval)

                fs_tgs = fs_run["tegrastats"]
                fs_skip_rate = 100.0 * fs_run["skip_count"] / n_frames
                fs_ft = float(np.mean(fs_run["frame_times_ms"]))

                print(f"  {system_label}: AP@50={fs_metrics['ap_50']:.4f}  "
                      f"mAP={fs_metrics['ap_50_95']:.4f}  "
                      f"skip={fs_skip_rate:.1f}%  t={fs_ft:.1f}ms/frame"
                      + (f"  E={fs_tgs['energy_j']:.1f}J" if fs_tgs["energy_j"] else ""))

                fs_result = {
                    "sequence":         video_name,
                    "system":           system_label,
                    "ap_50":            fs_metrics["ap_50"],
                    "ap_50_95":         fs_metrics["ap_50_95"],
                    "avg_precision":    fs_metrics["avg_precision"],
                    "avg_recall":       fs_metrics["avg_recall"],
                    "total_gt":         fs_metrics["total_gt"],
                    "total_preds":      fs_metrics["total_preds"],
                    "total_tp":         fs_metrics["total_tp"],
                    "total_fp":         fs_metrics["total_fp"],
                    "total_fn":         fs_metrics["total_fn"],
                    "infer_count":      fs_run["infer_count"],
                    "skip_count":       fs_run["skip_count"],
                    "skip_rate_pct":    round(fs_skip_rate, 2),
                    "erd_empty_count":  0,
                    "predict_count":    fs_run["skip_count"],
                    "erd_empty_rate_pct": 0.0,
                    "predict_rate_pct": round(fs_skip_rate, 2),
                    "erd_infer_count":  0,
                    "erd_infer_rate_pct": 0.0,
                    "augmented_infer_count": 0,
                    "augmented_infer_rate_pct": 0.0,
                    "pure_infer_count": fs_run["infer_count"],
                    "pure_infer_rate_pct": round(100.0 * fs_run["infer_count"] / n_frames, 2),
                    "wall_duration_s":  round(fs_run["wall_duration_s"], 2),
                    "avg_frame_time_ms": round(fs_ft, 3),
                    "pr_precision":     fs_metrics.get("pr_precision", []),
                    "pr_recall":        fs_metrics.get("pr_recall", []),
                    "per_frame_stats":  fs_metrics.get("per_frame_stats", []),
                    "per_frame_dets_raw": fs_run["per_frame_dets"],
                }
                fs_result.update(fs_tgs)
                all_results.append(fs_result)

        # --- CTD baseline (sweep over confidence thresholds) ---
        if run_ctd:
            for ctd_thr in args.ctd_thresholds:
                # Compact label: CTD-0.30, CTD-0.50, ...
                system_label = f"CTD-{ctd_thr:.2f}"
                print(f"  Running {system_label}...")
                ctd_run = run_ctd_on_sequence(
                    seq, model, device, args, Track, ctd_thr,
                    tegrastats_log=tgs_dir /
                        f"{video_name}_ctd{ctd_thr:.2f}.log")
                ctd_eval = evaluate_sequence(seq, ctd_run["per_frame_dets"])
                ctd_metrics = collect_metrics(ctd_eval)

                ctd_tgs = ctd_run["tegrastats"]
                ctd_skip_rate = 100.0 * ctd_run["skip_count"] / n_frames
                ctd_ft = float(np.mean(ctd_run["frame_times_ms"]))

                print(f"  {system_label}: AP@50={ctd_metrics['ap_50']:.4f}  "
                      f"mAP={ctd_metrics['ap_50_95']:.4f}  "
                      f"skip={ctd_skip_rate:.1f}%  t={ctd_ft:.1f}ms/frame"
                      + (f"  E={ctd_tgs['energy_j']:.1f}J"
                         if ctd_tgs["energy_j"] else ""))

                ctd_result = {
                    "sequence":         video_name,
                    "system":           system_label,
                    "ap_50":            ctd_metrics["ap_50"],
                    "ap_50_95":         ctd_metrics["ap_50_95"],
                    "avg_precision":    ctd_metrics["avg_precision"],
                    "avg_recall":       ctd_metrics["avg_recall"],
                    "total_gt":         ctd_metrics["total_gt"],
                    "total_preds":      ctd_metrics["total_preds"],
                    "total_tp":         ctd_metrics["total_tp"],
                    "total_fp":         ctd_metrics["total_fp"],
                    "total_fn":         ctd_metrics["total_fn"],
                    "infer_count":      ctd_run["infer_count"],
                    "skip_count":       ctd_run["skip_count"],
                    "skip_rate_pct":    round(ctd_skip_rate, 2),
                    "erd_empty_count":  0,
                    "predict_count":    ctd_run["skip_count"],
                    "erd_empty_rate_pct": 0.0,
                    "predict_rate_pct": round(ctd_skip_rate, 2),
                    "erd_infer_count":  0,
                    "erd_infer_rate_pct": 0.0,
                    "augmented_infer_count": 0,
                    "augmented_infer_rate_pct": 0.0,
                    "pure_infer_count": ctd_run["infer_count"],
                    "pure_infer_rate_pct": round(
                        100.0 * ctd_run["infer_count"] / n_frames, 2),
                    "wall_duration_s":  round(ctd_run["wall_duration_s"], 2),
                    "avg_frame_time_ms": round(ctd_ft, 3),
                    "pr_precision":     ctd_metrics.get("pr_precision", []),
                    "pr_recall":        ctd_metrics.get("pr_recall", []),
                    "per_frame_stats":  ctd_metrics.get("per_frame_stats", []),
                    "per_frame_dets_raw": ctd_run["per_frame_dets"],
                }
                ctd_result.update(ctd_tgs)
                all_results.append(ctd_result)

        # --- Statues baseline (sweep over component-score thresholds) ---
        if run_statues:
            for st_thr in args.statues_component_thresholds:
                system_label = f"Statues-{st_thr}"
                print(f"  Running {system_label}...")
                st_run = run_statues_on_sequence(
                    seq, model, device, args, st_thr,
                    tegrastats_log=tgs_dir /
                        f"{video_name}_statues{st_thr}.log")
                st_eval = evaluate_sequence(seq, st_run["per_frame_dets"])
                st_metrics = collect_metrics(st_eval)

                st_tgs = st_run["tegrastats"]
                st_skip_rate = 100.0 * st_run["skip_count"] / n_frames
                st_ft = float(np.mean(st_run["frame_times_ms"]))

                print(f"  {system_label}: AP@50={st_metrics['ap_50']:.4f}  "
                      f"mAP={st_metrics['ap_50_95']:.4f}  "
                      f"skip={st_skip_rate:.1f}%  t={st_ft:.1f}ms/frame"
                      + (f"  E={st_tgs['energy_j']:.1f}J"
                         if st_tgs["energy_j"] else ""))

                st_result = {
                    "sequence":         video_name,
                    "system":           system_label,
                    "ap_50":            st_metrics["ap_50"],
                    "ap_50_95":         st_metrics["ap_50_95"],
                    "avg_precision":    st_metrics["avg_precision"],
                    "avg_recall":       st_metrics["avg_recall"],
                    "total_gt":         st_metrics["total_gt"],
                    "total_preds":      st_metrics["total_preds"],
                    "total_tp":         st_metrics["total_tp"],
                    "total_fp":         st_metrics["total_fp"],
                    "total_fn":         st_metrics["total_fn"],
                    "infer_count":      st_run["infer_count"],
                    "skip_count":       st_run["skip_count"],
                    "skip_rate_pct":    round(st_skip_rate, 2),
                    "erd_empty_count":  0,
                    "predict_count":    st_run["skip_count"],
                    "erd_empty_rate_pct": 0.0,
                    "predict_rate_pct": round(st_skip_rate, 2),
                    "erd_infer_count":  0,
                    "erd_infer_rate_pct": 0.0,
                    "augmented_infer_count": 0,
                    "augmented_infer_rate_pct": 0.0,
                    "pure_infer_count": st_run["infer_count"],
                    "pure_infer_rate_pct": round(
                        100.0 * st_run["infer_count"] / n_frames, 2),
                    "wall_duration_s":  round(st_run["wall_duration_s"], 2),
                    "avg_frame_time_ms": round(st_ft, 3),
                    "pr_precision":     st_metrics.get("pr_precision", []),
                    "pr_recall":        st_metrics.get("pr_recall", []),
                    "per_frame_stats":  st_metrics.get("per_frame_stats", []),
                    "per_frame_dets_raw": st_run["per_frame_dets"],
                }
                st_result.update(st_tgs)
                all_results.append(st_result)

        # --- ERD-only ablation baseline ---
        if run_erd_only:
            system_label = "ERD-only"
            print(f"  Running {system_label}...")
            er_run = run_erd_only_on_sequence(
                seq, model, device, args,
                tegrastats_log=tgs_dir / f"{video_name}_erdonly.log")
            er_eval = evaluate_sequence(seq, er_run["per_frame_dets"])
            er_metrics = collect_metrics(er_eval)

            er_tgs = er_run["tegrastats"]
            er_skip_rate = 100.0 * er_run["skip_count"] / n_frames
            er_ft = float(np.mean(er_run["frame_times_ms"]))
            er_erd_empty_rate = 100.0 * er_run["erd_empty_count"] / n_frames
            er_erd_infer_rate = 100.0 * er_run["erd_infer_count"] / n_frames

            print(f"  {system_label}: AP@50={er_metrics['ap_50']:.4f}  "
                  f"mAP={er_metrics['ap_50_95']:.4f}  "
                  f"skip={er_skip_rate:.1f}% (E={er_erd_empty_rate:.1f}%)  "
                  f"t={er_ft:.1f}ms/frame"
                  + (f"  E={er_tgs['energy_j']:.1f}J"
                     if er_tgs["energy_j"] else ""))

            er_result = {
                "sequence":         video_name,
                "system":           system_label,
                "ap_50":            er_metrics["ap_50"],
                "ap_50_95":         er_metrics["ap_50_95"],
                "avg_precision":    er_metrics["avg_precision"],
                "avg_recall":       er_metrics["avg_recall"],
                "total_gt":         er_metrics["total_gt"],
                "total_preds":      er_metrics["total_preds"],
                "total_tp":         er_metrics["total_tp"],
                "total_fp":         er_metrics["total_fp"],
                "total_fn":         er_metrics["total_fn"],
                "infer_count":      er_run["infer_count"],
                "skip_count":       er_run["skip_count"],
                "skip_rate_pct":    round(er_skip_rate, 2),
                "erd_empty_count":  er_run["erd_empty_count"],
                "predict_count":    0,
                "erd_empty_rate_pct": round(er_erd_empty_rate, 2),
                "predict_rate_pct": 0.0,
                "erd_infer_count":  er_run["erd_infer_count"],
                "erd_infer_rate_pct": round(er_erd_infer_rate, 2),
                "augmented_infer_count": 0,
                "augmented_infer_rate_pct": 0.0,
                "pure_infer_count": 0,
                "pure_infer_rate_pct": 0.0,
                "wall_duration_s":  round(er_run["wall_duration_s"], 2),
                "avg_frame_time_ms": round(er_ft, 3),
                "pr_precision":     er_metrics.get("pr_precision", []),
                "pr_recall":        er_metrics.get("pr_recall", []),
                "per_frame_stats":  er_metrics.get("per_frame_stats", []),
                "per_frame_dets_raw": er_run["per_frame_dets"],
            }
            er_result.update(er_tgs)
            all_results.append(er_result)

    if not all_results:
        print("[ERROR] No sequences were processed.")
        sys.exit(1)

    print(f"\nWriting outputs to {results_dir} and {output_dir}...")
    write_csvs(all_results, results_dir)
    produce_plots(all_results, output_dir)
    print_summary(all_results, args)


if __name__ == "__main__":
    main()
