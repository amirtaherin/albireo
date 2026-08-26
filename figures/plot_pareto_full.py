"""
Script Name: plot_pareto_full.py
Description: Comprehensive Pareto plots (Energy / EDP / ED^2P / Latency vs
             {AP@50, mAP, F1, F1_best, AR}) covering Vanilla, Albireo
             (Albireo default + uncertainty sweep), FixedSkip, CTD, Statues,
             and ERD-only on Thor and Orin. Also prints per-platform summary
             tables and Pareto frontiers.

             Style is unified with plot_results.py — both scripts share the
             PERSIAN palette and setup_style() so figures match. This script
             can be run standalone (`python plot_pareto_full.py`) or driven
             by plot_results.py through generate_all_pareto_plots().

             Figure-level titles are intentionally omitted; the SEC paper
             uses LaTeX captions instead. Subplot titles (Thor / Orin) are
             retained.

Author: Amir Taherin
Email: taherin.a@northeastern.edu
Date Created: 2026-04-26
Last Modified: 2026-04-28
Version: 2.1  (unified style, Albireo rename, suptitles dropped)

License: MIT License

Usage:
    python plot_pareto_full.py
    # or, from plot_results.py:
    from plot_pareto_full import generate_all_pareto_plots
    generate_all_pareto_plots()
"""

import csv
import pathlib
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# ---------------------------------------------------------------------------
# Shared style — single source of truth in plot_results.py
# ---------------------------------------------------------------------------
from plot_results import (
    PERSIAN,
    FAMILY_COLOR_KEYS,
    PLATFORM_LABELS,
    setup_style,
    RESULTS_DIR,
    OUT_DIR,
)

# Override label dict to use the longer "Jetson AGX ..." names for the
# headline Pareto figures. plot_results.py uses shorter labels for the
# small/grouped subplots.
PARETO_PLATFORM_LABELS = {"thor": "Jetson AGX Thor",
                          "orin": "Jetson AGX Orin"}

PLATFORMS = ["thor", "orin"]

# Resolve family colors against the shared PERSIAN palette.
COLOR = {fam: PERSIAN[key] for fam, key in FAMILY_COLOR_KEYS.items()}

# Pre-computed best-F1 per system (sweep over confidence thresholds, computed
# 2026-04-26 from cached *_preds.json.gz files). Hardware-independent — same
# value for Thor and Orin since the predictions are deterministic.
BEST_F1_CSV = (pathlib.Path(__file__).resolve().parents[3]
               / "notes" / "data" / "best_f1_all_systems_2026-04-26.csv")
BEST_F1_MAP = {}
if BEST_F1_CSV.exists():
    with open(BEST_F1_CSV) as _f:
        for _r in csv.DictReader(_f):
            try:
                BEST_F1_MAP[_r["system"]] = float(_r["best_F1"])
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Systems table — yolo26x is the primary detector with full sweep coverage.
# To generate plots for yolo11x or rfdetr-large, pass detector= to the
# generation functions; the path templates use {detector} as a placeholder.
#
# Format: (label, results_path_template, system_csv_filter, family, marker)
# ---------------------------------------------------------------------------
SYSTEMS = [
    # --- Vanilla reference ---
    ("Vanilla",                                "{plat}/albireo/{detector}/summary.csv",                   "Vanilla",     "vanilla",   "s"),
    # --- Albireo: default + uncertainty-threshold sweep.
    # Labels render as small-caps "Albireo" because the paper defines
    # \Albireo = \textsc{Albireo}. We mirror that here for visual consistency.
    (r"\textsc{Albireo}",                      "{plat}/albireo/{detector}/summary.csv",                   "Adaptive",    "albireo",   "*"),
    (r"\textsc{Albireo} $u=5\!\times\!10^{-4}$", "{plat}/albireo_sweep/u5e-4/{detector}/summary.csv",     "Adaptive",    "albireo",   "*"),
    (r"\textsc{Albireo} $u=7\!\times\!10^{-4}$", "{plat}/albireo_sweep/u7e-4/{detector}/summary.csv",     "Adaptive",    "albireo",   "*"),
    (r"\textsc{Albireo} $u=1\!\times\!10^{-3}$", "{plat}/albireo_sweep/u1e-3/{detector}/summary.csv",     "Adaptive",    "albireo",   "*"),
    # u=3e-3 excluded from Pareto figures: its AP@50 = 0.4615 falls below the
    # paper's safety-relevant operating range (AP@50 >= 0.50). Discussed in §V
    # text as a "stress test beyond the operating range" where per-frame KF
    # overhead caps the savings and FixedSkip's zero-overhead skip catches up.
    # (r"\textsc{Albireo} $u=3\!\times\!10^{-3}$", "{plat}/albireo_sweep/u3e-3/{detector}/summary.csv",     "Adaptive",    "albireo",   "D"),
    # --- Fixed-skip baseline ---
    ("FixedSkip-2",   "{plat}/fixedskip/{detector}/summary.csv",               "FixedSkip-2", "fixedskip", "o"),
    ("FixedSkip-3",   "{plat}/fixedskip/{detector}/summary.csv",               "FixedSkip-3", "fixedskip", "o"),
    ("FixedSkip-5",   "{plat}/fixedskip/{detector}/summary.csv",               "FixedSkip-5", "fixedskip", "o"),
    # --- CTD baseline ---
    ("CTD-0.30",      "{plat}/ctd/{detector}/summary.csv",                     "CTD-0.30",    "ctd",       "^"),
    ("CTD-0.50",      "{plat}/ctd/{detector}/summary.csv",                     "CTD-0.50",    "ctd",       "^"),
    ("CTD-0.70",      "{plat}/ctd/{detector}/summary.csv",                     "CTD-0.70",    "ctd",       "^"),
    ("CTD-0.90",      "{plat}/ctd/{detector}/summary.csv",                     "CTD-0.90",    "ctd",       "^"),
    # --- Statues baseline ---
    ("Statues-25",    "{plat}/statues/{detector}/summary.csv",                 "Statues-25",  "statues",   "v"),
    ("Statues-75",    "{plat}/statues/{detector}/summary.csv",                 "Statues-75",  "statues",   "v"),
    ("Statues-200",   "{plat}/statues/{detector}/summary.csv",                 "Statues-200", "statues",   "v"),
    ("Statues-500",   "{plat}/statues/{detector}/summary.csv",                 "Statues-500", "statues",   "v"),
    # --- ERD-only ablation (commented: dominated, discussed in text) ---
    # ("ERD-only",   "{plat}/erd_only/{detector}/summary.csv",               "ERD-only",    "erd_only",  "P"),
]


FAMILY_LABELS = {
    "fixedskip": "FixedSkip-N (N=2,3,5)",
    "ctd":       r"CTD threshold (0.30 $\rightarrow$ 0.90)",
    "statues":   r"Statues threshold (25 $\rightarrow$ 500)",
    "ablation":  r"Ablation Ideas 4 $\rightarrow$ 5 $\rightarrow$ 6",
}

# Map each Albireo point's display label to the BEST_F1 cache key (which
# rename for display without re-running the F1 sweep.
F1_CSV_KEY = {
    r"\textsc{Albireo}":                        "Albireo",
    r"\textsc{Albireo} $u=5\!\times\!10^{-4}$": "Albireo u=5e-4",
    r"\textsc{Albireo} $u=7\!\times\!10^{-4}$": "Albireo u=7e-4",
    r"\textsc{Albireo} $u=1\!\times\!10^{-3}$": "Albireo u=1e-3",
    r"\textsc{Albireo} $u=3\!\times\!10^{-3}$": "Albireo u=3e-3",
}

# Headline points labelled near their markers.
ANNOT_LABELS = {
    "Vanilla",
    r"\textsc{Albireo}",
    r"\textsc{Albireo} $u=1\!\times\!10^{-3}$",
    "FixedSkip-2",
    "FixedSkip-5",
}

# Per-label annotation offset and horizontal alignment.
# Default places the label 7 pt right and 5 pt above the marker, left-aligned.
# Overrides flip specific labels to the left to avoid collisions.
DEFAULT_ANNOT_OFFSET = (7, 5, "left")
ANNOT_OFFSETS = {
    # FixedSkip-2 and Albireo u=1e-3 sit very close in (energy, AP) space —
    # send their labels to the left of the marker so they don't overlap.
    "FixedSkip-2":                              (-7, 5, "right"),
    r"\textsc{Albireo} $u=1\!\times\!10^{-3}$": (-7, 5, "right"),
}


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
def load_rows(csv_path):
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def aggregate(rows, sys_filter):
    """Mean AP@50, mAP, default-F1, energy_j, ms/fr over rows matching sys_filter."""
    sel = [r for r in rows if r.get("system") == sys_filter]
    if not sel:
        return None

    def fmean(key):
        vals = []
        for r in sel:
            v = r.get(key)
            if v in (None, "", "None"):
                continue
            try:
                vals.append(float(v))
            except Exception:
                pass
        return float(np.mean(vals)) if vals else None

    def isum(key):
        s = 0
        for r in sel:
            v = r.get(key)
            if v in (None, "", "None"):
                continue
            try:
                s += int(v)
            except Exception:
                pass
        return s

    tp = isum("total_tp")
    fp = isum("total_fp")
    fn = isum("total_fn")
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)

    return {
        "AP50":   fmean("ap_50"),
        "mAP":    fmean("ap_50_95"),
        "F1":     F1,
        "P":      P,
        "R":      R,
        "AR":     fmean("avg_recall"),
        "AP":     fmean("avg_precision"),
        "energy": fmean("energy_j"),
        "skip":   fmean("skip_rate_pct"),
        "ms":     fmean("avg_frame_time_ms"),
    }


def collect_per_platform(plat, detector="yolo26x"):
    out = []
    for label, path_tpl, sysfilter, family, marker in SYSTEMS:
        path = RESULTS_DIR / path_tpl.format(plat=plat, detector=detector)
        rows = load_rows(path)
        agg = aggregate(rows, sysfilter)
        if agg is None or agg["energy"] is None:
            continue
        # F1_best CSV cache predates the Albireo rename — translate via F1_CSV_KEY
        # for Albireo points; baselines use their label directly.
        f1_key = F1_CSV_KEY.get(label, label)
        agg["F1_best"] = BEST_F1_MAP.get(f1_key)
        out.append({
            "label":  label,
            "color":  COLOR[family],
            "marker": marker,
            "family": family,
            **agg,
        })
    return out


def is_pareto(p, all_pts, y_key, x_key="energy"):
    """Pareto-optimal on (x_key, y_key) where lower x and higher y are better."""
    for q in all_pts:
        if q is p:
            continue
        if q[x_key] is None or q[y_key] is None:
            continue
        if q[x_key] <= p[x_key] and q[y_key] > p[y_key]:
            return False
        if q[x_key] < p[x_key] and q[y_key] >= p[y_key]:
            return False
    return True


def add_edp(pts):
    """Attach EDP and ED^2P (lower is better) to each system point."""
    for p in pts:
        if p["energy"] is None or p["ms"] is None:
            p["edp"] = None
            p["ed2p"] = None
        else:
            p["edp"] = p["energy"] * p["ms"]
            p["ed2p"] = p["energy"] * p["ms"] * p["ms"]


# ---------------------------------------------------------------------------
# Custom legend builder — one entry per family, each showing its marker
# (with a thin black border) on the corresponding line style. This avoids
# the matplotlib auto-legend pulling duplicate handles from each scatter
# call and keeps marker styling consistent across all Pareto figures.
# ---------------------------------------------------------------------------
def _build_legend_handles(families_present, edge_color="black", edge_lw=0.6):
    handles = []
    if "vanilla" in families_present:
        handles.append(Line2D(
            [0], [0], marker="s", linestyle="none",
            markerfacecolor=COLOR["vanilla"],
            markeredgecolor=edge_color, markeredgewidth=edge_lw,
            markersize=10, label="Vanilla"))
    if "albireo" in families_present:
        handles.append(Line2D(
            [0], [0], marker="*", linestyle="-",
            color=COLOR["albireo"], markerfacecolor=COLOR["albireo"],
            markeredgecolor=edge_color, markeredgewidth=edge_lw,
            markersize=12, linewidth=2.4,
            label=r"\textsc{Albireo} sweep "
                  r"($u=3\!\times\!10^{-4} \rightarrow 1\!\times\!10^{-3}$)"))
    if "fixedskip" in families_present:
        handles.append(Line2D(
            [0], [0], marker="o", linestyle="--",
            color=COLOR["fixedskip"], markerfacecolor=COLOR["fixedskip"],
            markeredgecolor=edge_color, markeredgewidth=edge_lw,
            markersize=8, linewidth=1.6,
            label="FixedSkip-N (N=2,3,5)"))
    if "ctd" in families_present:
        handles.append(Line2D(
            [0], [0], marker="^", linestyle="--",
            color=COLOR["ctd"], markerfacecolor=COLOR["ctd"],
            markeredgecolor=edge_color, markeredgewidth=edge_lw,
            markersize=8, linewidth=1.6,
            label=r"CTD threshold (0.30 $\rightarrow$ 0.90)"))
    if "statues" in families_present:
        handles.append(Line2D(
            [0], [0], marker="v", linestyle="--",
            color=COLOR["statues"], markerfacecolor=COLOR["statues"],
            markeredgecolor=edge_color, markeredgewidth=edge_lw,
            markersize=8, linewidth=1.6,
            label=r"Statues threshold (25 $\rightarrow$ 500)"))
    if "ablation" in families_present:
        handles.append(Line2D(
            [0], [0], marker="o", linestyle="--",
            color=COLOR["ablation"], markerfacecolor=COLOR["ablation"],
            markeredgecolor=edge_color, markeredgewidth=edge_lw,
            markersize=8, linewidth=1.6,
            label=r"Ablation Ideas 4 $\rightarrow$ 5 $\rightarrow$ 6"))
    if "erd_only" in families_present:
        handles.append(Line2D(
            [0], [0], marker="P", linestyle="none",
            markerfacecolor=COLOR["erd_only"],
            markeredgecolor=edge_color, markeredgewidth=edge_lw,
            markersize=10, label="ERD-only"))
    return handles


# ---------------------------------------------------------------------------
# Pareto plot — single y axis, x can be energy / EDP / ED^2P / latency
# ---------------------------------------------------------------------------
def plot_pareto(y_key, y_label, fname, title_suffix,
                x_key="energy",
                x_label="Average Energy per Clip (J)",
                detector="yolo26x"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    family_line_color = {
        "fixedskip": COLOR["fixedskip"],
        "ctd":       COLOR["ctd"],
        "statues":   COLOR["statues"],
        "ablation":  COLOR["ablation"],
    }

    # Uniform marker styling — all scatter points get a thin black border
    # regardless of Pareto status, so the legend handle for each family
    # matches the rendered markers exactly.
    EDGE_COLOR = "black"
    EDGE_LW = 0.6

    families_present = set()

    for ax, plat in zip(axes, PLATFORMS):
        pts = collect_per_platform(plat, detector=detector)
        add_edp(pts)
        valid = [p for p in pts
                 if p[y_key] is not None and p[x_key] is not None]

        # ---- Per-family connecting lines (drawn first; markers sit on top) ----
        by_family = {}
        for p in valid:
            by_family.setdefault(p["family"], []).append(p)
            families_present.add(p["family"])

        for fam, fam_pts in by_family.items():
            if len(fam_pts) < 2:
                continue
            fam_pts_sorted = sorted(fam_pts, key=lambda r: r[x_key])
            xs = [p[x_key] for p in fam_pts_sorted]
            ys = [p[y_key] for p in fam_pts_sorted]
            if fam == "albireo":
                ax.plot(xs, ys, "-",
                        color=COLOR["albireo"],
                        linewidth=2.4, alpha=0.75, zorder=2)
            else:
                ax.plot(xs, ys, "--",
                        color=family_line_color.get(fam, PERSIAN["gray"]),
                        linewidth=1.6, alpha=0.6, zorder=2)

        # ---- Markers (uniform thin black border on all points) ----
        for p in valid:
            sz = 130 if p["family"] in ("vanilla", "albireo") else 70
            ax.scatter(p[x_key], p[y_key], color=p["color"], marker=p["marker"],
                       s=sz, edgecolors=EDGE_COLOR, linewidths=EDGE_LW,
                       zorder=3, alpha=0.92)

        # ---- Annotations on headline points ----
        for p in valid:
            if p["label"] not in ANNOT_LABELS:
                continue
            dx, dy, ha = ANNOT_OFFSETS.get(p["label"], DEFAULT_ANNOT_OFFSET)
            ax.annotate(p["label"], (p[x_key], p[y_key]),
                        textcoords="offset points", xytext=(dx, dy),
                        ha=ha, fontsize=11, color=p["color"])

        ax.set_xlabel(x_label)
        ax.set_xlim(left=0)
        ax.set_title(PARETO_PLATFORM_LABELS[plat], fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")

    # ---- Custom legend on Thor axes — one entry per family, with marker ----
    legend_handles = _build_legend_handles(families_present,
                                           edge_color=EDGE_COLOR,
                                           edge_lw=EDGE_LW)
    if legend_handles:
        axes[0].legend(handles=legend_handles, loc="lower right",
                       fontsize=11, framealpha=0.9,
                       handletextpad=0.4, borderpad=0.4,
                       numpoints=1)

    axes[0].set_ylabel(y_label)
    # Figure-level title commented for the paper — caption carries description.
    # fig.suptitle(f"{y_label} vs {x_label.split(' (')[0]} "
    #              f"({detector}, 200 BDD100K MOT clips){title_suffix}",
    #              fontweight="bold", fontsize=12)
    plt.tight_layout()
    out_png = OUT_DIR / f"{fname}.png"
    out_pdf = OUT_DIR / f"{fname}.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"  saved {fname}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Per-platform tabular summaries (printed to stdout)
# ---------------------------------------------------------------------------
def print_table_per_platform(detector="yolo26x"):
    for plat in PLATFORMS:
        print(f"\n=== {PARETO_PLATFORM_LABELS[plat]} ({detector}) ===")
        pts = collect_per_platform(plat, detector=detector)
        valid = [p for p in pts if p["energy"] is not None]
        if not valid:
            print(f"  (no data)")
            continue
        van = next((p for p in valid if p["label"] == "Vanilla"), None)
        van_E = van["energy"] if van else 0
        print(f"  {'System':<22} {'AP@50':>7} {'mAP':>7} {'F1':>7} "
              f"{'Skip%':>6} {'E (J)':>7} {'dE%':>7}")
        for p in sorted(valid, key=lambda r: r["energy"]):
            de = 100 * (p["energy"] - van_E) / van_E if van_E else float("nan")
            print(f"  {p['label']:<22} {p['AP50']:>7.4f} {p['mAP']:>7.4f} "
                  f"{p['F1']:>7.4f} {p['skip']:>5.1f}% "
                  f"{p['energy']:>7.1f} {de:>+6.1f}%")
        for ykey, ylab in [("AP50", "AP@50"), ("mAP", "mAP"),
                           ("F1", "F1_default")]:
            pareto = [p for p in valid if is_pareto(p, valid, ykey)]
            pareto.sort(key=lambda r: r["energy"])
            names = ", ".join(p["label"] for p in pareto)
            print(f"  Pareto on (E, {ylab}): {names}")


def print_edp_frontiers(detector="yolo26x"):
    print()
    print(f"=== EDP / ED^2P frontiers ({detector}) ===")
    for plat in PLATFORMS:
        pts = collect_per_platform(plat, detector=detector)
        add_edp(pts)
        for axis_key, axis_label in [("edp", "EDP"), ("ed2p", "ED^2P")]:
            valid = [p for p in pts if p[axis_key] is not None]
            if not valid:
                continue
            van = next((p for p in valid if p["label"] == "Vanilla"), None)
            van_v = van[axis_key] if van else 0
            print(f"\n{PARETO_PLATFORM_LABELS[plat]} — sorted by {axis_label}")
            print(f"  {'System':<22} {'AP@50':>7} {'mAP':>7} {'F1':>7} "
                  f"{'E (J)':>7} {'ms/fr':>7} {axis_label:>12} {'d%':>8}")
            for p in sorted(valid, key=lambda r: r[axis_key]):
                d = (100 * (p[axis_key] - van_v) / van_v
                     if van_v else float("nan"))
                print(f"  {p['label']:<22} {p['AP50']:>7.4f} "
                      f"{p['mAP']:>7.4f} {p['F1']:>7.4f} "
                      f"{p['energy']:>7.1f} {p['ms']:>7.1f} "
                      f"{p[axis_key]:>12.0f} {d:>+7.1f}%")
            for ykey, ylab in [("AP50", "AP@50"), ("mAP", "mAP"),
                               ("F1", "F1_default")]:
                pareto = [p for p in valid if is_pareto(p, valid, ykey, axis_key)]
                pareto.sort(key=lambda r: r[axis_key])
                names = ", ".join(p["label"] for p in pareto)
                print(f"  Pareto on ({axis_label}, {ylab}): {names}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_all_pareto_plots(detector="yolo26x", print_tables=True):
    """Generate every Pareto figure for the given detector.

    Saves PNG + PDF pairs to the visualizations dir. Returns nothing.
    Bumps font sizes above setup_style() defaults so the figures stay legible
    when rendered at IEEE 2-column figure* width (~7 inches).
    """
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    })
    print(f"Building Pareto plots — detector={detector}")
    print("  axis: Energy")
    plot_pareto("AP50",    "AP@50",
                "pareto_ap50_full",     " --- Full system comparison",
                detector=detector)
    plot_pareto("mAP",     "mAP@0.5:0.95",
                "pareto_map_full",      " --- Full system comparison",
                detector=detector)
    plot_pareto("F1",      "F1 (default threshold)",
                "pareto_f1_full",       " --- Full system comparison",
                detector=detector)
    plot_pareto("AR",      "Average Recall (mean of PR curve)",
                "pareto_ar_full",       " --- Full system comparison",
                detector=detector)
    plot_pareto("F1_best", "F1 (best operating threshold per system)",
                "pareto_f1best_full",   " --- Full system comparison",
                detector=detector)

    print("  axis: EDP (Energy x Delay)")
    EDP_LABEL = r"Per-Frame EDP (J$\cdot$ms/fr)"
    plot_pareto("AP50",    "AP@50",            "pareto_edp_ap50_full",
                " --- EDP", x_key="edp", x_label=EDP_LABEL, detector=detector)
    plot_pareto("mAP",     "mAP@0.5:0.95",     "pareto_edp_map_full",
                " --- EDP", x_key="edp", x_label=EDP_LABEL, detector=detector)
    plot_pareto("F1",      "F1 (default threshold)", "pareto_edp_f1_full",
                " --- EDP", x_key="edp", x_label=EDP_LABEL, detector=detector)
    plot_pareto("AR",      "Average Recall",   "pareto_edp_ar_full",
                " --- EDP", x_key="edp", x_label=EDP_LABEL, detector=detector)
    plot_pareto("F1_best", "F1 (best operating threshold)",
                "pareto_edp_f1best_full",
                " --- EDP", x_key="edp", x_label=EDP_LABEL, detector=detector)

    print("  axis: Latency (avg frame time)")
    LAT_LABEL = "Average Frame Time (ms)"
    plot_pareto("AP50",    "AP@50",            "pareto_lat_ap50_full",
                " --- Latency", x_key="ms", x_label=LAT_LABEL, detector=detector)
    plot_pareto("mAP",     "mAP@0.5:0.95",     "pareto_lat_map_full",
                " --- Latency", x_key="ms", x_label=LAT_LABEL, detector=detector)
    plot_pareto("F1",      "F1 (default threshold)", "pareto_lat_f1_full",
                " --- Latency", x_key="ms", x_label=LAT_LABEL, detector=detector)
    plot_pareto("F1_best", "F1 (best operating threshold)",
                "pareto_lat_f1best_full",
                " --- Latency", x_key="ms", x_label=LAT_LABEL, detector=detector)
    plot_pareto("AR",      "Average Recall",   "pareto_lat_ar_full",
                " --- Latency", x_key="ms", x_label=LAT_LABEL, detector=detector)

    print(r"  axis: ED^2P (Energy x Delay^2)")
    ED2P_LABEL = r"ED$^2$P per Clip (J$\cdot$ms$^2$)"
    plot_pareto("AP50",    "AP@50",            "pareto_ed2p_ap50_full",
                r" --- ED$^2$P", x_key="ed2p", x_label=ED2P_LABEL,
                detector=detector)
    plot_pareto("mAP",     "mAP@0.5:0.95",     "pareto_ed2p_map_full",
                r" --- ED$^2$P", x_key="ed2p", x_label=ED2P_LABEL,
                detector=detector)
    plot_pareto("F1",      "F1 (default threshold)", "pareto_ed2p_f1_full",
                r" --- ED$^2$P", x_key="ed2p", x_label=ED2P_LABEL,
                detector=detector)
    plot_pareto("AR",      "Average Recall",   "pareto_ed2p_ar_full",
                r" --- ED$^2$P", x_key="ed2p", x_label=ED2P_LABEL,
                detector=detector)
    plot_pareto("F1_best", "F1 (best operating threshold)",
                "pareto_ed2p_f1best_full",
                r" --- ED$^2$P", x_key="ed2p", x_label=ED2P_LABEL,
                detector=detector)

    if print_tables:
        print_table_per_platform(detector=detector)
        print_edp_frontiers(detector=detector)


if __name__ == "__main__":
    setup_style()
    generate_all_pareto_plots(detector="yolo26x")
    print("\nDone.")
