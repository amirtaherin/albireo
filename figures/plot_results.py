"""
Script Name: plot_results.py
Description: Generate SEC 2026 paper figures from experiment results.
             Produces 5 plots: Pareto curve, frame state distribution,
             per-domain energy breakdown, energy savings bar, power trace.

Author: Amir Taherin
Email: amirtaherin@gmail.com
Email: taherin.a@northeastern.edu
Date Created: 2026-04-20
Last Modified: 2026-04-20
Version: 1.0

License: MIT License

Usage:
    python plot_results.py
"""

import csv
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PERSIAN = {
    "red":        "#c81d11",
    "plum":       "#701c1c",
    "pink":       "#f77fbe",
    "rose":       "#fe28a2",
    "green":      "#00a693",
    "orange":     "#d99058",
    "indigo":     "#32127a",
    "blue":       "#1c39bb",
    "mediumBlue": "#0067a5",
    "gray":       "#777777",
    "lightGray":  "#b0b0b0",
}

# Family → PERSIAN-key mapping used by the comprehensive Pareto plot in
# plot_pareto_full.py. Keeping it here so both files share a single visual
# language.
FAMILY_COLOR_KEYS = {
    "vanilla":   "red",
    "albireo":   "green",
    "fixedskip": "mediumBlue",
    "ctd":       "orange",
    "statues":   "indigo",
    "erd_only":  "gray",
    "ablation":  "lightGray",
}

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUT_DIR = Path(__file__).resolve().parent

PLATFORMS = ["thor", "orin"]
DETECTORS = ["yolo26x", "yolo11x", "rfdetr-large"]
DETECTOR_LABELS = {"yolo26x": "YOLO26x", "yolo11x": "YOLO11x",
                   "rfdetr-large": "RF-DETR-L"}
# Compact labels for table column headers where horizontal space is tight.
DETECTOR_SHORT_LABELS = {"yolo26x": "Y26x", "yolo11x": "Y11x",
                         "rfdetr-large": "RFD-L"}
PLATFORM_LABELS = {"thor": "Thor", "orin": "Orin"}

RAIL_PATTERN = re.compile(r'(\w+)\s+(\d+)mW/(\d+)mW')
TOTAL_RAIL_CANDIDATES = ["SYS5V", "VDD_IN", "VIN", "VDD_SYS"]


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def by_system(rows):
    d = {}
    for r in rows:
        d.setdefault(r["system"], []).append(r)
    return d


def avg(rows, key):
    vals = [float(r[key]) for r in rows
            if r.get(key) and r[key] not in ("None", "")]
    return np.mean(vals) if vals else None


def std(rows, key):
    vals = [float(r[key]) for r in rows
            if r.get(key) and r[key] not in ("None", "")]
    return np.std(vals) if vals else None


def sem95(rows, key):
    vals = [float(r[key]) for r in rows
            if r.get(key) and r[key] not in ("None", "")]
    if len(vals) < 2:
        return 0.0
    return np.std(vals, ddof=1) / np.sqrt(len(vals)) * 1.96


def setup_style():
    plt.style.use("classic")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# =========================================================================
# Plot 1: Pareto Curve — AP@50 vs Energy
# =========================================================================

def plot_pareto():
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, plat in zip(axes, PLATFORMS):
        # Adaptive + Vanilla (yolo26x)
        Albireo = load_csv(RESULTS_DIR / plat / "albireo" / "yolo26x" / "summary.csv")
        systems = by_system(Albireo)

        van_ap = avg(systems["Vanilla"], "ap_50")
        van_ej = avg(systems["Vanilla"], "energy_j")
        adp_ap = avg(systems["Adaptive"], "ap_50")
        adp_ej = avg(systems["Adaptive"], "energy_j")

        # Fixed-skip baselines
        fskip = load_csv(RESULTS_DIR / plat / "fixedskip" / "yolo26x" / "summary.csv")
        fskip_sys = by_system(fskip)

        fs_aps, fs_ejs, fs_labels = [], [], []
        for skip_n in [2, 3, 5]:
            key = f"FixedSkip-{skip_n}"
            if key in fskip_sys:
                fs_aps.append(avg(fskip_sys[key], "ap_50"))
                fs_ejs.append(avg(fskip_sys[key], "energy_j"))
                fs_labels.append(f"Skip-{skip_n}")

        # Plot fixed-skip line
        ax.plot(fs_ejs, fs_aps, '--o', color=PERSIAN["mediumBlue"],
                markersize=7, linewidth=1.5, zorder=2)
        for i, lbl in enumerate(fs_labels):
            ax.annotate(lbl, (fs_ejs[i], fs_aps[i]),
                        textcoords="offset points", xytext=(8, -2),
                        fontsize=8, color=PERSIAN["mediumBlue"])

        # Plot vanilla
        ax.scatter(van_ej, van_ap, color=PERSIAN["red"], s=120, marker='s',
                   zorder=3)
        ax.annotate("Vanilla", (van_ej, van_ap),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=9, fontweight="bold", color=PERSIAN["red"])

        # Plot adaptive
        ax.scatter(adp_ej, adp_ap, color=PERSIAN["green"], s=150, marker='*',
                   zorder=4)
        ax.annotate(r"\textsc{Albireo}", (adp_ej, adp_ap),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=9, fontweight="bold", color=PERSIAN["green"])

        ax.set_xlabel("Energy per Clip (J)")
        ax.set_title(PLATFORM_LABELS[plat], fontweight="bold")

        legend_handles = [
            Line2D([0], [0], marker='o', color=PERSIAN["mediumBlue"],
                   linestyle='--', markersize=6, linewidth=1.5,
                   label="Fixed-Skip"),
            Line2D([0], [0], marker='s', color=PERSIAN["red"],
                   linestyle='None', markersize=7, label="Vanilla"),
            Line2D([0], [0], marker='*', color=PERSIAN["green"],
                   linestyle='None', markersize=9, label=r"\textsc{Albireo}"),
        ]
        ax.legend(handles=legend_handles, loc="lower right",
                  numpoints=1, handletextpad=0.5, borderpad=0.4)

    axes[0].set_ylabel("AP@50")
    # Title commented for the paper — caption carries the description.
# fig.suptitle(r"Accuracy vs Energy: \textsc{Albireo} is Pareto-Optimal (YOLO26x)",
# fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "pareto_ap_vs_energy.png")
    fig.savefig(OUT_DIR / "pareto_ap_vs_energy.pdf")
    plt.close(fig)
    print("  [1/8] pareto_ap_vs_energy")


# =========================================================================
# Plot 2: Frame State Distribution (Stacked Bar)
# =========================================================================

def plot_frame_states():
    # Colors picked for maximum hue separation across the 5 categories.
    # Previous palette (green/mediumBlue/blue) had E, P, I sitting on the
    # same green-to-blue arc and was hard to read; rose for P breaks the
    # cool-color cluster and indigo for I keeps inference visually heavy.
    states = [
        ("E",   "erd_empty_rate_pct",        PERSIAN["green"]),
        ("P",   "predict_rate_pct",          PERSIAN["rose"]),
        ("I",   "pure_infer_rate_pct",       PERSIAN["indigo"]),
        ("E+I", "erd_infer_rate_pct",        PERSIAN["orange"]),
        ("AI",  "augmented_infer_rate_pct",  PERSIAN["red"]),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(DETECTORS))
    bar_width = 0.5

    bottoms = np.zeros(len(DETECTORS))
    for label, key, color in states:
        vals = []
        for det in DETECTORS:
            rows = load_csv(RESULTS_DIR / "thor" / "albireo" / det / "summary.csv")
            adp = by_system(rows).get("Adaptive", [])
            vals.append(avg(adp, key) or 0)
        vals = np.array(vals)
        bars = ax.bar(x, vals, bar_width, bottom=bottoms, color=color,
                       label=label, edgecolor="black", linewidth=0.5)
        for i, v in enumerate(vals):
            if v > 3:
                ax.text(x[i], bottoms[i] + v / 2, f"{v:.1f}\\%",
                        ha="center", va="center", fontsize=15,
                        color="white" if label in ("I", "AI", "P") else "black",
                        fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([DETECTOR_LABELS[d] for d in DETECTORS])
    ax.set_ylabel(r"Percentage of Frames (\%)")
    # Title commented for the paper — caption carries the description.
    # ax.set_title(r"Frame State Distribution by Detector (\textsc{Albireo} Mode)",
    #              fontweight="bold")
    ax.legend(loc="upper center", ncol=5, title="Frame State",
              bbox_to_anchor=(0.5, -0.10), frameon=True, fancybox=True,
              edgecolor="black", borderpad=0.6, handletextpad=0.5,
              columnspacing=1.2)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "frame_state_distribution.png")
    fig.savefig(OUT_DIR / "frame_state_distribution.pdf")
    plt.close(fig)
    print("  [2/8] frame_state_distribution")


# =========================================================================
# Plot 3: Per-Domain Energy Breakdown (Grouped Stacked Bar)
# =========================================================================

def plot_domain_energy():
    domains = [
        ("GPU", "energy_gpu_j", PERSIAN["red"]),
        ("CPU", "energy_cpu_j", PERSIAN["blue"]),
        ("IO/MEM", "energy_io_j", PERSIAN["green"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

    for col_idx, (ax, plat) in enumerate(zip(axes, PLATFORMS)):
        n_det = len(DETECTORS)
        x = np.arange(n_det)
        bar_width = 0.35

        for sys_idx, (sname, offset) in enumerate(
                [("Vanilla", -bar_width / 2), ("Adaptive", bar_width / 2)]):
            bottoms = np.zeros(n_det)
            for dom_label, key, color in domains:
                vals = []
                errs = []
                for det in DETECTORS:
                    rows = load_csv(
                        RESULTS_DIR / plat / "albireo" / det / "summary.csv")
                    sys_rows = by_system(rows).get(sname, [])
                    vals.append(avg(sys_rows, key) or 0)
                    errs.append(sem95(sys_rows, key))
                vals = np.array(vals)
                errs = np.array(errs)
                alpha = 1.0 if sname == "Vanilla" else 0.7
                hatch = "" if sname == "Vanilla" else "///"
                ax.bar(x + offset, vals, bar_width, bottom=bottoms,
                       yerr=errs, capsize=2,
                       color=color, alpha=alpha, hatch=hatch,
                       edgecolor="black", linewidth=0.5, ecolor="black",
                       label=f"{dom_label} ({sname[0]})" if x[0] == 0 else "")
                bottoms += vals

            # Total energy label on top
            for i in range(n_det):
                ax.text(x[i] + offset, bottoms[i] + 5,
                        f"{bottoms[i]:.0f}",
                        ha="center", va="bottom", fontsize=15)

        ax.set_xticks(x)
        ax.set_xticklabels([DETECTOR_LABELS[d] for d in DETECTORS])
        # Y-label only on the leftmost subplot — both subplots share the same units
        if col_idx == 0:
            ax.set_ylabel("Mean Energy per Clip (J)")
        ax.set_title(PLATFORM_LABELS[plat], fontweight="bold")

    # Build legend manually to avoid duplicates
    from matplotlib.patches import Patch
    legend_elements = []
    for dom_label, _, color in domains:
        legend_elements.append(Patch(facecolor=color, edgecolor="black",
                                      linewidth=0.5, label=dom_label))
    legend_elements.append(Patch(facecolor="gray", edgecolor="black",
                                  linewidth=0.5, alpha=1.0, label="Vanilla"))
    legend_elements.append(Patch(facecolor="gray", edgecolor="black",
                                  linewidth=0.5, alpha=0.7, hatch="///",
                                  label=r"\textsc{Albireo}"))
    axes[1].legend(handles=legend_elements, loc="upper right", fontsize=14)

    # Title commented for the paper — caption carries the description.
# fig.suptitle(r"Per-Domain Energy Breakdown: Vanilla vs \textsc{Albireo}",
# fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "per_domain_energy.png")
    fig.savefig(OUT_DIR / "per_domain_energy.pdf")
    plt.close(fig)
    print("  [3/8] per_domain_energy")


# =========================================================================
# Plot 3b: Per-Domain Energy Breakdown with Table
# =========================================================================

def plot_domain_energy_table():
    from matplotlib.patches import Patch
    import matplotlib.gridspec as gridspec

    domains = [
        ("GPU", "energy_gpu_j", PERSIAN["red"]),
        ("CPU", "energy_cpu_j", PERSIAN["blue"]),
        ("IO/MEM", "energy_io_j", PERSIAN["green"]),
    ]

    fig = plt.figure(figsize=(11, 6.5))
    outer = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.05,
                              wspace=0.12)

    for col, plat in enumerate(PLATFORMS):
        ax = fig.add_subplot(outer[0, col])
        n_det = len(DETECTORS)
        x = np.arange(n_det)
        bar_width = 0.35

        table_data = {}  # (domain, det, system) -> value

        for sys_idx, (sname, offset) in enumerate(
                [("Vanilla", -bar_width / 2), ("Adaptive", bar_width / 2)]):
            bottoms = np.zeros(n_det)
            for dom_label, key, color in domains:
                vals = []
                errs = []
                for det in DETECTORS:
                    rows = load_csv(
                        RESULTS_DIR / plat / "albireo" / det / "summary.csv")
                    sys_rows = by_system(rows).get(sname, [])
                    v = avg(sys_rows, key) or 0
                    vals.append(v)
                    errs.append(sem95(sys_rows, key))
                vals = np.array(vals)
                errs = np.array(errs)
                alpha = 1.0 if sname == "Vanilla" else 0.7
                hatch = "" if sname == "Vanilla" else "///"
                ax.bar(x + offset, vals, bar_width, bottom=bottoms,
                       yerr=errs, capsize=2,
                       color=color, alpha=alpha, hatch=hatch,
                       edgecolor="black", linewidth=0.5, ecolor="black")
                for i, v in enumerate(vals):
                    table_data[(dom_label, DETECTORS[i], sname)] = v
                bottoms += vals

            for i in range(n_det):
                ax.text(x[i] + offset, bottoms[i] + 5,
                        f"{bottoms[i]:.0f}",
                        ha="center", va="bottom", fontsize=7.5)
                table_data[("Total", DETECTORS[i], sname)] = bottoms[i]

        ax.set_xticks(x)
        ax.set_xticklabels([DETECTOR_LABELS[d] for d in DETECTORS])
        # Y-label only on the leftmost subplot — both subplots share the same units
        if col == 0:
            ax.set_ylabel("Mean Energy per Clip (J)")
        ax.set_title(PLATFORM_LABELS[plat], fontweight="bold")

        # Build table below this subplot
        ax_tab = fig.add_subplot(outer[1, col])
        ax_tab.axis("off")

        row_labels = ["GPU", "CPU", "IO/MEM", "Total"]
        col_labels = []
        for det in DETECTORS:
            col_labels.extend([f"{DETECTOR_SHORT_LABELS[det]} V",
                               f"{DETECTOR_SHORT_LABELS[det]} A"])

        cell_text = []
        cell_colors = []
        domain_color_map = {"GPU": PERSIAN["red"], "CPU": PERSIAN["blue"],
                            "IO/MEM": PERSIAN["green"], "Total": "#dddddd"}
        for row_name in row_labels:
            row_vals = []
            row_colors = []
            base_color = domain_color_map[row_name]
            for det in DETECTORS:
                for sname in ["Vanilla", "Adaptive"]:
                    v = table_data.get((row_name, det, sname), 0)
                    row_vals.append(f"{v:.0f}")
                    import matplotlib.colors as mcolors
                    rgb = mcolors.to_rgba(base_color)
                    light = (rgb[0] * 0.3 + 0.7, rgb[1] * 0.3 + 0.7,
                             rgb[2] * 0.3 + 0.7, 1.0)
                    row_colors.append(light)
            cell_text.append(row_vals)
            cell_colors.append(row_colors)

        tbl = ax_tab.table(cellText=cell_text, rowLabels=row_labels,
                           colLabels=col_labels, cellColours=cell_colors,
                           loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.2)

        for (row, col_idx), cell in tbl.get_celld().items():
            cell.set_edgecolor("gray")
            cell.set_linewidth(0.5)
            if row == 0:
                cell.set_text_props(fontweight="bold", fontsize=7)
                cell.set_facecolor("#f0f0f0")
            if col_idx == -1:
                cell.set_text_props(fontweight="bold")

    # Legend on the first bar axis
    legend_elements = []
    for dom_label, _, color in domains:
        legend_elements.append(Patch(facecolor=color, edgecolor="black",
                                      linewidth=0.5, label=dom_label))
    legend_elements.append(Patch(facecolor="gray", edgecolor="black",
                                  linewidth=0.5, alpha=1.0, label="Vanilla"))
    legend_elements.append(Patch(facecolor="gray", edgecolor="black",
                                  linewidth=0.5, alpha=0.7, hatch="///",
                                  label=r"\textsc{Albireo}"))
    fig.axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8)

    # Title commented for the paper — caption carries the description.
# fig.suptitle(r"Per-Domain Energy Breakdown: Vanilla vs \textsc{Albireo} (J)",
# fontweight="bold", fontsize=13)
    fig.savefig(OUT_DIR / "per_domain_energy_table.png")
    fig.savefig(OUT_DIR / "per_domain_energy_table.pdf")
    plt.close(fig)
    print("  [3b] per_domain_energy_table")


# =========================================================================
# Plot 3c: Per-Domain Power Breakdown (Option 1 — mirror of energy plot)
# =========================================================================

def plot_domain_power():
    from matplotlib.patches import Patch

    domains = [
        ("GPU", "avg_power_gpu_w", PERSIAN["red"]),
        ("CPU", "avg_power_cpu_w", PERSIAN["blue"]),
        ("IO/MEM", "avg_power_io_w", PERSIAN["green"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

    for col_idx, (ax, plat) in enumerate(zip(axes, PLATFORMS)):
        n_det = len(DETECTORS)
        x = np.arange(n_det)
        bar_width = 0.35

        for sys_idx, (sname, offset) in enumerate(
                [("Vanilla", -bar_width / 2), ("Adaptive", bar_width / 2)]):
            bottoms = np.zeros(n_det)
            for dom_label, key, color in domains:
                vals = []
                errs = []
                for det in DETECTORS:
                    rows = load_csv(
                        RESULTS_DIR / plat / "albireo" / det / "summary.csv")
                    sys_rows = by_system(rows).get(sname, [])
                    vals.append(avg(sys_rows, key) or 0)
                    errs.append(sem95(sys_rows, key))
                vals = np.array(vals)
                errs = np.array(errs)
                alpha = 1.0 if sname == "Vanilla" else 0.7
                hatch = "" if sname == "Vanilla" else "///"
                ax.bar(x + offset, vals, bar_width, bottom=bottoms,
                       yerr=errs, capsize=2,
                       color=color, alpha=alpha, hatch=hatch,
                       edgecolor="black", linewidth=0.5, ecolor="black")
                bottoms += vals

            for i in range(n_det):
                ax.text(x[i] + offset, bottoms[i] + 0.3,
                        f"{bottoms[i]:.1f}",
                        ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels([DETECTOR_LABELS[d] for d in DETECTORS])
        # Y-label only on the leftmost subplot — both subplots share the same units
        if col_idx == 0:
            ax.set_ylabel("Mean Power (W)")
        ax.set_title(PLATFORM_LABELS[plat], fontweight="bold")

    legend_elements = []
    for dom_label, _, color in domains:
        legend_elements.append(Patch(facecolor=color, edgecolor="black",
                                      linewidth=0.5, label=dom_label))
    legend_elements.append(Patch(facecolor="gray", edgecolor="black",
                                  linewidth=0.5, alpha=1.0, label="Vanilla"))
    legend_elements.append(Patch(facecolor="gray", edgecolor="black",
                                  linewidth=0.5, alpha=0.7, hatch="///",
                                  label=r"\textsc{Albireo}"))
    axes[1].legend(handles=legend_elements, loc="upper right", fontsize=8)

    # Title commented for the paper — caption carries the description.
# fig.suptitle(r"Per-Domain Power Breakdown: Vanilla vs \textsc{Albireo}",
# fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "per_domain_power.png")
    fig.savefig(OUT_DIR / "per_domain_power.pdf")
    plt.close(fig)
    print("  [3c] per_domain_power")


# =========================================================================
# Plot 3c-table: Per-Domain Power Breakdown with Table
# =========================================================================

def plot_domain_power_table():
    from matplotlib.patches import Patch
    import matplotlib.gridspec as gridspec
    import matplotlib.colors as mcolors

    domains = [
        ("GPU", "avg_power_gpu_w", PERSIAN["red"]),
        ("CPU", "avg_power_cpu_w", PERSIAN["blue"]),
        ("IO/MEM", "avg_power_io_w", PERSIAN["green"]),
    ]

    fig = plt.figure(figsize=(11, 6.5))
    outer = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.05,
                              wspace=0.12)

    for col, plat in enumerate(PLATFORMS):
        ax = fig.add_subplot(outer[0, col])
        n_det = len(DETECTORS)
        x = np.arange(n_det)
        bar_width = 0.35

        table_data = {}

        for sys_idx, (sname, offset) in enumerate(
                [("Vanilla", -bar_width / 2), ("Adaptive", bar_width / 2)]):
            bottoms = np.zeros(n_det)
            for dom_label, key, color in domains:
                vals = []
                errs = []
                for det in DETECTORS:
                    rows = load_csv(
                        RESULTS_DIR / plat / "albireo" / det / "summary.csv")
                    sys_rows = by_system(rows).get(sname, [])
                    v = avg(sys_rows, key) or 0
                    vals.append(v)
                    errs.append(sem95(sys_rows, key))
                vals = np.array(vals)
                errs = np.array(errs)
                alpha = 1.0 if sname == "Vanilla" else 0.7
                hatch = "" if sname == "Vanilla" else "///"
                ax.bar(x + offset, vals, bar_width, bottom=bottoms,
                       yerr=errs, capsize=2,
                       color=color, alpha=alpha, hatch=hatch,
                       edgecolor="black", linewidth=0.5, ecolor="black")
                for i, v in enumerate(vals):
                    table_data[(dom_label, DETECTORS[i], sname)] = v
                bottoms += vals

            for i in range(n_det):
                ax.text(x[i] + offset, bottoms[i] + 0.3,
                        f"{bottoms[i]:.1f}",
                        ha="center", va="bottom", fontsize=7.5)
                table_data[("Total", DETECTORS[i], sname)] = bottoms[i]

        ax.set_xticks(x)
        ax.set_xticklabels([DETECTOR_LABELS[d] for d in DETECTORS])
        # Y-label only on the leftmost subplot — both subplots share the same units
        if col == 0:
            ax.set_ylabel("Mean Power (W)")
        ax.set_title(PLATFORM_LABELS[plat], fontweight="bold")

        # Table below subplot
        ax_tab = fig.add_subplot(outer[1, col])
        ax_tab.axis("off")

        row_labels = ["GPU", "CPU", "IO/MEM", "Total"]
        col_labels = []
        for det in DETECTORS:
            col_labels.extend([f"{DETECTOR_SHORT_LABELS[det]} V",
                               f"{DETECTOR_SHORT_LABELS[det]} A"])

        cell_text = []
        cell_colors = []
        domain_color_map = {"GPU": PERSIAN["red"], "CPU": PERSIAN["blue"],
                            "IO/MEM": PERSIAN["green"], "Total": "#dddddd"}
        for row_name in row_labels:
            row_vals = []
            row_colors = []
            base_color = domain_color_map[row_name]
            for det in DETECTORS:
                for sname in ["Vanilla", "Adaptive"]:
                    v = table_data.get((row_name, det, sname), 0)
                    row_vals.append(f"{v:.1f}")
                    rgb = mcolors.to_rgba(base_color)
                    light = (rgb[0] * 0.3 + 0.7, rgb[1] * 0.3 + 0.7,
                             rgb[2] * 0.3 + 0.7, 1.0)
                    row_colors.append(light)
            cell_text.append(row_vals)
            cell_colors.append(row_colors)

        tbl = ax_tab.table(cellText=cell_text, rowLabels=row_labels,
                           colLabels=col_labels, cellColours=cell_colors,
                           loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.2)

        for (row, col_idx), cell in tbl.get_celld().items():
            cell.set_edgecolor("gray")
            cell.set_linewidth(0.5)
            if row == 0:
                cell.set_text_props(fontweight="bold", fontsize=7)
                cell.set_facecolor("#f0f0f0")
            if col_idx == -1:
                cell.set_text_props(fontweight="bold")

    # Legend
    legend_elements = []
    for dom_label, _, color in domains:
        legend_elements.append(Patch(facecolor=color, edgecolor="black",
                                      linewidth=0.5, label=dom_label))
    legend_elements.append(Patch(facecolor="gray", edgecolor="black",
                                  linewidth=0.5, alpha=1.0, label="Vanilla"))
    legend_elements.append(Patch(facecolor="gray", edgecolor="black",
                                  linewidth=0.5, alpha=0.7, hatch="///",
                                  label=r"\textsc{Albireo}"))
    fig.axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8)

    # Title commented for the paper — caption carries the description.
# fig.suptitle(r"Per-Domain Power Breakdown: Vanilla vs \textsc{Albireo} (W)",
# fontweight="bold", fontsize=13)
    fig.savefig(OUT_DIR / "per_domain_power_table.png")
    fig.savefig(OUT_DIR / "per_domain_power_table.pdf")
    plt.close(fig)
    print("  [3c-table] per_domain_power_table")


# =========================================================================
# Plot 3d: Power + Savings Combo (Option 2)
# =========================================================================

def plot_power_savings_combo():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    twin_axes = []
    all_pct = []

    for idx, (ax, plat) in enumerate(zip(axes, PLATFORMS)):
        n_det = len(DETECTORS)
        x = np.arange(n_det)
        bar_width = 0.3

        van_vals, adp_vals = [], []
        van_errs, adp_errs = [], []
        pct_savings = []

        for det in DETECTORS:
            rows = load_csv(
                RESULTS_DIR / plat / "albireo" / det / "summary.csv")
            systems = by_system(rows)

            vp = [float(r["avg_power_w"]) for r in systems["Vanilla"]
                  if r.get("avg_power_w") and r["avg_power_w"] != "None"]
            ap = [float(r["avg_power_w"]) for r in systems["Adaptive"]
                  if r.get("avg_power_w") and r["avg_power_w"] != "None"]

            vm, am = np.mean(vp), np.mean(ap)
            van_vals.append(vm)
            adp_vals.append(am)
            van_errs.append(np.std(vp, ddof=1) / np.sqrt(len(vp)) * 1.96)
            adp_errs.append(np.std(ap, ddof=1) / np.sqrt(len(ap)) * 1.96)
            pct_savings.append(100.0 * (vm - am) / vm)

        all_pct.extend(pct_savings)

        bars_v = ax.bar(x - bar_width / 2, van_vals, bar_width,
                        yerr=van_errs, capsize=3,
                        color=PERSIAN["indigo"], edgecolor="black",
                        linewidth=0.5, label="Vanilla")
        bars_a = ax.bar(x + bar_width / 2, adp_vals, bar_width,
                        yerr=adp_errs, capsize=3,
                        color=PERSIAN["orange"], edgecolor="black",
                        linewidth=0.5, label=r"\textsc{Albireo}")

        for i in range(n_det):
            ax.text(x[i] - bar_width / 2, van_vals[i] + van_errs[i] + 0.3,
                    f"{van_vals[i]:.1f}", ha="center", va="bottom",
                    fontsize=8)
            ax.text(x[i] + bar_width / 2, adp_vals[i] + adp_errs[i] + 0.3,
                    f"{adp_vals[i]:.1f}", ha="center", va="bottom",
                    fontsize=8)

        ax2 = ax.twinx()
        twin_axes.append((ax2, pct_savings))
        ax2.plot(x, pct_savings, 's--', color=PERSIAN["rose"],
                 markersize=7, linewidth=1.5, label="Reduction", zorder=5)
        for i, pct in enumerate(pct_savings):
            ax2.annotate(f"{pct:.1f}" + r"\%", (x[i], pct),
                         textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=9, fontweight="bold",
                         color=PERSIAN["rose"])

        ax.set_xticks(x)
        ax.set_xticklabels([DETECTOR_LABELS[d] for d in DETECTORS])
        ax.set_title(PLATFORM_LABELS[plat], fontweight="bold")

    axes[0].set_ylabel("Mean Total Power (W)")

    # Shared y-axis range for twin axes (reduction %)
    pct_min = min(all_pct) - 3
    pct_max = max(all_pct) + 5
    for ax2, _ in twin_axes:
        ax2.set_ylim(pct_min, pct_max)
    twin_axes[0][0].set_yticklabels([])
    twin_axes[1][0].set_ylabel(r"Power Reduction (\%)")
    twin_axes[1][0].spines["right"].set_visible(True)

    # Legend on right panel (top right, above the bars)
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = twin_axes[1][0].get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2,
                   loc="upper right", fontsize=8)

    # Title commented for the paper — caption carries the description.
# fig.suptitle(r"Mean Power: Vanilla vs \textsc{Albireo} with Reduction",
# fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "power_savings_combo.png")
    fig.savefig(OUT_DIR / "power_savings_combo.pdf")
    plt.close(fig)
    print("  [3d] power_savings_combo")


# =========================================================================
# Plot 4: Energy Savings Bar Chart
# =========================================================================

def plot_energy_savings():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    n_det = len(DETECTORS)
    x = np.arange(n_det)
    bar_width = 0.3
    plat_colors = {"thor": PERSIAN["indigo"], "orin": PERSIAN["orange"]}

    for i, plat in enumerate(PLATFORMS):
        savings = []
        errs = []
        for det in DETECTORS:
            rows = load_csv(RESULTS_DIR / plat / "albireo" / det / "summary.csv")
            systems = by_system(rows)

            van_energies = [float(r["energy_j"]) for r in systems["Vanilla"]
                            if r.get("energy_j") and r["energy_j"] != "None"]
            adp_energies = [float(r["energy_j"]) for r in systems["Adaptive"]
                            if r.get("energy_j") and r["energy_j"] != "None"]

            # Per-clip ratios (used for the SEM-based error bar) and
            # ratio-of-means (used for the central value, to match
            # Table II / Table III).
            clip_savings = []
            for ve, ae in zip(van_energies, adp_energies):
                if ve > 0:
                    clip_savings.append(100.0 * (ve - ae) / ve)

            mean_van = np.mean(van_energies)
            mean_adp = np.mean(adp_energies)
            savings.append(100.0 * (mean_van - mean_adp) / mean_van)
            errs.append(np.std(clip_savings) / np.sqrt(len(clip_savings)) * 1.96)

        offset = (i - 0.5) * bar_width
        bars = ax.bar(x + offset, savings, bar_width, yerr=errs,
                       color=plat_colors[plat], capsize=3,
                       label=PLATFORM_LABELS[plat], edgecolor="black",
                       linewidth=0.5)
        for j, (bar, val) in enumerate(zip(bars, savings)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + errs[j] + 0.5,
                    f"{val:.1f}" + r"\%", ha="center", va="bottom", fontsize=12,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([DETECTOR_LABELS[d] for d in DETECTORS])
    ax.set_ylabel(r"Mean Energy Saving vs Vanilla (\%)")
    # Title commented for the paper -- caption carries the description.
    # ax.set_title("Energy Savings Across Detectors and Platforms",
    #              fontweight="bold")
    ax.legend()
    ax.set_ylim(0, max(30, ax.get_ylim()[1]))
    ax.axhline(y=0, color="gray", linewidth=0.5)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "energy_savings.png")
    fig.savefig(OUT_DIR / "energy_savings.pdf")
    plt.close(fig)
    print("  [4/8] energy_savings")


# =========================================================================
# Plot 5: Power Trace (Single Clip Example)
# =========================================================================

RAIL_TO_GROUP = {
    "GPU": "gpu", "CPU": "cpu",
    "SOC": "io", "CV": "io", "VDDRQ": "io",
    "VDD_GPU_SOC": "gpu", "VDD_CPU_CV": "cpu",
    "VIN_SYS_5V0": "io",
    "VDD_GPU": "gpu", "VDD_CPU_SOC_MSS": "cpu",
}

TEMP_PATTERN = re.compile(r'(\w+)@([\d.]+)C', re.IGNORECASE)
GPU_UTIL_PATTERN = re.compile(r'GR3D_FREQ\s+(\d+)%')
GPU_UTIL_NVML_PATTERN = re.compile(r'GPU_UTIL\s+(\d+)%')
CPU_PATTERN = re.compile(r'CPU\s+\[([^\]]*)\]')
CORE_LOAD_PATTERN = re.compile(r'(\d+)%@\d+')
EMC_PATTERN = re.compile(r'EMC_FREQ\s+(\d+)%')


def parse_tegrastats_log(log_path):
    """Parse a per-clip tegrastats log into structured arrays."""
    times = []
    total_power = []
    gpu_power = []
    cpu_power = []
    io_power = []
    gpu_temps = []
    cpu_temps = []
    gpu_utils = []
    cpu_utils = []
    emc_utils = []
    t0 = None

    with open(log_path) as f:
        for line in f:
            parts = line.strip().split(" ", 2)
            if len(parts) < 3:
                continue
            try:
                time_main, ns_str = parts[1].split(".")
                h, m, s = time_main.split(":")
                ts = int(h) * 3600 + int(m) * 60 + int(s) + int(ns_str) / 1e9
            except (ValueError, IndexError):
                continue

            if t0 is None:
                t0 = ts
            times.append(ts - t0)

            tgs_line = parts[2]

            # Power rails
            rails = {}
            for name, cur_mw, _ in RAIL_PATTERN.findall(tgs_line):
                rails[name] = int(cur_mw) / 1000.0

            total_rail = next(
                (r for r in TOTAL_RAIL_CANDIDATES if r in rails), None)
            total_power.append(rails[total_rail] if total_rail
                               else sum(rails.values()) if rails else 0)

            groups = {"gpu": 0.0, "cpu": 0.0, "io": 0.0}
            for rname, watts in rails.items():
                grp = RAIL_TO_GROUP.get(rname)
                if grp:
                    groups[grp] += watts
            gpu_power.append(groups["gpu"])
            cpu_power.append(groups["cpu"])
            io_power.append(groups["io"])

            # Temperatures
            gt, ct = np.nan, np.nan
            for name, val in TEMP_PATTERN.findall(tgs_line):
                key = name.lower()
                if key.startswith("gpu"):
                    gt = float(val)
                elif key.startswith("cpu"):
                    ct = float(val)
            gpu_temps.append(gt)
            cpu_temps.append(ct)

            # GPU utilization
            m = GPU_UTIL_PATTERN.search(tgs_line)
            if m:
                gpu_utils.append(int(m.group(1)))
            else:
                m = GPU_UTIL_NVML_PATTERN.search(tgs_line)
                gpu_utils.append(int(m.group(1)) if m else np.nan)

            # CPU utilization
            m = CPU_PATTERN.search(tgs_line)
            if m:
                cores = CORE_LOAD_PATTERN.findall(m.group(1))
                cpu_utils.append(
                    sum(int(c) for c in cores) / len(cores) if cores else np.nan)
            else:
                cpu_utils.append(np.nan)

            # EMC utilization
            m = EMC_PATTERN.search(tgs_line)
            emc_utils.append(int(m.group(1)) if m else np.nan)

    return {
        "time": np.array(times),
        "total_power": np.array(total_power),
        "gpu_power": np.array(gpu_power),
        "cpu_power": np.array(cpu_power),
        "io_power": np.array(io_power),
        "gpu_temp": np.array(gpu_temps),
        "cpu_temp": np.array(cpu_temps),
        "gpu_util": np.array(gpu_utils),
        "cpu_util": np.array(cpu_utils),
        "emc_util": np.array(emc_utils),
    }


TRACE_CLIP = "b1c81faa-3df17267"


def _load_trace_pair():
    tgs_dir = RESULTS_DIR / "thor" / "albireo" / "yolo26x" / "tegrastats"
    if not tgs_dir.exists():
        return None, None
    van_log = tgs_dir / f"{TRACE_CLIP}_vanilla.log"
    adp_log = tgs_dir / f"{TRACE_CLIP}_adaptive.log"
    if not van_log.exists() or not adp_log.exists():
        return None, None
    return parse_tegrastats_log(van_log), parse_tegrastats_log(adp_log)


def _trace_subplot(ax, van, adp, key, ylabel, title):
    ax.plot(van["time"], van[key], color=PERSIAN["red"], alpha=0.8,
            linewidth=1.0, label="Vanilla")
    ax.plot(adp["time"], adp[key], color=PERSIAN["green"], alpha=0.8,
            linewidth=1.0, label=r"\textsc{Albireo}")

    van_vals = van[key][van[key] > 0]
    adp_vals = adp[key][adp[key] > 0]
    if len(van_vals):
        van_mean = np.mean(van_vals)
        ax.axhline(y=van_mean, color=PERSIAN["red"], linestyle=":", alpha=0.4)
    if len(adp_vals):
        adp_mean = np.mean(adp_vals)
        ax.axhline(y=adp_mean, color=PERSIAN["green"], linestyle=":", alpha=0.4)

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=24, fontweight="bold")


def plot_power_trace():
    van, adp = _load_trace_pair()
    if van is None:
        print("  [5/8] power_trace — SKIPPED")
        return

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    traces = [
        ("total_power", "Total Power (W)", "Total Board Power"),
        ("gpu_power",   "GPU Power (W)",   "GPU Domain"),
        ("cpu_power",   "CPU Power (W)",   "CPU Domain"),
        ("io_power",    "IO/MEM Power (W)", "IO/MEM Domain"),
    ]

    for ax, (key, ylabel, title) in zip(axes, traces):
        _trace_subplot(ax, van, adp, key, ylabel, title)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    clip_tex = TRACE_CLIP.replace("_", r"\_")
    # Title commented for the paper — caption carries the description.
# fig.suptitle(rf"Power Trace: Vanilla vs \textsc{{Albireo}} (Thor, YOLO26x, clip {clip_tex})",
# fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "power_trace.png")
    fig.savefig(OUT_DIR / "power_trace.pdf")
    plt.close(fig)
    print(f"  [5/8] power_trace")


# =========================================================================
# Plot 6: Thermal Comparison
# =========================================================================

def plot_thermal():
    metrics = [
        ("avg_gpu_temp_c", "Avg GPU Temp"),
        ("avg_cpu_temp_c", "Avg CPU Temp"),
        ("max_tj_temp_c",  "Max Tj Temp"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, plat in zip(axes, PLATFORMS):
        n_met = len(metrics)
        n_det = len(DETECTORS)
        x = np.arange(n_met)
        bar_width = 0.13
        det_colors = [PERSIAN["red"], PERSIAN["blue"], PERSIAN["green"]]

        for d_idx, det in enumerate(DETECTORS):
            rows = load_csv(RESULTS_DIR / plat / "albireo" / det / "summary.csv")
            systems = by_system(rows)

            for s_idx, sname in enumerate(["Vanilla", "Adaptive"]):
                sys_rows = systems.get(sname, [])
                vals = [avg(sys_rows, k) or 0 for k, _ in metrics]
                errs = [sem95(sys_rows, k) for k, _ in metrics]
                offset = (d_idx * 2 + s_idx - 2.5) * bar_width
                hatch = "" if sname == "Vanilla" else "///"
                alpha = 1.0 if sname == "Vanilla" else 0.7
                label = (f"{DETECTOR_LABELS[det]} ({sname[0]})"
                         if plat == "thor" else "")
                ax.bar(x + offset, vals, bar_width, yerr=errs,
                       color=det_colors[d_idx],
                       alpha=alpha, hatch=hatch, edgecolor="black",
                       linewidth=0.5, capsize=2, label=label)

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in metrics])
        ax.set_title(PLATFORM_LABELS[plat], fontweight="bold")

    axes[0].set_ylabel(r"Mean Temperature ($^\circ$C)")
    axes[0].legend(fontsize=7, ncol=2, loc="upper left")
    # Title commented for the paper — caption carries the description.
# fig.suptitle(r"Thermal Comparison: Vanilla vs \textsc{Albireo}",
# fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "thermal_comparison.png")
    fig.savefig(OUT_DIR / "thermal_comparison.pdf")
    plt.close(fig)
    print("  [6/8] thermal_comparison")


# =========================================================================
# Plot 7: Utilization Comparison
# =========================================================================

def plot_utilization():
    metrics = [
        ("avg_gpu_util_pct", "GPU Util"),
        ("avg_cpu_util_pct", "CPU Util"),
        ("avg_emc_util_pct", "EMC Util"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for ax, plat in zip(axes, PLATFORMS):
        n_met = len(metrics)
        x = np.arange(n_met)
        bar_width = 0.13
        det_colors = [PERSIAN["red"], PERSIAN["blue"], PERSIAN["green"]]

        for d_idx, det in enumerate(DETECTORS):
            rows = load_csv(RESULTS_DIR / plat / "albireo" / det / "summary.csv")
            systems = by_system(rows)

            for s_idx, sname in enumerate(["Vanilla", "Adaptive"]):
                sys_rows = systems.get(sname, [])
                vals = [avg(sys_rows, k) or 0 for k, _ in metrics]
                errs = [sem95(sys_rows, k) for k, _ in metrics]
                offset = (d_idx * 2 + s_idx - 2.5) * bar_width
                hatch = "" if sname == "Vanilla" else "///"
                alpha = 1.0 if sname == "Vanilla" else 0.7
                label = (f"{DETECTOR_LABELS[det]} ({sname[0]})"
                         if plat == "thor" else "")
                ax.bar(x + offset, vals, bar_width, yerr=errs,
                       color=det_colors[d_idx],
                       alpha=alpha, hatch=hatch, edgecolor="black",
                       linewidth=0.5, capsize=2, label=label)

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in metrics])
        ax.set_title(PLATFORM_LABELS[plat], fontweight="bold")

    axes[0].set_ylabel(r"Mean Utilization (\%)")
    axes[0].legend(fontsize=7, ncol=2, loc="upper right")
    # Title commented for the paper — caption carries the description.
# fig.suptitle(r"System Utilization: Vanilla vs \textsc{Albireo}",
# fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "utilization_comparison.png")
    fig.savefig(OUT_DIR / "utilization_comparison.pdf")
    plt.close(fig)
    print("  [7/10] utilization_comparison")


# =========================================================================
# Plot 8: Dual-Clip Power Trace (Dense vs Empty)
# =========================================================================

CLIP_DENSE = "b1d22ed6-f1cac061"   # 0% E, 96% AI — dense traffic, heavy rescue
CLIP_EMPTY = "b1c81faa-3df17267"   # 69% E — lots of empty road, ERD skipping


def _load_clip_pair(clip_name):
    tgs_dir = RESULTS_DIR / "thor" / "albireo" / "yolo26x" / "tegrastats"
    van_log = tgs_dir / f"{clip_name}_vanilla.log"
    adp_log = tgs_dir / f"{clip_name}_adaptive.log"
    if not van_log.exists() or not adp_log.exists():
        return None, None
    return parse_tegrastats_log(van_log), parse_tegrastats_log(adp_log)


def plot_power_trace_dual():
    dense_van, dense_adp = _load_clip_pair(CLIP_DENSE)
    empty_van, empty_adp = _load_clip_pair(CLIP_EMPTY)
    if dense_van is None or empty_van is None:
        print("  [8/10] power_trace_dual — SKIPPED")
        return

    # The 14x10 figsize is squeezed to ~3.5" column width in the paper,
    # so source fonts must be very large to render legibly after the
    # ~4x downscaling.
    saved_rc = {k: plt.rcParams[k] for k in
                ("axes.labelsize", "xtick.labelsize",
                 "ytick.labelsize", "legend.fontsize")}
    plt.rcParams.update({
        "axes.labelsize":  20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 22,
    })

    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex="col")

    traces = [
        ("total_power", "Total Power (W)", "Total Board Power"),
        ("gpu_power",   "GPU Power (W)",   "GPU Domain"),
        ("cpu_power",   "CPU Power (W)",   "CPU Domain"),
        ("io_power",    "IO/MEM Power (W)", "IO/MEM Domain"),
    ]

    # Left column: dense traffic. Y-axis labels live here only; the right
    # column shares the same units (W) row-by-row, so we suppress its
    # ylabels. Per-row domain titles ("GPU Domain", etc.) are suppressed on
    # both columns — the y-axis label already names the domain.
    for row, (key, ylabel, title) in enumerate(traces):
        _trace_subplot(axes[row, 0], dense_van, dense_adp, key, ylabel, "")

    for row, (key, ylabel, title) in enumerate(traces):
        _trace_subplot(axes[row, 1], empty_van, empty_adp, key, "", "")

    axes[0, 0].set_title(r"Dense Traffic (0\% E, 96\% AI)", fontsize=26,
                         fontweight="bold")
    axes[0, 1].set_title(r"Empty Road (69\% E, 4\% AI)", fontsize=26,
                         fontweight="bold")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")

    # Two-column legend below all subplots so it doesn't collide with curves.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=22, bbox_to_anchor=(0.5, -0.02),
               frameon=True)

    # Title commented for the paper — caption carries the description.
# fig.suptitle("Power Trace Comparison: Two Scene Types (Thor, YOLO26x)",
# fontweight="bold", fontsize=14, y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT_DIR / "power_trace_dual.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "power_trace_dual.pdf", bbox_inches="tight")
    plt.close(fig)
    plt.rcParams.update(saved_rc)
    print("  [8/10] power_trace_dual")


# =========================================================================
# Plot 9: Per-Clip Scatter — Skip Rate vs AI Rate (Option A)
# =========================================================================

def plot_scatter_skip_vs_ai():
    rows = load_csv(RESULTS_DIR / "thor" / "albireo" / "yolo26x" / "summary.csv")
    adp = [r for r in rows if r["system"] == "Adaptive"]

    e_rates = [float(r["erd_empty_rate_pct"]) for r in adp]
    p_rates = [float(r["predict_rate_pct"]) for r in adp]
    ep_rates = [e + p for e, p in zip(e_rates, p_rates)]
    ai_rates = [float(r["augmented_infer_rate_pct"]) for r in adp]

    panels = [
        (e_rates,  r"ERD Empty Rate -- E (\%)",      "Skip (E) vs Rescue"),
        (p_rates,  r"KF Predict Rate -- P (\%)",      "Skip (P) vs Rescue"),
        (ep_rates, r"Total Skip Rate -- E + P (\%)",   "Skip (E+P) vs Rescue"),
    ]
    panel_colors = [PERSIAN["green"], PERSIAN["indigo"], PERSIAN["mediumBlue"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    # Find extreme clip indices
    idx_max_e = int(np.argmax(e_rates))
    idx_max_p = int(np.argmax(p_rates))
    idx_max_ai = int(np.argmax(ai_rates))

    extremes = [
        (idx_max_e,  f"Max E ({e_rates[idx_max_e]:.0f}\\%)",   PERSIAN["green"]),
        (idx_max_p,  f"Max P ({p_rates[idx_max_p]:.0f}\\%)",   PERSIAN["indigo"]),
        (idx_max_ai, f"Max AI ({ai_rates[idx_max_ai]:.0f}\\%)", PERSIAN["red"]),
    ]

    x_arrays = [e_rates, p_rates, ep_rates]

    for ax, (x_vals, xlabel, title), color in zip(axes, panels, panel_colors):
        ax.scatter(x_vals, ai_rates, c=color, s=25, alpha=0.6,
                   edgecolors="white", linewidths=0.3)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3)

    for panel_idx in range(3):
        ax = axes[panel_idx]
        for clip_idx, label, ann_color in extremes:
            xv = x_arrays[panel_idx][clip_idx]
            yv = ai_rates[clip_idx]
            ax.scatter([xv], [yv], c=ann_color, s=80, marker='D', zorder=5,
                       edgecolors="white", linewidths=0.8)
            ax.annotate(label, (xv, yv),
                        textcoords="offset points", xytext=(8, -8),
                        fontsize=8, color=ann_color, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=ann_color,
                                        lw=0.8))

    axes[0].set_ylabel(r"Augmented Inference Rate -- AI (\%)")
    # Title commented for the paper — caption carries the description.
# fig.suptitle(r"Per-Clip \textsc{Albireo} Behavior: Skip Rate vs Rescue Rate "
# "(200 clips, Thor, YOLO26x)",
# fontweight="bold", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "scatter_skip_vs_ai.png")
    fig.savefig(OUT_DIR / "scatter_skip_vs_ai.pdf")
    plt.close(fig)
    print("  [9/10] scatter_skip_vs_ai")


# =========================================================================
# Plot 10: Per-Clip Scatter — Skip Rate vs AP@50 Delta (Option B)
# =========================================================================

def plot_scatter_skip_vs_ap_delta():
    rows = load_csv(RESULTS_DIR / "thor" / "albireo" / "yolo26x" / "summary.csv")
    van_map = {r["sequence"]: float(r["ap_50"])
               for r in rows if r["system"] == "Vanilla"}
    adp = [r for r in rows if r["system"] == "Adaptive"]

    e_rates, p_rates, ep_rates, ap_deltas = [], [], [], []
    for r in adp:
        seq = r["sequence"]
        if seq in van_map:
            e_rates.append(float(r["erd_empty_rate_pct"]))
            p_rates.append(float(r["predict_rate_pct"]))
            ep_rates.append(float(r["erd_empty_rate_pct"])
                            + float(r["predict_rate_pct"]))
            ap_deltas.append((float(r["ap_50"]) - van_map[seq]) * 100)

    e_rates = np.array(e_rates)
    p_rates = np.array(p_rates)
    ep_rates = np.array(ep_rates)
    ap_deltas = np.array(ap_deltas)

    panels = [
        (e_rates,  r"ERD Empty Rate -- E (\%)",      "E vs AP@50 Delta"),
        (p_rates,  r"KF Predict Rate -- P (\%)",      "P vs AP@50 Delta"),
        (ep_rates, r"Total Skip Rate -- E + P (\%)",   "E+P vs AP@50 Delta"),
    ]
    panel_colors = [PERSIAN["green"], PERSIAN["indigo"], PERSIAN["mediumBlue"]]

    n_better = int(np.sum(ap_deltas >= 0))
    n_worse = int(np.sum(ap_deltas < 0))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, (x_vals, xlabel, title), color in zip(axes, panels, panel_colors):
        dot_colors = [PERSIAN["green"] if d >= 0 else PERSIAN["red"]
                      for d in ap_deltas]
        ax.scatter(x_vals, ap_deltas, c=dot_colors, s=45, alpha=0.7,
                   edgecolors="white", linewidths=0.4)
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontweight="bold", fontsize=14)

    axes[0].set_ylabel(r"AP@50 Change (\textsc{Albireo} $-$ Vanilla, pp)")
    axes[2].text(0.98, 0.97,
                 r"\textsc{Albireo} better: " + f"{n_better} clips\n"
                 + r"\textsc{Albireo} worse: " + f"{n_worse} clips",
                 transform=axes[2].transAxes, fontsize=12,
                 verticalalignment="top", horizontalalignment="right",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="gray", alpha=0.8))

    # Title commented for the paper — caption carries the description.
# fig.suptitle("Per-Clip Accuracy Impact: Skip Rate vs AP@50 Delta "
# "(200 clips, Thor, YOLO26x)",
# fontweight="bold", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "scatter_skip_vs_ap_delta.png")
    fig.savefig(OUT_DIR / "scatter_skip_vs_ap_delta.pdf")
    plt.close(fig)
    print("  [10/10] scatter_skip_vs_ap_delta")


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    setup_style()
    print("Generating SEC 2026 figures...")
    print()
    print("--- System / energy / power / utilization plots ---")
    # The simple plot_pareto() is superseded by the comprehensive Pareto
    # plots in plot_pareto_full.py (called below). Kept as a function for
    # legacy / quick-look use, but not regenerated by the main entry point.
    # plot_pareto()
    plot_frame_states()
    plot_domain_energy()
    plot_domain_energy_table()
    plot_domain_power()
    plot_domain_power_table()
    plot_power_savings_combo()
    plot_energy_savings()
    plot_power_trace()
    plot_thermal()
    plot_utilization()
    plot_power_trace_dual()
    plot_scatter_skip_vs_ai()
    plot_scatter_skip_vs_ap_delta()

    print()
    print("--- Comprehensive Pareto plots (multi-system, multi-axis) ---")
    # Imported lazily so this module is importable even if plot_pareto_full
    # is missing or broken.
    from plot_pareto_full import generate_all_pareto_plots
    generate_all_pareto_plots()

    # 3D Pareto plots (plot_pareto_3d.py) are intentionally NOT called.
    # The 3D variants were exploratory and are not used in the SEC 2026
    # paper. The script remains in the repo with its __main__ commented
    # out — see plot_pareto_3d.py for details.

    print()
    print(f"Done. Figures saved to {OUT_DIR}")
