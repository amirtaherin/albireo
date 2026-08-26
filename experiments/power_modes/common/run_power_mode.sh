#!/usr/bin/env bash
# ============================================================================
# run_power_mode.sh — B1 power-mode runs for the SEC 2026 camera-ready.
#
# Runs Vanilla + Albireo (Albireo, default u=3e-4) on the primary detector
# (yolo26x) at ONE power mode. Called from a per-mode run.sh; never directly.
#
# The power mode itself must be set manually (needs a reboot):
#   sudo nvpmodel -q --verbose     # list available modes + indices
#   sudo nvpmodel -m <index>       # set mode
#   sudo reboot
# This script only VERIFIES the active mode and refuses to run on mismatch.
#
# Results go into the calling mode folder (results/ + output/), so paper
# results under exp3/results/ are never touched. Resume: if summary.csv
# already has 400 rows (200 clips x 2 systems), the run is skipped.
#
# Usage (from a mode folder):  sudo -E env "PATH=$PATH" ./run.sh /path/to/bdd100k [num_clips]
# ============================================================================
set -euo pipefail

BOARD="${1:?internal: board}"          # orin | thor
MODE_LABEL="${2:?internal: mode label}" # e.g. 15W
MODE_DIR="${3:?internal: mode dir}"
BDD_ROOT="${4:?Usage: run.sh /path/to/bdd100k [num_clips]}"
NUM_CLIPS="${5:-200}"

SEED=42
YOLO_CONF=0.50   # locked — identical across all systems in the paper
MODEL=yolo26x    # primary detector (paper Tables I-II primary configuration)
EXP_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
TEL_DIR="$(cd "$(dirname "$0")/../../../telemetry" && pwd)"

# --- board check -----------------------------------------------------------
HOSTNAME_LC="$(hostname | tr '[:upper:]' '[:lower:]')"
if [[ "$HOSTNAME_LC" != *"$BOARD"* ]]; then
    echo "[ERROR] This folder is for $BOARD but hostname is $(hostname)."
    exit 1
fi

# --- power-mode check ------------------------------------------------------
NVP_OUT="$(nvpmodel -q 2>/dev/null || sudo nvpmodel -q)"
if ! grep -qi "$MODE_LABEL" <<< "$NVP_OUT"; then
    echo "[ERROR] Active power mode does not match this folder ($MODE_LABEL)."
    echo "        nvpmodel -q reports:"
    sed 's/^/          /' <<< "$NVP_OUT"
    echo "        Set the mode and reboot first:"
    echo "          sudo nvpmodel -q --verbose   # find the index for $MODE_LABEL"
    echo "          sudo nvpmodel -m <index> && sudo reboot"
    exit 1
fi
echo "[OK] Active power mode matches: $MODE_LABEL"

RESULTS_DIR="$MODE_DIR/results"
OUTPUT_DIR="$MODE_DIR/output"
mkdir -p "$RESULTS_DIR" "$OUTPUT_DIR"

# --- provenance ------------------------------------------------------------
{
    echo "date:      $(date -Is)"
    echo "hostname:  $(hostname)"
    echo "mode:      $MODE_LABEL"
    echo "model:     $MODEL   clips: $NUM_CLIPS   seed: $SEED   conf: $YOLO_CONF"
    echo "--- nvpmodel -q ---"
    echo "$NVP_OUT"
    echo "--- jetson_clocks --show ---"
    jetson_clocks --show 2>/dev/null || sudo jetson_clocks --show 2>/dev/null || echo "(unavailable)"
} > "$RESULTS_DIR/power_mode_info.txt"

# --- resume check: 200 clips x 2 systems = 400 rows ------------------------
CSV="$RESULTS_DIR/summary.csv"
EXPECTED=$((NUM_CLIPS * 2))
if [[ -f "$CSV" ]]; then
    ACTUAL=$(tail -n +2 "$CSV" | wc -l)
    if [[ "$ACTUAL" -ge "$EXPECTED" ]]; then
        echo "[SKIP] Already complete ($ACTUAL/$EXPECTED rows): $CSV"
        exit 0
    fi
    echo "[RESUME] Partial ($ACTUAL/$EXPECTED rows) — restarting this mode"
fi

# --- run -------------------------------------------------------------------
echo "[RUN] $BOARD $MODE_LABEL: vanilla+adaptive, $MODEL, $NUM_CLIPS clips"
python3 "$EXP_DIR/run_experiment.py" \
    --bdd-root "$BDD_ROOT" --platform "$BOARD" \
    --num-clips "$NUM_CLIPS" --seed "$SEED" --yolo-conf "$YOLO_CONF" \
    --model "$MODEL" \
    --mode vanilla+adaptive \
    --results-dir "$RESULTS_DIR" \
    --output-dir  "$OUTPUT_DIR"

# --- tegrastats parsing ----------------------------------------------------
case "$BOARD" in
    thor) PARSER="$TEL_DIR/tegrastats_parser_thor.py" ;;
    orin) PARSER="$TEL_DIR/tegrastats_parser.py" ;;
esac
LOG="$RESULTS_DIR/tegrastats_raw.log"
if [[ -f "$LOG" && ! -f "${LOG%.log}_parsed.csv" ]]; then
    python3 "$PARSER" -i "$LOG" -o "${LOG%.log}_parsed.csv"
    echo "[OK] tegrastats parsed"
fi

echo "[DONE] $BOARD $MODE_LABEL"
