#!/usr/bin/env bash
# ============================================================================
# run_jetson_multidetector_sweep.sh — Fill the gaps for the SEC 2026 paper
#
# Existing data (do not touch):
#   yolo26x × {Thor, Orin}: vanilla, adaptive default (u=3e-4), Albireo sweep
#                           (5e-4, 7e-4, 1e-3, 3e-3), CTD sweep, Statues sweep,
#                           ERD-only, FixedSkip-{2,3,5}.
#   yolo11x  × {Thor, Orin}: vanilla, adaptive default ONLY.
#   rfdetr-large × {Thor, Orin}: vanilla, adaptive default ONLY.
#
# This script adds, for yolo11x and rfdetr-large on the auto-detected board:
#   1. Albireo uncertainty-threshold sweep      (u = 5e-4, 7e-4, 1e-3, 3e-3)
#   2. CTD sweep                               (thresholds = 0.30, 0.50, 0.70, 0.90)
#   3. Statues sweep                           (component thresholds = 25, 75, 200, 500)
#   4. ERD-only                                (single config)
#   5. FixedSkip                               (N = 2, 3, 5)
#
# Results are written under results/<platform>/<config>/<model>/ following the
# existing layout. yolo26x runs are NOT touched.
#
# All runs use yolo_conf=0.50 — same gating threshold as every other system in
# the paper, to avoid the apples-to-oranges issue that voided the Albireo sweep.
#
# Pause/resume: any run can be interrupted with Ctrl-C. Re-running the script
# detects per-config completion via summary.csv row count and skips finished
# runs. Configs that were partially complete are restarted from clip 0 (the
# Python runner does not currently support per-clip resume within a single
# threshold value — but per-threshold and per-config resume both work).
#
# Usage:
#   ./run_jetson_multidetector_sweep.sh /path/to/bdd100k
#
# Optional second arg restricts to a single model:
#   ./run_jetson_multidetector_sweep.sh /path/to/bdd100k yolo11x
#   ./run_jetson_multidetector_sweep.sh /path/to/bdd100k rfdetr-large
#
# Auto-detects platform from hostname.
# ============================================================================
set -euo pipefail

BDD_ROOT="${1:?Usage: $0 /path/to/bdd100k [model_filter]}"
MODEL_FILTER="${2:-all}"   # "all", "yolo11x", or "rfdetr-large"

NUM_CLIPS=200
SEED=42
YOLO_CONF=0.50              # MUST stay at 0.50 — locked for apples-to-apples comparison with all
                            # prior results in this repo. Do not change without
                            # invalidating every prior comparison in this dir.
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEL_DIR="$(cd "$(dirname "$0")/../../telemetry" && pwd)"

# Models to sweep (yolo26x already complete and intentionally excluded).
ALL_MODELS=(yolo11x rfdetr-large)
if [[ "$MODEL_FILTER" == "all" ]]; then
    MODELS=("${ALL_MODELS[@]}")
elif [[ " ${ALL_MODELS[*]} " == *" $MODEL_FILTER "* ]]; then
    MODELS=("$MODEL_FILTER")
else
    echo "[ERROR] model filter must be one of: all ${ALL_MODELS[*]}"
    exit 1
fi

# Sweep configurations
SWEEP_THRESHOLDS="5e-4 7e-4 1e-3 3e-3"          # 3e-4 default already collected
N_SWEEP_THRESHOLDS=4

CTD_THRESHOLDS="0.30 0.50 0.70 0.90"            # paper Fig. 3 brackets
CTD_MAX_SKIP=8
N_CTD_THRESHOLDS=4

STATUES_COMPONENT_THRESHOLDS="25 75 200 500"    # paper default 75, bracket aggressive→strict
N_STATUES_THRESHOLDS=4

FIXEDSKIP_N="2 3 5"
N_FIXEDSKIP=3

# Auto-detect platform
HOSTNAME_LC="$(hostname | tr '[:upper:]' '[:lower:]')"
if [[ "$HOSTNAME_LC" == *"thor"* ]]; then
    PLATFORM="thor"
elif [[ "$HOSTNAME_LC" == *"orin"* ]]; then
    PLATFORM="orin"
elif [[ "$HOSTNAME_LC" == *"xavier"* ]]; then
    PLATFORM="xavier"
elif [[ "$HOSTNAME_LC" == *"polaris"* ]]; then
    PLATFORM="polaris"
else
    echo "[ERROR] Cannot auto-detect platform from hostname: $(hostname)"
    echo "  Expected hostname containing 'thor', 'orin', 'xavier', or 'polaris'"
    exit 1
fi

echo "============================================================="
echo "  Albireo Exp3 multi-detector sweep"
echo "============================================================="
echo "  Platform:           $PLATFORM"
echo "  BDD root:           $BDD_ROOT"
echo "  Models:             ${MODELS[*]}"
echo "  Clips:              $NUM_CLIPS"
echo "  Seed:               $SEED"
echo "  YOLO conf:          $YOLO_CONF  (locked — see Albireo cleanup)"
echo "  Albireo sweep:       $SWEEP_THRESHOLDS"
echo "  CTD thresholds:     $CTD_THRESHOLDS  (max-skip=$CTD_MAX_SKIP)"
echo "  Statues thresholds: $STATUES_COMPONENT_THRESHOLDS"
echo "  FixedSkip N:        $FIXEDSKIP_N"
echo ""

# ---------------------------------------------------------------------------
# Resume helper. Returns 0 (skip) if the summary.csv already has at least
# expected_rows data rows. Returns 1 (run) otherwise.
# ---------------------------------------------------------------------------
check_resume() {
    local csv="$1"
    local expected_rows="$2"
    if [[ -f "$csv" ]]; then
        local actual
        actual=$(tail -n +2 "$csv" | wc -l)
        if [[ "$actual" -ge "$expected_rows" ]]; then
            echo "  [SKIP] Already complete ($actual/$expected_rows rows)"
            return 0
        else
            echo "  [RESUME] Partial ($actual/$expected_rows rows) — restarting this config"
            return 1
        fi
    fi
    return 1
}

# ---------------------------------------------------------------------------
# Outer loop: per model
# ---------------------------------------------------------------------------
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "============================================================="
    echo "  Model: $MODEL"
    echo "============================================================="

    COMMON_ARGS="--bdd-root $BDD_ROOT --platform $PLATFORM \
        --num-clips $NUM_CLIPS --seed $SEED --yolo-conf $YOLO_CONF \
        --model $MODEL"

    # -----------------------------------------------------------------------
    # 1. Albireo uncertainty-threshold sweep
    #    Each u value is a separate run with custom --results-dir/--output-dir
    #    so the existing default Albireo (u=3e-4) results stay untouched.
    # -----------------------------------------------------------------------
    echo ""
    echo "--- [Albireo sweep] $MODEL ---"
    for U in $SWEEP_THRESHOLDS; do
        echo ""
        echo "  >> u=$U"
        SWEEP_DIR="$SRC_DIR/../results/$PLATFORM/albireo_sweep/u$U/$MODEL"
        SWEEP_OUT="$SRC_DIR/../output/$PLATFORM/albireo_sweep/u$U/$MODEL"
        if check_resume "$SWEEP_DIR/summary.csv" "$NUM_CLIPS"; then
            continue
        fi
        mkdir -p "$SWEEP_DIR" "$SWEEP_OUT"
        python3 "$SRC_DIR/run_experiment.py" \
            $COMMON_ARGS \
            --mode adaptive \
            --uncertainty-threshold "$U" \
            --results-dir "$SWEEP_DIR" \
            --output-dir  "$SWEEP_OUT"
        echo "  [DONE] Albireo u=$U $MODEL"
    done

    # -----------------------------------------------------------------------
    # 2. CTD sweep
    # -----------------------------------------------------------------------
    echo ""
    echo "--- [CTD sweep] $MODEL  thresholds={$CTD_THRESHOLDS} ---"
    CTD_DIR="$SRC_DIR/../results/$PLATFORM/ctd/$MODEL"
    if ! check_resume "$CTD_DIR/summary.csv" $((NUM_CLIPS * N_CTD_THRESHOLDS)); then
        python3 "$SRC_DIR/run_experiment.py" \
            $COMMON_ARGS \
            --mode ctd \
            --ctd-thresholds $CTD_THRESHOLDS \
            --ctd-max-skip $CTD_MAX_SKIP
        echo "  [DONE] CTD $MODEL"
    fi

    # -----------------------------------------------------------------------
    # 3. Statues sweep
    # -----------------------------------------------------------------------
    echo ""
    echo "--- [Statues sweep] $MODEL  component={$STATUES_COMPONENT_THRESHOLDS} ---"
    STATUES_DIR="$SRC_DIR/../results/$PLATFORM/statues/$MODEL"
    if ! check_resume "$STATUES_DIR/summary.csv" $((NUM_CLIPS * N_STATUES_THRESHOLDS)); then
        python3 "$SRC_DIR/run_experiment.py" \
            $COMMON_ARGS \
            --mode statues \
            --statues-component-thresholds $STATUES_COMPONENT_THRESHOLDS
        echo "  [DONE] Statues $MODEL"
    fi

    # -----------------------------------------------------------------------
    # 4. ERD-only
    # -----------------------------------------------------------------------
    echo ""
    echo "--- [ERD-only] $MODEL ---"
    ERD_DIR="$SRC_DIR/../results/$PLATFORM/erd_only/$MODEL"
    if ! check_resume "$ERD_DIR/summary.csv" $NUM_CLIPS; then
        python3 "$SRC_DIR/run_experiment.py" \
            $COMMON_ARGS \
            --mode erd_only
        echo "  [DONE] ERD-only $MODEL"
    fi

    # -----------------------------------------------------------------------
    # 5. FixedSkip (N = 2, 3, 5)
    # -----------------------------------------------------------------------
    echo ""
    echo "--- [FixedSkip] $MODEL  N={$FIXEDSKIP_N} ---"
    FIXED_DIR="$SRC_DIR/../results/$PLATFORM/fixedskip/$MODEL"
    if ! check_resume "$FIXED_DIR/summary.csv" $((NUM_CLIPS * N_FIXEDSKIP)); then
        python3 "$SRC_DIR/run_experiment.py" \
            $COMMON_ARGS \
            --mode fixedskip \
            --skip-n $FIXEDSKIP_N
        echo "  [DONE] FixedSkip $MODEL"
    fi
done

echo ""
echo "============================================================="
echo "  All multi-detector configurations complete (or skipped)"
echo "============================================================="
echo ""

# ---------------------------------------------------------------------------
# Tegrastats parsing for the newly-generated logs.
# Mirrors the glob pattern in run_jetson_baselines.sh so we capture every
# result subdir produced above.
# ---------------------------------------------------------------------------
echo "--- Parsing tegrastats logs ---"
PARSER=""
case "$PLATFORM" in
    thor)    PARSER="$TEL_DIR/tegrastats_parser_thor.py" ;;
    xavier)  PARSER="$TEL_DIR/tegrastats_parser_xavier.py" ;;
    orin)    PARSER="$TEL_DIR/tegrastats_parser.py" ;;
    polaris) PARSER="" ;;   # no tegrastats on polaris
esac

if [[ -n "$PARSER" ]]; then
    for log in \
        "$SRC_DIR"/../results/"$PLATFORM"/albireo_sweep/*/yolo11x/tegrastats_raw.log \
        "$SRC_DIR"/../results/"$PLATFORM"/albireo_sweep/*/rfdetr-large/tegrastats_raw.log \
        "$SRC_DIR"/../results/"$PLATFORM"/ctd/yolo11x/tegrastats_raw.log \
        "$SRC_DIR"/../results/"$PLATFORM"/ctd/rfdetr-large/tegrastats_raw.log \
        "$SRC_DIR"/../results/"$PLATFORM"/statues/yolo11x/tegrastats_raw.log \
        "$SRC_DIR"/../results/"$PLATFORM"/statues/rfdetr-large/tegrastats_raw.log \
        "$SRC_DIR"/../results/"$PLATFORM"/erd_only/yolo11x/tegrastats_raw.log \
        "$SRC_DIR"/../results/"$PLATFORM"/erd_only/rfdetr-large/tegrastats_raw.log \
        "$SRC_DIR"/../results/"$PLATFORM"/fixedskip/yolo11x/tegrastats_raw.log \
        "$SRC_DIR"/../results/"$PLATFORM"/fixedskip/rfdetr-large/tegrastats_raw.log
    do
        [[ -f "$log" ]] || continue
        out="${log%.log}_parsed.csv"
        if [[ -f "$out" ]]; then
            echo "  [SKIP] Already parsed: $out"
            continue
        fi
        echo "  Parsing: $log"
        python3 "$PARSER" -i "$log" -o "$out"
        echo "  Saved:   $out"
    done
fi

echo ""
echo "============================================================="
echo "  Done. Sync results back to Polaris when finished:"
echo "    rsync -avz $PLATFORM:~/albireo/experiments/exp3/results/$PLATFORM/ \\"
echo "        experiments/exp3/results/$PLATFORM/"
echo "============================================================="
