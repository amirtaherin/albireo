#!/usr/bin/env bash
# ============================================================================
# run_jetson_experiments.sh — Run Albireo Exp3 on Jetson boards
#
# Usage: ./run_jetson_experiments.sh /path/to/bdd100k
#
# Auto-detects platform from hostname. Supports interrupt+resume: skips
# configurations where summary.csv already has the expected number of rows.
# ============================================================================
set -euo pipefail

BDD_ROOT="${1:?Usage: $0 /path/to/bdd100k}"
NUM_CLIPS=200
SEED=42
YOLO_CONF=0.50
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEL_DIR="$(cd "$(dirname "$0")/../../telemetry" && pwd)"

# Auto-detect platform
HOSTNAME_LC="$(hostname | tr '[:upper:]' '[:lower:]')"
if [[ "$HOSTNAME_LC" == *"thor"* ]]; then
    PLATFORM="thor"
elif [[ "$HOSTNAME_LC" == *"orin"* ]]; then
    PLATFORM="orin"
elif [[ "$HOSTNAME_LC" == *"xavier"* ]]; then
    PLATFORM="xavier"
else
    echo "[ERROR] Cannot auto-detect platform from hostname: $(hostname)"
    echo "  Expected hostname containing 'thor', 'orin', or 'xavier'"
    exit 1
fi

echo "=== Albireo Exp3 Jetson Runner ==="
echo "  Platform:  $PLATFORM"
echo "  BDD root:  $BDD_ROOT"
echo "  Clips:     $NUM_CLIPS"
echo "  Seed:      $SEED"
echo "  Idea:      $IDEA"
echo "  YOLO conf: $YOLO_CONF"
echo ""

COMMON_ARGS="--bdd-root $BDD_ROOT --platform $PLATFORM \
    --num-clips $NUM_CLIPS --seed $SEED --yolo-conf $YOLO_CONF"

# Resume check: count data rows in summary.csv (header + N_CLIPS * num_systems)
check_resume() {
    local csv="$1"
    local expected_rows="$2"
    if [[ -f "$csv" ]]; then
        local actual
        actual=$(tail -n +2 "$csv" | wc -l)
        if [[ "$actual" -ge "$expected_rows" ]]; then
            echo "  [SKIP] Already complete ($actual/$expected_rows rows in summary.csv)"
            return 0
        else
            echo "  [RESUME] Partial results found ($actual/$expected_rows rows), re-running..."
            return 1
        fi
    fi
    return 1
}

run_config() {
    local mode="$1"
    local model="$2"
    local label="$3"
    local extra_args="${4:-}"
    local expected_rows="$5"

    echo ""
    echo "--- [$label] mode=$mode model=$model ---"

    local results_dir
    if [[ "$mode" == "fixedskip" ]]; then
        results_dir="$SRC_DIR/../results/$PLATFORM/fixedskip/$model"
    else
        results_dir="$SRC_DIR/../results/$PLATFORM/albireo/$model"
    fi

    if check_resume "$results_dir/summary.csv" "$expected_rows"; then
        return 0
    fi

    python3 "$SRC_DIR/run_experiment.py" \
        $COMMON_ARGS \
        --model "$model" \
        --mode "$mode" \
        $extra_args

    echo "  [DONE] $label"
}

# Config 1: Vanilla + Adaptive with YOLO26x (primary detector)
# Expected rows: NUM_CLIPS * 2 (Vanilla + Adaptive)
run_config "vanilla+adaptive" "yolo26x" "YOLO26x V+A" "" $((NUM_CLIPS * 2))

# Config 2: Fixed-skip baselines with YOLO26x (N=2,3,5)
# Expected rows: NUM_CLIPS * 3 (one per skip-N)
run_config "fixedskip" "yolo26x" "YOLO26x FixedSkip" "--skip-n 2 3 5" $((NUM_CLIPS * 3))

# Config 3: Vanilla + Adaptive with YOLO11x (secondary — detector-agnostic)
run_config "vanilla+adaptive" "yolo11x" "YOLO11x V+A" "" $((NUM_CLIPS * 2))

# Config 4: Vanilla + Adaptive with RF-DETR-Large (secondary — detector-agnostic)
run_config "vanilla+adaptive" "rfdetr-large" "RF-DETR V+A" "" $((NUM_CLIPS * 2))

echo ""
echo "=== All configurations complete ==="
echo ""

# Parse tegrastats logs
echo "--- Parsing tegrastats logs ---"
for log in "$SRC_DIR"/../results/"$PLATFORM"/*/tegrastats_raw.log \
           "$SRC_DIR"/../results/"$PLATFORM"/*/*/tegrastats_raw.log; do
    if [[ -f "$log" ]]; then
        out="${log%.log}_parsed.csv"
        if [[ -f "$out" ]]; then
            echo "  [SKIP] Already parsed: $out"
            continue
        fi
        echo "  Parsing: $log"
        if [[ "$PLATFORM" == "thor" ]]; then
            python3 "$TEL_DIR/tegrastats_parser_thor.py" -i "$log" -o "$out"
        elif [[ "$PLATFORM" == "xavier" ]]; then
            python3 "$TEL_DIR/tegrastats_parser_xavier.py" -i "$log" -o "$out"
        else
            python3 "$TEL_DIR/tegrastats_parser.py" -i "$log" -o "$out"
        fi
        echo "  Saved: $out"
    fi
done

echo ""
echo "=== Done ==="
