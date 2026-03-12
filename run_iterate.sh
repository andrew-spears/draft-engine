#!/bin/bash
# Iterative training: generate data with NN leaf -> train -> repeat.
#
# Usage:
#   ./run_iterate.sh                  # 5 rounds, defaults
#   ./run_iterate.sh 10               # 10 rounds
#   ./run_iterate.sh 5 value_net.pt   # start from existing model
#
# Each round:
#   1. Generate data using previous model as leaf evaluator (or pure search for round 1)
#   2. Train on ALL accumulated data, continuing from previous weights

set -e

ROUNDS=${1:-5}
START_MODEL=${2:-""}
GAMES=5000
DEPTH=2
FANOUT=20
EPOCHS=100
OUTDIR="iterations"

mkdir -p "$OUTDIR"

echo "=== Iterative training: $ROUNDS rounds ==="
echo "  Games/round: $GAMES, depth: $DEPTH, fanout: $FANOUT, epochs: $EPOCHS"
echo ""

for i in $(seq 1 "$ROUNDS"); do
    echo "========================================"
    echo "=== Round $i / $ROUNDS ==="
    echo "========================================"

    DATA_FILE="$OUTDIR/data_r${i}.npz"
    MODEL_FILE="$OUTDIR/model_r${i}.pt"

    # Determine leaf model: use start model for round 1, previous round's model after
    if [ "$i" -eq 1 ] && [ -n "$START_MODEL" ]; then
        LEAF_FLAG="--leaf-model $START_MODEL"
        RESUME_FLAG="--resume $START_MODEL"
    elif [ "$i" -gt 1 ]; then
        PREV_MODEL="$OUTDIR/model_r$((i-1)).pt"
        LEAF_FLAG="--leaf-model $PREV_MODEL"
        RESUME_FLAG="--resume $PREV_MODEL"
    else
        LEAF_FLAG=""
        RESUME_FLAG=""
    fi

    # Generate data
    echo ""
    echo "--- Data generation (round $i) ---"
    uv run python run_datagen.py \
        --games "$GAMES" --depth "$DEPTH" --fanout "$FANOUT" \
        $LEAF_FLAG --out "$DATA_FILE"

    # Train on ALL data so far
    ALL_DATA=$(ls "$OUTDIR"/data_r*.npz | sort -V)
    echo ""
    echo "--- Training (round $i) ---"
    echo "  Data files: $ALL_DATA"
    uv run python run_train.py $ALL_DATA \
        --epochs "$EPOCHS" --output "$MODEL_FILE" $RESUME_FLAG

    echo ""
    echo "Round $i complete: $MODEL_FILE"
    echo ""
done

echo "========================================"
echo "Done. Final model: $OUTDIR/model_r${ROUNDS}.pt"
echo "All data: $(ls $OUTDIR/data_r*.npz)"
echo "All models: $(ls $OUTDIR/model_r*.pt)"
