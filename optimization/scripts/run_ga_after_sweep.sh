#!/bin/bash
# Wait for auto_optimize to finish, then run GA optimizer
# Usage: docker exec trellis2-opt bash /workspace/TRELLIS.2/optimization/scripts/run_ga_after_sweep.sh

set -e

SCRIPTS_DIR="/workspace/TRELLIS.2/optimization/scripts"
RESULTS_DIR="/workspace/TRELLIS.2/optimization/run_20260216_062154/results"

echo "=== GA After Sweep Runner ==="
echo "Waiting for auto_optimize to complete..."

# Wait until no auto_optimize process is running
while pgrep -f "auto_optimize" > /dev/null 2>&1; do
    CURRENT=$(ls "$RESULTS_DIR" 2>/dev/null | wc -l)
    echo "  auto_optimize still running... ($CURRENT iterations completed)"
    sleep 60
done

echo "auto_optimize completed!"
TOTAL=$(ls "$RESULTS_DIR" 2>/dev/null | wc -l)
echo "Total sweep iterations: $TOTAL"

# Ingest sweep results into GA population
echo ""
echo "=== Starting GA Optimizer ==="
echo "Evaluating Generation 1 candidates..."

cd /workspace/TRELLIS.2
python "$SCRIPTS_DIR/genetic_optimizer.py" \
    --generations 5 \
    --population-size 25 \
    --examples 4,6,7 \
    --seed 42

echo ""
echo "=== GA Optimization Complete ==="
