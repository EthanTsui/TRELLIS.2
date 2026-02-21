#!/bin/bash
# Resilient GA runner: restarts Python process on CUDA crash
# Usage: bash optimization/scripts/run_ga_resilient.sh [--generations N] [--population-size K]

GENERATIONS=${1:-5}
POP_SIZE=${2:-25}
EXAMPLES=${3:-"4,7"}
MAX_RESTARTS=20
RESTART_COUNT=0
LOG_DIR="/workspace/TRELLIS.2/optimization"

echo "=== Resilient GA Runner ==="
echo "Generations: $GENERATIONS | Pop size: $POP_SIZE | Examples: $EXAMPLES | Max restarts: $MAX_RESTARTS"
echo ""

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    RESTART_COUNT=$((RESTART_COUNT + 1))
    LOG_FILE="$LOG_DIR/ga_run_${RESTART_COUNT}.log"

    echo "[$(date '+%H:%M:%S')] Starting GA (attempt $RESTART_COUNT/$MAX_RESTARTS)..."

    python optimization/scripts/genetic_optimizer.py \
        --generations "$GENERATIONS" \
        --population-size "$POP_SIZE" \
        --examples "$EXAMPLES" \
        > "$LOG_FILE" 2>&1
    EXIT_CODE=$?

    # Check if GA completed normally
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] GA completed successfully!"
        cat "$LOG_FILE" | grep -E 'Fitness:|Best|Generation|Evolv'
        break
    fi

    # Check for CUDA errors
    if grep -q 'CUDA error\|illegal memory access\|cudaStreamSynchronize' "$LOG_FILE"; then
        echo "[$(date '+%H:%M:%S')] CUDA crash detected (exit code $EXIT_CODE)"
        echo "  Last fitness scores:"
        grep 'Fitness:' "$LOG_FILE" | tail -5
        echo "  Restarting in 5 seconds..."
        sleep 5
        continue
    fi

    # Check for other Python errors
    if grep -q 'Traceback\|Error\|Exception' "$LOG_FILE"; then
        echo "[$(date '+%H:%M:%S')] Python error (exit code $EXIT_CODE)"
        grep -A 2 'Error\|Exception' "$LOG_FILE" | tail -10
        echo "  Restarting in 5 seconds..."
        sleep 5
        continue
    fi

    echo "[$(date '+%H:%M:%S')] Unknown exit (code $EXIT_CODE). Stopping."
    break
done

echo ""
echo "=== Resilient GA Runner finished ==="
echo "Total restarts: $RESTART_COUNT"

# Print final summary
python3 -c "
import json
with open('/workspace/TRELLIS.2/optimization/research/population.json') as f:
    pop = json.load(f)
evaluated = [p for p in pop['individuals'] if p.get('fitness') and isinstance(p['fitness'], dict) and p['fitness'].get('overall', 0) > 0]
evaluated.sort(key=lambda x: x['fitness']['overall'], reverse=True)
print(f'Total evaluated: {len(evaluated)} individuals')
print(f'Top 5:')
for p in evaluated[:5]:
    print(f'  {p[\"id\"]:30s} fitness={p[\"fitness\"][\"overall\"]:.1f}')
"
