#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STEPS=50
LOG_DIR="logs/llama8B_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== Running all llama8B configs with $STEPS steps ==="
echo "Log directory: $LOG_DIR"
echo ""

for config in model_config/llama8B*.json; do
    name="$(basename "$config" .json)"
    echo "--- Starting: $name ---"

    bash run_launcher.sh "$config" false "$STEPS" \
        >"$LOG_DIR/${name}.stdout.log" \
        2>"$LOG_DIR/${name}.stderr.log" \
    && echo "    [DONE]  $name" \
    || echo "    [FAIL]  $name (exit $?)"
done

echo ""
echo "=== All runs complete. Logs in $LOG_DIR ==="
