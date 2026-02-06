#!/bin/bash
# Smoke test with low concurrency and short duration
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== saturated-blitz-bench quick smoke test ==="
echo ""

BASE_URL="${1:-http://localhost:8000/v1}"
MODEL="${2:-deepseek-ai/DeepSeek-V3}"

echo "Endpoint: $BASE_URL"
echo "Model:    $MODEL"
echo ""

saturated-blitz-bench run \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --concurrency 2 \
    --duration 60 \
    --warmup 5 \
    --format both \
    --verbose
