#!/usr/bin/env bash
# =============================================================================
# 02_create_sft/run.sh -- Create SFT datasets for plaintext and manualrubric.
#
# Assumes 01_serialize has already been run.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"

echo "=== Creating SFT datasets from plaintext serializations ==="
python "$SCRIPT_DIR/create_sft.py" \
    --input_dir "$DATA_DIR/serialized/plaintext" \
    --output_dir "$DATA_DIR/sft/plaintext"

echo ""
echo "=== Creating SFT datasets from manualrubric serializations ==="
python "$SCRIPT_DIR/create_sft.py" \
    --input_dir "$DATA_DIR/serialized/manualrubric" \
    --output_dir "$DATA_DIR/sft/manualrubric"

echo ""
echo "=== Done ==="
