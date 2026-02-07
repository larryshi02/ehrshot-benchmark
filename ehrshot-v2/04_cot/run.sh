#!/usr/bin/env bash
# =============================================================================
# 04_cot/run.sh -- Generate supervised and unsupervised CoT reasoning traces.
#
# Assumes 02_create_sft (plaintext) has been run.
# Edit --tasks to select which tasks to generate traces for.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"

# By default, generate for all 15 tasks. Override with e.g.:
#   TASKS="guo_icu new_acutemi" bash 04_cot/run.sh
TASKS="${TASKS:-guo_icu guo_los guo_readmission lab_thrombocytopenia lab_hyperkalemia lab_hypoglycemia lab_hyponatremia lab_anemia new_hypertension new_hyperlipidemia new_pancan new_celiac new_lupus new_acutemi chexpert}"

echo "=== Generating supervised CoT traces (train + val) ==="
python "$SCRIPT_DIR/generate_supervised_cot.py" \
    --sft_dir "$DATA_DIR/sft/plaintext" \
    --output_dir "$DATA_DIR/sft/cot_supervised" \
    --tasks $TASKS \
    --splits train val

echo ""
echo "=== Generating unsupervised CoT traces (train + val + test) ==="
python "$SCRIPT_DIR/generate_unsupervised_cot.py" \
    --sft_dir "$DATA_DIR/sft/plaintext" \
    --output_dir "$DATA_DIR/sft/cot_unsupervised" \
    --tasks $TASKS \
    --splits train val test

echo ""
echo "=== Done ==="
