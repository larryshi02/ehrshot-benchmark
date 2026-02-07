#!/usr/bin/env bash
# =============================================================================
# 03_rubric/run.sh -- Full rubric pipeline: cohort -> rubric -> apply -> SFT.
#
# Assumes 01_serialize (plaintext) has been run.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
RUBRIC_DIR="${DATA_DIR}/rubric"

echo "=== Step 1: Build cohorts (k-means + medoid selection) ==="
python "$SCRIPT_DIR/build_cohort.py" \
    --input_dir "$DATA_DIR/serialized/plaintext" \
    --output_dir "$RUBRIC_DIR"

echo ""
echo "=== Step 2: Generate rubric instructions via GPT-5-mini ==="
python "$SCRIPT_DIR/create_rubric.py" \
    --cohort_dir "$RUBRIC_DIR" \
    --output_dir "$RUBRIC_DIR"

echo ""
echo "=== Step 3: Apply rubric to all patients via GPT-5-mini ==="
python "$SCRIPT_DIR/apply_rubric.py" \
    --rubric_dir "$RUBRIC_DIR" \
    --serialized_dir "$DATA_DIR/serialized/plaintext" \
    --output_dir "$RUBRIC_DIR/rubricified"

echo ""
echo "=== Step 4: Create SFT datasets from rubricified data ==="
python "$SCRIPT_DIR/create_llmrubric_sft.py" \
    --input_dir "$RUBRIC_DIR/rubricified" \
    --output_dir "$DATA_DIR/sft/llmrubric"

echo ""
echo "=== Done. Rubric data in $RUBRIC_DIR ==="
