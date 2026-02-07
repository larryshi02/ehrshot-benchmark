#!/usr/bin/env bash
# =============================================================================
# 01_serialize/run.sh -- Serialize EHRs into plaintext and manualrubric formats.
#
# Usage:
#   bash 01_serialize/run.sh
#
# Edit the paths below to match your environment.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DB_PATH="${EHRSHOT_DB:-/home/demirel/clinical-reasoning/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract}"
LABELS_DIR="${EHRSHOT_LABELS:-/home/demirel/clinical-reasoning/ehrshot-benchmark/EHRSHOT_ASSETS/labels}"
SPLITS_FILE="${EHRSHOT_SPLITS:-/home/demirel/clinical-reasoning/ehrshot-benchmark/EHRSHOT_ASSETS/splits/person_id_map.csv}"
DATA_DIR="${SCRIPT_DIR}/../data/serialized"

echo "=== Generating plaintext serializations ==="
python "$SCRIPT_DIR/serialize.py" \
    --path_to_database "$DB_PATH" \
    --path_to_labels_dir "$LABELS_DIR" \
    --path_to_splits "$SPLITS_FILE" \
    --output_dir "$DATA_DIR/plaintext" \
    --mode plaintext \
    --force

echo ""
echo "=== Generating manualrubric serializations ==="
python "$SCRIPT_DIR/serialize.py" \
    --path_to_database "$DB_PATH" \
    --path_to_labels_dir "$LABELS_DIR" \
    --path_to_splits "$SPLITS_FILE" \
    --output_dir "$DATA_DIR/manualrubric" \
    --mode manualrubric \
    --force

echo ""
echo "=== Done. Outputs in $DATA_DIR ==="
