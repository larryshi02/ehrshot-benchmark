#!/bin/bash
# Bash script to serialize patient data for 4 tasks with all splits

set -e  # Exit on error

# Default paths (adjust these to your environment)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
DATABASE_PATH="${DATABASE_PATH:-${PROJECT_ROOT}/EHRSHOT_ASSETS/femr/extract}"
LABELS_DIR="${LABELS_DIR:-${PROJECT_ROOT}/EHRSHOT_ASSETS/benchmark}"
SPLITS_FILE="${SPLITS_FILE:-${PROJECT_ROOT}/EHRSHOT_ASSETS/splits/person_id_map.csv}"
TASK_INSTRUCTIONS="${PROJECT_ROOT}/ehrshot/serialization/task_to_instructions.json"
OUTPUT_DIR="${SCRIPT_DIR}/serialized_multi_task_data"

# Serialization configuration
EXCLUDED_ONTOLOGIES="LOINC,Domain,CARE_SITE,ICDO3,Medicare Specialty,CMS Place of Service,OMOP Extension,Condition Type"
NUM_AGGREGATED_EVENTS=3

echo "=========================================="
echo "Multi-Task Data Serialization"
echo "=========================================="
echo "Database: $DATABASE_PATH"
echo "Labels Dir: $LABELS_DIR"
echo "Splits File: $SPLITS_FILE"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Check if required files/directories exist
if [ ! -d "$DATABASE_PATH" ]; then
    echo "ERROR: Database directory not found: $DATABASE_PATH"
    exit 1
fi

if [ ! -d "$LABELS_DIR" ]; then
    echo "ERROR: Labels directory not found: $LABELS_DIR"
    exit 1
fi

if [ ! -f "$SPLITS_FILE" ]; then
    echo "ERROR: Splits file not found: $SPLITS_FILE"
    exit 1
fi

if [ ! -f "$TASK_INSTRUCTIONS" ]; then
    echo "WARNING: Task instructions file not found: $TASK_INSTRUCTIONS"
    echo "Will proceed without custom instructions"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check NumPy version and warn if needed
echo "Checking NumPy version..."
python -c "import numpy as np; version = np.__version__; major = int(version.split('.')[0]); exit(0 if major < 2 else 1)" 2>/dev/null || {
    echo "WARNING: NumPy 2.x detected. Some dependencies may be incompatible."
    echo "Consider downgrading: pip install 'numpy<2.0'"
}

# Run serialization
echo "Starting serialization..."
python "$SCRIPT_DIR/serialize_multi_task_data.py" \
    --path_to_database "$DATABASE_PATH" \
    --path_to_labels_dir "$LABELS_DIR" \
    --path_to_splits "$SPLITS_FILE" \
    --task_to_instructions "$TASK_INSTRUCTIONS" \
    --excluded_ontologies "$EXCLUDED_ONTOLOGIES" \
    --num_aggregated_events "$NUM_AGGREGATED_EVENTS" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Serialization completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

