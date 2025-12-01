#!/bin/bash
# Shell script to run build_cohort.py for selecting diverse samples from EHRSHOT tasks

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Default paths
DATA_DIR="${PROJECT_ROOT}/ehrshot/sft/serialized_multi_task_data_raw"
OUTPUT_DIR="${PROJECT_ROOT}/ehrshot/sft/rubric/cohorts"
EMBEDDING_CACHE_DIR="${PROJECT_ROOT}/ehrshot/sft/rubric/embedding_cache"

# Default parameters
N_POS=20
N_NEG=20
MAX_INPUT_LENGTH=8192
TASK_TO_INSTRUCTIONS="${PROJECT_ROOT}/ehrshot/serialization/task_to_instructions.json"
USE_FALLBACK_ENCODER=""
TASKS="acute_mi hyperlipidemia hypertension pancreatic_cancer"
#TASKS="acute_mi"


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --embedding_cache_dir)
            EMBEDDING_CACHE_DIR="$2"
            shift 2
            ;;
        --n_pos)
            N_POS="$2"
            shift 2
            ;;
        --n_neg)
            N_NEG="$2"
            shift 2
            ;;
        --max_input_length)
            MAX_INPUT_LENGTH="$2"
            shift 2
            ;;
        --task_to_instructions)
            TASK_TO_INSTRUCTIONS="$2"
            shift 2
            ;;
        --use_fallback_encoder)
            USE_FALLBACK_ENCODER="--use_fallback_encoder"
            shift
            ;;
        --tasks)
            shift
            TASKS=""
            # Collect all remaining arguments as tasks until we hit another --option
            while [[ $# -gt 0 ]]; do
                if [[ "$1" =~ ^-- ]]; then
                    # Hit another option, break and let the main loop handle it
                    break
                else
                    # Add to tasks list
                    if [ -z "$TASKS" ]; then
                        TASKS="$1"
                    else
                        TASKS="$TASKS $1"
                    fi
                    shift
                fi
            done
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data_dir DIR              Directory containing serialized task data (default: ehrshot/sft/serialized_multi_task_data_raw)"
            echo "  --output_dir DIR            Output directory for selected cohorts (default: ehrshot/sft/rubric/cohorts)"
            echo "  --embedding_cache_dir DIR   Directory to cache embeddings (default: ehrshot/sft/rubric/embedding_cache)"
            echo "  --n_pos N                   Number of positive samples per task (default: 50)"
            echo "  --n_neg N                   Number of negative samples per task (default: 20)"
            echo "  --max_input_length N         Maximum input length for embeddings (default: 8192)"
            echo "  --task_to_instructions FILE Path to task instructions JSON file (default: ehrshot/serialization/task_to_instructions.json)"
            echo "  --use_fallback_encoder      Use GTEQwen2-1.5B instead of 7B (may avoid flash_attn issues)"
            echo "  --tasks TASK1 TASK2 ...     List of tasks to process (default: acute_mi hyperlipidemia hypertension pancreatic_cancer)"
            echo "                             Available tasks: acute_mi, hyperlipidemia, hypertension, pancreatic_cancer"
            echo "  --help                      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Change to project root
cd "$PROJECT_ROOT"

# Check if Python script exists
SCRIPT_PATH="${SCRIPT_DIR}/build_cohort.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: build_cohort.py not found at $SCRIPT_PATH"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

# Print configuration
echo "=========================================="
echo "Building Diverse Cohorts for EHRSHOT Tasks"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Embedding cache directory: $EMBEDDING_CACHE_DIR"
echo "Positive samples per task: $N_POS"
echo "Negative samples per task: $N_NEG"
echo "Max input length: $MAX_INPUT_LENGTH"
echo "Task instructions: $TASK_TO_INSTRUCTIONS"
echo "Tasks to process: $TASKS"
echo "=========================================="
echo ""

# Run the Python script
python "$SCRIPT_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --embedding_cache_dir "$EMBEDDING_CACHE_DIR" \
    --n_pos "$N_POS" \
    --n_neg "$N_NEG" \
    --max_input_length "$MAX_INPUT_LENGTH" \
    --task_to_instructions "$TASK_TO_INSTRUCTIONS" \
    $USE_FALLBACK_ENCODER \
    --tasks $TASKS

echo ""
echo "=========================================="
echo "Cohort building completed!"
echo "=========================================="

