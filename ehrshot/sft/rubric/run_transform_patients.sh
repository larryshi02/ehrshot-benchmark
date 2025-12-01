#!/bin/bash
# Shell script to run transform_patients.py for transforming patient contexts into rubric format

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Default paths
RUBRIC_DIR="${PROJECT_ROOT}/ehrshot/sft/rubric/rubrics"
DATA_DIR="${PROJECT_ROOT}/ehrshot/sft/serialized_multi_task_data"
OUTPUT_DIR="${PROJECT_ROOT}/ehrshot/sft/rubric/rubricified_patients"

# Default parameters
NUM_SAMPLES=-1
SPLITS="train val test"
MAX_COMPLETION_TOKENS=16384
TEMPERATURE=1
RANDOM_SEED=42
MAX_WORKERS=20
#TASKS="acute_mi hyperlipidemia hypertension pancreatic_cancer"
TASKS="pancreatic_cancer"
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --rubric_dir)
            RUBRIC_DIR="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_samples)
            if [ "$2" = "-1" ]; then
                NUM_SAMPLES="-1"
            elif [ -z "$2" ] || [ "$2" = "" ]; then
                NUM_SAMPLES=""
            else
                NUM_SAMPLES="$2"
            fi
            shift 2
            ;;
        --splits)
            shift
            SPLITS=""
            # Collect all remaining arguments as splits until we hit another --option
            while [[ $# -gt 0 ]]; do
                if [[ "$1" =~ ^-- ]]; then
                    # Hit another option, break and let the main loop handle it
                    break
                else
                    # Add to splits list
                    if [ -z "$SPLITS" ]; then
                        SPLITS="$1"
                    else
                        SPLITS="$SPLITS $1"
                    fi
                    shift
                fi
            done
            ;;
        --max_completion_tokens)
            MAX_COMPLETION_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --random_seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        --max_workers)
            MAX_WORKERS="$2"
            shift 2
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
            echo "  --rubric_dir DIR          Directory containing rubric JSON files (default: ehrshot/sft/rubric/rubrics)"
            echo "  --data_dir DIR            Directory containing serialized patient data (default: ehrshot/sft/serialized_multi_task_data)"
            echo "  --output_dir DIR          Output directory for rubricified patients (default: ehrshot/sft/rubric/rubricified_patients)"
            echo "  --num_samples N           Number of samples per split: n = n positive + n negative (2n total), -1 = all (default: all)"
            echo "  --splits SPLIT1 SPLIT2 ... List of splits to process: train, val, test (default: train)"
            echo "  --max_completion_tokens N  Maximum completion tokens for GPT-5 (default: 4096)"
            echo "  --temperature F            Temperature for GPT-5 generation (default: 0.3)"
            echo "  --random_seed N           Random seed for sampling (default: 42)"
            echo "  --max_workers N           Number of parallel workers for concurrent API calls (default: 1 = sequential)"
            echo "  --tasks TASK1 TASK2 ...   List of tasks to process (default: all tasks)"
            echo "  --help                     Show this help message"
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
SCRIPT_PATH="${SCRIPT_DIR}/transform_patients.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: transform_patients.py not found at $SCRIPT_PATH"
    exit 1
fi

# Check if rubric directory exists
if [ ! -d "$RUBRIC_DIR" ]; then
    echo "Error: Rubric directory not found: $RUBRIC_DIR"
    echo "Please run create_rubric.py first to generate rubric files."
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

# Print configuration
echo "=========================================="
echo "Transforming Patients to Rubric Format"
echo "=========================================="
echo "Rubric directory: $RUBRIC_DIR"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
if [ -z "$NUM_SAMPLES" ] || [ "$NUM_SAMPLES" = "" ]; then
    echo "Number of samples per split: all"
elif [ "$NUM_SAMPLES" = "-1" ]; then
    echo "Number of samples per split: all (-1)"
else
    echo "Number of samples per split: $NUM_SAMPLES positive + $NUM_SAMPLES negative = $((NUM_SAMPLES * 2)) total"
fi
echo "Splits to process: $SPLITS"
echo "Max completion tokens: $MAX_COMPLETION_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "Random seed: $RANDOM_SEED"
echo "Max workers: $MAX_WORKERS"
echo "Tasks to process: $TASKS"
echo "=========================================="
echo ""

# Build command
CMD="python \"$SCRIPT_PATH\" \
    --rubric_dir \"$RUBRIC_DIR\" \
    --data_dir \"$DATA_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --splits $SPLITS \
    --max_completion_tokens \"$MAX_COMPLETION_TOKENS\" \
    --temperature \"$TEMPERATURE\" \
    --random_seed \"$RANDOM_SEED\" \
    --max_workers \"$MAX_WORKERS\" \
    --tasks $TASKS"

# Add num_samples if specified (including -1)
if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples \"$NUM_SAMPLES\""
fi

# Run the Python script
eval $CMD

echo ""
echo "=========================================="
echo "Patient transformation completed!"
echo "=========================================="

