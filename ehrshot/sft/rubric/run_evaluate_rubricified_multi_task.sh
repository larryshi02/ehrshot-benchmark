#!/bin/bash
# Shell script to run azure_evaluate_rubricified_multi_task.py for evaluating GPT-5 on rubricified EHRs

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Default paths
RUBRICIFIED_DIR="${PROJECT_ROOT}/ehrshot/sft/rubric/rubricified_patients"
OUTPUT_DIR="${PROJECT_ROOT}/ehrshot/sft/rubric/azure_gpt-5-mini_rubricified_results"

# Default parameters
NUM_SAMPLES=10
TEMPERATURE=1
MAX_COMPLETION_TOKENS=4096
MAX_WORKERS=15
TASKS="acute_mi hyperlipidemia hypertension pancreatic_cancer"
#TASKS="pancreatic_cancer"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --rubricified_dir)
            RUBRICIFIED_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max_completion_tokens)
            MAX_COMPLETION_TOKENS="$2"
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
        --patient_ids)
            shift
            PATIENT_IDS=""
            # Collect all remaining arguments as patient IDs until we hit another --option
            while [[ $# -gt 0 ]]; do
                if [[ "$1" =~ ^-- ]]; then
                    # Hit another option, break and let the main loop handle it
                    break
                else
                    # Add to patient IDs list
                    if [ -z "$PATIENT_IDS" ]; then
                        PATIENT_IDS="$1"
                    else
                        PATIENT_IDS="$PATIENT_IDS $1"
                    fi
                    shift
                fi
            done
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --rubricified_dir DIR       Directory containing rubricified JSON files (default: ehrshot/sft/rubric/rubricified_patients)"
            echo "  --output_dir DIR            Output directory for evaluation results (default: ehrshot/sft/rubric/azure_gpt-5-mini_rubricified_results)"
            echo "  --num_samples N             Number of samples per patient (default: 1)"
            echo "  --temperature F            Temperature for GPT-5 generation (default: 1)"
            echo "  --max_completion_tokens N   Maximum completion tokens for GPT-5 (default: 4096)"
            echo "  --max_workers N             Number of parallel workers (default: 1 = sequential)"
            echo "  --tasks TASK1 TASK2 ...     List of tasks to process (default: all tasks)"
            echo "  --patient_ids ID1 ID2 ...   Optional list of patient IDs to restrict evaluation"
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
SCRIPT_PATH="${SCRIPT_DIR}/azure_evaluate_rubricified_multi_task.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: azure_evaluate_rubricified_multi_task.py not found at $SCRIPT_PATH"
    exit 1
fi

# Check if rubricified directory exists
if [ ! -d "$RUBRICIFIED_DIR" ]; then
    echo "Error: Rubricified directory not found: $RUBRICIFIED_DIR"
    echo "Please run transform_patients.py first to generate rubricified patient files."
    exit 1
fi

# Print configuration
echo "=========================================="
echo "Evaluating GPT-5 on Rubricified EHRs"
echo "=========================================="
echo "Rubricified directory: $RUBRICIFIED_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Number of samples per patient: $NUM_SAMPLES"
echo "Temperature: $TEMPERATURE"
echo "Max completion tokens: $MAX_COMPLETION_TOKENS"
echo "Max workers: $MAX_WORKERS"
echo "Tasks to process: $TASKS"
if [ -n "$PATIENT_IDS" ]; then
    echo "Patient IDs filter: $PATIENT_IDS"
fi
echo "=========================================="
echo ""

# Build command
CMD="python \"$SCRIPT_PATH\" \
    --rubricified_dir \"$RUBRICIFIED_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --num_samples \"$NUM_SAMPLES\" \
    --temperature \"$TEMPERATURE\" \
    --max_completion_tokens \"$MAX_COMPLETION_TOKENS\" \
    --max_workers \"$MAX_WORKERS\" \
    --tasks $TASKS"

# Add patient_ids if specified
if [ -n "$PATIENT_IDS" ]; then
    CMD="$CMD --patient_ids $PATIENT_IDS"
fi

# Run the Python script
eval $CMD

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

