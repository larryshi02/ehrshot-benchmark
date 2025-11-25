#!/bin/bash
# Shell script to run create_rubric.py for generating rubric instructions

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Default paths
COHORT_DIR="${PROJECT_ROOT}/ehrshot/sft/rubric/cohorts"
OUTPUT_DIR="${PROJECT_ROOT}/ehrshot/sft/rubric/rubrics"
TASK_TO_INSTRUCTIONS="${PROJECT_ROOT}/ehrshot/serialization/task_to_instructions.json"

# Default parameters
MAX_COMPLETION_TOKENS=8192
TEMPERATURE=1
#TASKS="acute_mi hyperlipidemia hypertension pancreatic_cancer"
TASKS="acute_mi"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cohort_dir)
            COHORT_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --task_to_instructions)
            TASK_TO_INSTRUCTIONS="$2"
            shift 2
            ;;
        --max_completion_tokens)
            MAX_COMPLETION_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
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
            echo "  --cohort_dir DIR           Directory containing cohort JSON files (default: ehrshot/sft/rubric/cohorts)"
            echo "  --output_dir DIR          Output directory for rubric files (default: ehrshot/sft/rubric/rubrics)"
            echo "  --task_to_instructions FILE Path to task instructions JSON file (default: ehrshot/serialization/task_to_instructions.json)"
            echo "  --max_completion_tokens N  Maximum completion tokens for GPT-5 (default: 8192)"
            echo "  --temperature F           Temperature for GPT-5 generation (default: 1)"
            echo "  --tasks TASK1 TASK2 ...   List of tasks to process (default: all tasks)"
            echo "  --help                    Show this help message"
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
SCRIPT_PATH="${SCRIPT_DIR}/create_rubric.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: create_rubric.py not found at $SCRIPT_PATH"
    exit 1
fi

# Check if cohort directory exists
if [ ! -d "$COHORT_DIR" ]; then
    echo "Error: Cohort directory not found: $COHORT_DIR"
    echo "Please run build_cohort.py first to generate cohort files."
    exit 1
fi

# Print configuration
echo "=========================================="
echo "Creating Rubric Instructions for EHRSHOT Tasks"
echo "=========================================="
echo "Cohort directory: $COHORT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Task instructions: $TASK_TO_INSTRUCTIONS"
echo "Max completion tokens: $MAX_COMPLETION_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "Tasks to process: $TASKS"
echo "=========================================="
echo ""

# Run the Python script
python "$SCRIPT_PATH" \
    --cohort_dir "$COHORT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --task_to_instructions "$TASK_TO_INSTRUCTIONS" \
    --max_completion_tokens "$MAX_COMPLETION_TOKENS" \
    --temperature "$TEMPERATURE" \
    --tasks $TASKS

echo ""
echo "=========================================="
echo "Rubric generation completed!"
echo "=========================================="

