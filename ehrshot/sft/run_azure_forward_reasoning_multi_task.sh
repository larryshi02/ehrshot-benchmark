#!/bin/bash
# Wrapper to generate forward reasoning traces using Azure GPT-5-mini for multi-task fine-tuning.
# Generates unsupervised reasoning traces (no label exposure) for train and val sets.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SERIALIZED_DATA_DIR="${SERIALIZED_DATA_DIR:-${SCRIPT_DIR}/serialized_multi_task_data}"
TASK_TO_INSTRUCTIONS="${TASK_TO_INSTRUCTIONS:-${PROJECT_ROOT}/ehrshot/serialization/task_to_instructions.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/data_gpt-5-mini/sft_rt_unsupervised}"
TASKS="${TASKS:-acute_mi hyperlipidemia hypertension pancreatic_cancer}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-4096}"
MAX_WORKERS="${MAX_WORKERS:-8}"  # Number of parallel workers (1 = sequential, increase for faster processing)
RANDOM_SEED="${RANDOM_SEED:-42}"

echo "=========================================="
echo "Azure GPT-5-mini Forward Reasoning Pipeline"
echo "=========================================="
echo "Serialized Data Dir    : ${SERIALIZED_DATA_DIR}"
echo "Task Instructions      : ${TASK_TO_INSTRUCTIONS}"
echo "Output Dir             : ${OUTPUT_DIR}"
echo "Tasks                  : ${TASKS}"
echo "Temperature            : ${TEMPERATURE}"
echo "Max Completion Tokens  : ${MAX_COMPLETION_TOKENS}"
echo "Max Workers            : ${MAX_WORKERS}"
echo "Random Seed            : ${RANDOM_SEED}"
echo "=========================================="
echo ""

if [[ ! -d "${SERIALIZED_DATA_DIR}" ]]; then
  echo "ERROR: Serialized data directory not found: ${SERIALIZED_DATA_DIR}"
  exit 1
fi

if [[ ! -f "${TASK_TO_INSTRUCTIONS}" ]]; then
  echo "ERROR: Task instructions file not found: ${TASK_TO_INSTRUCTIONS}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

PYTHON_ARGS=(
  "--serialized_data_dir" "${SERIALIZED_DATA_DIR}"
  "--task_to_instructions" "${TASK_TO_INSTRUCTIONS}"
  "--output_dir" "${OUTPUT_DIR}"
  "--temperature" "${TEMPERATURE}"
  "--max_completion_tokens" "${MAX_COMPLETION_TOKENS}"
  "--max_workers" "${MAX_WORKERS}"
  "--random_seed" "${RANDOM_SEED}"
)

# shellcheck disable=SC2206
TASK_ARRAY=(${TASKS})
PYTHON_ARGS+=("--tasks" "${TASK_ARRAY[@]}")

echo "Running Azure forward reasoning pipeline..."
python "${SCRIPT_DIR}/azure_forward_reasoning_multi_task.py" "${PYTHON_ARGS[@]}"
EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
  echo ""
  echo "=========================================="
  echo "Forward reasoning pipeline completed successfully."
  echo "Output saved to: ${OUTPUT_DIR}"
  echo "=========================================="
else
  echo "ERROR: Forward reasoning pipeline failed with exit code ${EXIT_CODE}"
fi

exit $EXIT_CODE

