#!/bin/bash
# Wrapper to evaluate Azure GPT-5 on the EHRShot multi-task benchmark.
# Mirrors run_base_model_multi_task_evaluation.sh but targets Azure OpenAI completions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SERIALIZED_DATA_DIR="${SERIALIZED_DATA_DIR:-${SCRIPT_DIR}/serialized_multi_task_data}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/azure_gpt-5-mini_multi_task_results}"
#TASKS="${TASKS:-acute_mi hyperlipidemia hypertension pancreatic_cancer}"
TASKS="${TASKS:-hyperlipidemia hypertension pancreatic_cancer}"
PATIENT_IDS="${PATIENT_IDS:-}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
TEMPERATURE="${TEMPERATURE:-1}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-4096}"
MAX_WORKERS="${MAX_WORKERS:-8}"  # Number of parallel workers (1 = sequential, increase for faster processing)

echo "=========================================="
echo "Azure GPT-5 Multi-Task Evaluation"
echo "=========================================="
echo "Serialized Data Dir : ${SERIALIZED_DATA_DIR}"
echo "Output Dir          : ${OUTPUT_DIR}"
echo "Tasks               : ${TASKS}"
if [[ -n "${PATIENT_IDS}" ]]; then
  echo "Patient IDs Filter  : ${PATIENT_IDS}"
fi
echo "Num Samples         : ${NUM_SAMPLES}"
echo "Temperature         : ${TEMPERATURE}"
echo "Max Completion Tokens          : ${MAX_COMPLETION_TOKENS}"
echo "Max Workers         : ${MAX_WORKERS}"
echo "=========================================="
echo ""

if [[ ! -d "${SERIALIZED_DATA_DIR}" ]]; then
  echo "ERROR: Serialized data directory not found: ${SERIALIZED_DATA_DIR}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

PYTHON_ARGS=(
  "--serialized_data_dir" "${SERIALIZED_DATA_DIR}"
  "--output_dir" "${OUTPUT_DIR}"
  "--num_samples" "${NUM_SAMPLES}"
  "--temperature" "${TEMPERATURE}"
  "--max_completion_tokens" "${MAX_COMPLETION_TOKENS}"
  "--max_workers" "${MAX_WORKERS}"
)

# shellcheck disable=SC2206
TASK_ARRAY=(${TASKS})
PYTHON_ARGS+=("--tasks" "${TASK_ARRAY[@]}")

if [[ -n "${PATIENT_IDS}" ]]; then
  # shellcheck disable=SC2206
  PATIENT_ID_ARRAY=(${PATIENT_IDS})
  PYTHON_ARGS+=("--patient_ids" "${PATIENT_ID_ARRAY[@]}")
fi

echo "Running Azure GPT-5 evaluation pipeline..."
python "${SCRIPT_DIR}/azure_evaluate_multi_task.py" "${PYTHON_ARGS[@]}"
EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
  echo ""
  echo "=========================================="
  echo "Evaluation completed successfully."
  echo "Results saved to: ${OUTPUT_DIR}"
  echo "=========================================="
else
  echo "ERROR: Evaluation failed with exit code ${EXIT_CODE}"
fi

exit $EXIT_CODE

