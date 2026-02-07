# 03_rubric/

Build LLM-generated rubrics for structured EHR evaluation.

## Pipeline

```
build_cohort.py -> create_rubric.py -> apply_rubric.py -> create_llmrubric_sft.py
```

1. **build_cohort.py**: Embed training patients with Qwen3-Embedding-8B, run label-stratified k-means (k=20 per class), select medoids -> 40 patients per task.
2. **create_rubric.py**: Prompt GPT-5-mini with the 40 cohort examples to generate a step-by-step rubric template.
3. **apply_rubric.py**: Apply the rubric to all patients' plaintext EHRs via GPT-5-mini (parallel, resumable).
4. **create_llmrubric_sft.py**: Wrap rubricified text into SFT conversation format.

## GPT-5-mini Parameters

- `max_completion_tokens=16384`
- `temperature=1`

## Inputs

- Plaintext serialized data (`data/serialized/plaintext`)

## Outputs

```
data/rubric/{task}/cohort.json          # 40 cohort records
data/rubric/{task}/patient_ids.json     # 40 patient IDs (reused for n=40 experiments)
data/rubric/{task}/rubric.json          # rubric instructions
data/rubric/rubricified/{task}/{split}.json  # rubricified patient records
data/sft/llmrubric/{split}/{task}.json  # SFT datasets
```

## Next Step

- `05_train/` trains on `data/sft/llmrubric/`.
- `patient_ids.json` is reused by `--n_train 40` experiments.
