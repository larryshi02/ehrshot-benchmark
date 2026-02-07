# ehrshot-v2: Clean Clinical Prediction Pipeline

A modular pipeline for clinical outcome prediction using EHR data.
Supports multiple representation strategies, fine-tuning approaches,
and embedding-based evaluation.

## Pipeline Overview

```
01_serialize  ->  02_create_sft  ->  05_train  ->  06_eval
                       |                              |
               03_rubric (GPT-5-mini)         generate_embeddings
                       |                      + eval_embeddings
               04_cot (GPT-5-mini)
```

### Execution Order

```bash
# Step 1: Serialize EHRs (plaintext + manualrubric)
bash 01_serialize/run.sh

# Step 2: Create SFT datasets
bash 02_create_sft/run.sh

# Step 3: Build rubrics + rubricified representations (requires GPT-5-mini)
bash 03_rubric/run.sh

# Step 4: Generate CoT reasoning traces (requires GPT-5-mini)
bash 04_cot/run.sh

# Step 5: Fine-tune Qwen3-8B (requires GPU)
bash 05_train/run.sh

# Step 6: Evaluate everything
bash 06_eval/run.sh
```

## Directory Structure

```
ehrshot-v2/
├── README.md
├── config/
│   ├── tasks.py                      # 15 task definitions + model names
│   ├── azure.py                      # Azure OpenAI config loader
│   └── azure_config.json             # Credentials (not committed)
│
├── 01_serialize/
│   ├── serialize.py                  # EHR -> Markdown text
│   ├── ehr_serializer.py             # Core serializer (no "Past Medical Visits")
│   └── run.sh
│
├── 02_create_sft/
│   ├── create_sft.py                 # Serialized data -> SFT conversations
│   └── run.sh
│
├── 03_rubric/
│   ├── build_cohort.py               # K-means + medoid selection (40 patients)
│   ├── create_rubric.py              # GPT-5-mini rubric generation
│   ├── apply_rubric.py               # Apply rubric to all patients
│   ├── create_llmrubric_sft.py       # Rubricified -> SFT format
│   └── run.sh
│
├── 04_cot/
│   ├── generate_supervised_cot.py    # CoT with ground truth (train+val)
│   ├── generate_unsupervised_cot.py  # CoT without ground truth (train+val+test)
│   └── run.sh
│
├── 05_train/
│   ├── finetune_direct.py            # Qwen3-8B LoRA -> Yes/No
│   ├── finetune_reasoning.py         # Qwen3-8B LoRA -> <think>+Yes/No
│   └── run.sh
│
├── 06_eval/
│   ├── eval_direct.py                # vLLM logprobs evaluation
│   ├── eval_reasoning.py             # Sampling-based evaluation
│   ├── generate_embeddings.py        # Qwen3-Embedding-8B embeddings
│   ├── eval_embeddings.py            # LogReg with val-based C selection
│   ├── compute_metrics.py            # AUROC/AUPRC + bootstrap CIs
│   └── run.sh
│
└── data/                             # All generated outputs (gitignored)
```

## Tasks (15)

| Category | Task | Query |
|----------|------|-------|
| Operational | `guo_icu` | Will the patient be transferred to the ICU? |
| Operational | `guo_los` | Will the patient stay > 7 days? |
| Operational | `guo_readmission` | Will the patient be readmitted within 30 days? |
| Lab | `lab_thrombocytopenia` | Will the thrombocytopenia lab come back as abnormal? |
| Lab | `lab_hyperkalemia` | Will the hyperkalemia lab come back as abnormal? |
| Lab | `lab_hypoglycemia` | Will the hypoglycemia lab come back as abnormal? |
| Lab | `lab_hyponatremia` | Will the hyponatremia lab come back as abnormal? |
| Lab | `lab_anemia` | Will the anemia lab come back as abnormal? |
| Diagnosis | `new_hypertension` | Will the patient develop hypertension in the next year? |
| Diagnosis | `new_hyperlipidemia` | Will the patient develop hyperlipidemia in the next year? |
| Diagnosis | `new_pancan` | Will the patient develop pancreatic cancer in the next year? |
| Diagnosis | `new_celiac` | Will the patient develop celiac disease in the next year? |
| Diagnosis | `new_lupus` | Will the patient develop lupus in the next year? |
| Diagnosis | `new_acutemi` | Will the patient develop an acute MI in the next year? |
| Imaging | `chexpert` | Does the patient have abnormal chest X-ray findings? |

## Experiment Matrix

All experiments run with **n=full** and **n=40** (rubric cohort patients):

| Representation | Direct FT (Yes/No) | Embedding (LogReg) | Reasoning FT |
|----------------|---------------------|---------------------|--------------|
| plaintext | Yes | Yes | - |
| manualrubric | Yes | Yes | - |
| llmrubric | Yes | Yes | - |
| cot_supervised | - | - | Yes |
| cot_unsupervised | Yes | Yes | - |

## Key Design Decisions

1. **Yes/No everywhere** -- no Positive/Negative, consistent across all tasks.
2. **Sandwich-style prompts** -- task query appears before and after the EHR.
3. **Validation-based model selection** -- both embedding (LogReg C) and fine-tuning (early stopping) use the val set. No cross-validation.
4. **No "Past Medical Visits" section** -- removed from serializer to save tokens for detailed visits.
5. **8192-token cap** -- clipped with `Qwen/Qwen3-Embedding-8B` tokenizer.
6. **n=40 uses rubric cohort** -- same 40 patient IDs per task across all n=40 experiments for fair comparison.
7. **GPT-5-mini parameters** -- `max_completion_tokens=16384`, `temperature=1` for all generation.
