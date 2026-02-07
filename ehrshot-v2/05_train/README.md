# 05_train/

Fine-tune Qwen3-8B for clinical prediction tasks.

## Scripts

| Script | Input | Output Format | Use Case |
|--------|-------|---------------|----------|
| `finetune_direct.py` | Any SFT dataset (plaintext, manualrubric, llmrubric, cot_unsupervised) | Yes / No | Direct classification |
| `finetune_reasoning.py` | Supervised CoT SFT dataset | `<think>...</think> Final Answer: Yes/No` | Reasoning |

## Key Design

- **LoRA**: r=16, alpha=32, applied to all attention + MLP projections
- **Optimizer**: AdamW fused, cosine LR schedule
- **Early stopping**: patience=20 (direct) / patience=10 (reasoning), based on val loss
- **Prompt masking**: loss computed only on assistant response tokens
- **Qwen3 thinking mode**: disabled (`enable_thinking=False`)

## n=40 Experiments

Both scripts support `--n_train 40 --cohort_file path/to/patient_ids.json` to filter training data to the 40 rubric cohort patients.

## Inputs

```
data/sft/{repr}/{split}/{task}.json
data/rubric/{task}/patient_ids.json   # for n=40 only
```

## Outputs

```
data/models/{direct,reasoning}_{repr}_{full,n40}/{task}/
  adapter_config.json
  adapter_model.safetensors
  tokenizer files
```

## Next Step

`06_eval/` evaluates the trained models.
