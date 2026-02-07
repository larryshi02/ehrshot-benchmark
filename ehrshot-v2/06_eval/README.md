# 06_eval/

Evaluate all trained models and embedding-based approaches.

## Scripts

| Script | Method | Input | Output |
|--------|--------|-------|--------|
| `eval_direct.py` | vLLM logprobs P(Yes) = sigmoid(logp_yes - logp_no) | SFT test JSON + LoRA adapter | predictions.csv |
| `eval_reasoning.py` | Sample N=10 responses, parse "Final Answer" | SFT test JSON + LoRA adapter | predictions.csv |
| `generate_embeddings.py` | Qwen3-Embedding-8B (last-token pool, L2 norm) | SFT dataset | .npz files |
| `eval_embeddings.py` | L2-regularized LogReg, val-based C selection | .npz files | predictions.csv + metrics.json |
| `compute_metrics.py` | AUROC / AUPRC with bootstrap 95% CIs | predictions.csv | per_task + summary JSON |

## Embedding Evaluation Pipeline

```
generate_embeddings.py -> eval_embeddings.py -> compute_metrics.py
```

C values searched: `[1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]`
Best C selected on validation set (not cross-validation).

## n=40 Experiments

`eval_embeddings.py` supports `--n_train 40 --cohort_file ...` to filter training embeddings.

## Outputs

```
data/results/{approach}_{repr}_{full,n40}/{task}/
  predictions.csv
  metrics.json (embeddings only)

data/results/metrics/{approach}/
  per_task_metrics.json
  summary.json
```

## predictions.csv Schema

```
patient_id, label_time, ground_truth, probability_score, target_task
```
