# 04_cot/

Generate Chain-of-Thought reasoning traces using GPT-5-mini.

## Scripts

| Script | Purpose | Splits | Ground Truth? |
|--------|---------|--------|---------------|
| `generate_supervised_cot.py` | Backward reasoning WITH ground truth label | train + val | Yes |
| `generate_unsupervised_cot.py` | Risk profile inference WITHOUT ground truth | train + val + test | No |

## GPT-5-mini Parameters

- `max_completion_tokens=16384`
- `temperature=1`

## Reasoning Trace Structure

```
<think>
1. ### PATIENT SNAPSHOT
2. ### MAIN RISK FACTORS
3. ### PROTECTIVE FACTORS
4. ### WHAT'S UNKNOWN / COULD SWING THE RISK
5. ### WEIGHING AND AGGREGATING THE EVIDENCE
6. ### CONCLUSION AND AN OVERALL RISK IMPRESSION

Final Answer: Yes/No
</think>
```

## Task Configuration

Both scripts accept `--tasks` to select which tasks to generate for (default: all 15).

## Inputs

- Plaintext SFT datasets (`data/sft/plaintext`)

## Outputs

```
data/sft/cot_supervised/{split}/{task}.json   # reasoning traces with answer
data/sft/cot_unsupervised/{split}/{task}.json  # reasoning traces without label
```

## Usage Downstream

- **Supervised CoT** -> `05_train/finetune_reasoning.py` (learns to generate reasoning + answer)
- **Unsupervised CoT** -> `05_train/finetune_direct.py` (trace as input, Yes/No output)
- **Unsupervised CoT** -> `06_eval/generate_embeddings.py` (embed traces for LogReg)
