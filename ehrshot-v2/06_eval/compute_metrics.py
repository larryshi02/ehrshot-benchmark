#!/usr/bin/env python3
"""
Compute AUROC and AUPRC with bootstrap 95% confidence intervals.

Reads predictions.csv files produced by eval_direct.py, eval_reasoning.py,
or eval_embeddings.py, and computes per-task and aggregate metrics.

Inputs:
  --predictions : One or more paths to predictions.csv files.
  --output_dir  : Where to write metrics.

Outputs:
  {output_dir}/per_task_metrics.json   -- per-task AUROC / AUPRC with CIs
  {output_dir}/summary.json            -- mean across tasks

predictions.csv format:
  patient_id, label_time, ground_truth, probability_score, target_task

Connects to:
  - Upstream: eval_direct.py, eval_reasoning.py, eval_embeddings.py
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import SEED

BOOTSTRAP_N = 1000


def _bootstrap(y_true, y_score, fn, n=BOOTSTRAP_N, seed=SEED):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        vals.append(fn(yt, ys))
    if not vals:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    arr = np.array(vals)
    return {
        "mean": float(np.mean(arr)),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", nargs="+", required=True,
                   help="Path(s) to predictions.csv file(s)")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    # Load and concatenate all prediction files
    dfs = []
    for path in args.predictions:
        dfs.append(pd.read_csv(path))
    df = pd.concat(dfs, ignore_index=True)

    tasks = sorted(df["target_task"].unique())
    per_task = {}
    aurocs, auprcs = [], []

    for task in tasks:
        sub = df[df["target_task"] == task]
        y_true = sub["ground_truth"].values
        y_score = sub["probability_score"].values

        if len(np.unique(y_true)) < 2:
            logger.warning(f"  {task}: only one class, skipping")
            continue

        auroc = _bootstrap(y_true, y_score, roc_auc_score)
        auprc = _bootstrap(y_true, y_score, average_precision_score)

        per_task[task] = {
            "n": len(sub),
            "auroc": auroc,
            "auprc": auprc,
        }
        aurocs.append(auroc["mean"])
        auprcs.append(auprc["mean"])
        logger.info(f"  {task}: AUROC={auroc['mean']:.4f} "
                    f"[{auroc['ci_lo']:.4f}, {auroc['ci_hi']:.4f}]  "
                    f"AUPRC={auprc['mean']:.4f} "
                    f"[{auprc['ci_lo']:.4f}, {auprc['ci_hi']:.4f}]")

    summary = {
        "n_tasks": len(per_task),
        "mean_auroc": float(np.mean(aurocs)) if aurocs else 0.0,
        "mean_auprc": float(np.mean(auprcs)) if auprcs else 0.0,
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_task_metrics.json", "w") as f:
        json.dump(per_task, f, indent=2)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nSummary: mean AUROC={summary['mean_auroc']:.4f}  "
                f"mean AUPRC={summary['mean_auprc']:.4f}")
    logger.success(f"Metrics saved to {out}")


if __name__ == "__main__":
    main()
