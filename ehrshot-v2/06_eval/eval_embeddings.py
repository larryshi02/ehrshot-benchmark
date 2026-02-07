#!/usr/bin/env python3
"""
Evaluate embeddings via logistic regression (train -> val -> test).

Unlike the original ehrshot code which used 5-fold cross-validation on
training data, this script uses the validation set for hyperparameter (C)
selection -- providing an apples-to-apples comparison with the fine-tuning
approaches that also use the val set for early stopping.

Pipeline:
  1. Load train / val / test embeddings (.npz from generate_embeddings.py).
  2. For each C in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]:
     - Fit LogisticRegression on train, evaluate AUROC on val.
  3. Refit with best C on train, evaluate on test.
  4. Compute bootstrap 95% CIs on test metrics.

Inputs:
  --embeddings_dir : Directory with {task}/{split}.npz files.
  --output_dir     : Where to write results.
  --tasks          : Space-separated task list (default: all 15).
  --n_train        : If set, filter training embeddings to cohort patients.
  --cohort_file    : Path to patient_ids.json (required when --n_train).

Outputs:
  {output_dir}/{task}/metrics.json
  {output_dir}/{task}/predictions.csv

Connects to:
  - Upstream  : generate_embeddings.py
  - Downstream: compute_metrics.py (or standalone)
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.tasks import ALL_TASK_NAMES, SEED

C_VALUES = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
BOOTSTRAP_N = 1000


def _load_npz(path: str):
    d = np.load(path)
    return d["embeddings"], d["patient_ids"], d["labels"]


def _bootstrap_ci(y_true, y_score, metric_fn, n=BOOTSTRAP_N, seed=SEED):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(metric_fn(yt, ys))
    if not scores:
        return 0.0, 0.0, 0.0
    arr = np.array(scores)
    return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def evaluate_task(
    emb_dir: str, task: str, output_dir: str,
    n_train: int | None = None, cohort_ids: set | None = None,
):
    task_dir = Path(emb_dir) / task
    train_path = task_dir / "train.npz"
    val_path = task_dir / "val.npz"
    test_path = task_dir / "test.npz"

    if not train_path.exists() or not test_path.exists():
        logger.warning(f"  {task}: missing train or test embeddings, skipping")
        return

    X_train, pids_train, y_train = _load_npz(str(train_path))

    # Optional: filter to cohort
    if n_train is not None and cohort_ids is not None:
        mask = np.isin(pids_train, list(cohort_ids))
        X_train, pids_train, y_train = X_train[mask], pids_train[mask], y_train[mask]
        logger.info(f"  filtered train to {len(y_train)} (n_train={n_train})")

    X_test, pids_test, y_test = _load_npz(str(test_path))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Hyperparameter selection on val set
    best_c = C_VALUES[0]
    best_val_auc = -1.0
    if val_path.exists():
        X_val, _, y_val = _load_npz(str(val_path))
        X_val_s = scaler.transform(X_val)
        for c in C_VALUES:
            clf = LogisticRegression(C=c, penalty="l2", solver="lbfgs",
                                     max_iter=1000, class_weight="balanced",
                                     random_state=SEED)
            clf.fit(X_train_s, y_train)
            if len(np.unique(y_val)) < 2:
                continue
            val_auc = roc_auc_score(y_val, clf.predict_proba(X_val_s)[:, 1])
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_c = c
        logger.info(f"  best C={best_c} (val AUROC={best_val_auc:.4f})")
    else:
        logger.info(f"  no val split, using C={best_c}")

    # Final model
    clf = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs",
                             max_iter=1000, class_weight="balanced",
                             random_state=SEED)
    clf.fit(X_train_s, y_train)
    y_score = clf.predict_proba(X_test_s)[:, 1]

    auroc, auroc_lo, auroc_hi = _bootstrap_ci(y_test, y_score, roc_auc_score)
    auprc, auprc_lo, auprc_hi = _bootstrap_ci(y_test, y_score, average_precision_score)

    metrics = {
        "task": task,
        "best_c": best_c,
        "val_auroc": best_val_auc,
        "test_auroc": auroc,
        "test_auroc_ci": [auroc_lo, auroc_hi],
        "test_auprc": auprc,
        "test_auprc_ci": [auprc_lo, auprc_hi],
        "n_train": len(y_train),
        "n_test": len(y_test),
    }

    out = Path(output_dir) / task
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out / "predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "patient_id", "label_time", "ground_truth",
            "probability_score", "target_task"])
        w.writeheader()
        for pid, gt, ps in zip(pids_test, y_test, y_score):
            w.writerow({"patient_id": int(pid), "label_time": "",
                        "ground_truth": int(gt),
                        "probability_score": float(ps),
                        "target_task": task})

    logger.info(f"  {task}: AUROC={auroc:.4f} [{auroc_lo:.4f}, {auroc_hi:.4f}]  "
                f"AUPRC={auprc:.4f} [{auprc_lo:.4f}, {auprc_hi:.4f}]")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tasks", nargs="+", default=ALL_TASK_NAMES)
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--cohort_file", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cohort_ids = None
    if args.n_train is not None:
        if args.cohort_file is None:
            raise ValueError("--cohort_file required when --n_train is set")
        with open(args.cohort_file) as f:
            cohort_ids = set(json.load(f))

    for task in args.tasks:
        logger.info(f"\nEvaluating embeddings for: {task}")
        evaluate_task(args.embeddings_dir, task, args.output_dir,
                      args.n_train, cohort_ids)

    logger.success("All embedding evaluations complete.")


if __name__ == "__main__":
    main()
