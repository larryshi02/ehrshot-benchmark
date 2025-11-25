import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

TASKS = ['acute_mi', 'pancreatic_cancer', 'hypertension', 'hyperlipidemia']

def calculate_auprc(y_true, y_scores):
    """Calculates Area Under Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)

def bootstrap_metric(y_true, y_scores, metric_fn, n_bootstraps=1000):
    """
    Returns [mean, low_ci, high_ci] for a given metric using bootstrapping.
    """
    rng = np.random.RandomState(42)
    bootstrapped_scores = []
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_scores), len(y_scores))
        # Need at least two classes to calculate ROC/PRC
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        try:
            score = metric_fn(y_true[indices], y_scores[indices])
            bootstrapped_scores.append(score)
        except ValueError:
            continue
            
    if not bootstrapped_scores:
        return [0.0, 0.0, 0.0]

    sorted_scores = np.sort(bootstrapped_scores)
    mean_score = np.mean(sorted_scores)
    ci_lower = np.percentile(sorted_scores, 2.5)
    ci_upper = np.percentile(sorted_scores, 97.5)
    
    return [mean_score, ci_lower, ci_upper]

def get_optimal_f1_stats(y_true, y_scores):
    """Calculates F1, Precision, Recall at optimal threshold."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # Avoid division by zero
    denominator = recall + precision
    f1_scores = np.divide(
        2 * recall * precision, 
        denominator, 
        out=np.zeros_like(denominator), 
        where=denominator != 0
    )
    
    best_idx = np.argmax(f1_scores) if len(f1_scores) > 0 else 0
    
    # Handle edge case where thresholds might be shorter than p/r arrays
    opt_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    
    return {
        "optimal_threshold": opt_thresh,
        "f1": float(f1_scores[best_idx]) if len(f1_scores) > 0 else 0.0,
        "precision": float(precision[best_idx]) if len(precision) > 0 else 0.0,
        "recall": float(recall[best_idx]) if len(recall) > 0 else 0.0
    }

def process_single_task(folder_path, task_name):
    """
    Reads predictions.csv, calculates metrics, saves summary.json, 
    and returns the stats dict.
    """
    csv_path = os.path.join(folder_path, "predictions.csv")
    stats_path = os.path.join(folder_path, "run_stats.json")
    
    if not os.path.exists(csv_path) or not os.path.exists(stats_path):
        return None

    df = pd.read_csv(csv_path)
    with open(stats_path, 'r', encoding='utf-8') as f:
        run_stats = json.load(f)

    # Filter: Only use examples where we got at least one valid syntax response
    valid_df = df[df['has_valid_samples'] == True]
    
    if valid_df.empty:
        return None
        
    y_true = valid_df['ground_truth'].values
    y_scores = valid_df['probability_score'].values

    # Calculate Metrics
    auroc_stats = bootstrap_metric(y_true, y_scores, roc_auc_score)
    auprc_stats = bootstrap_metric(y_true, y_scores, calculate_auprc)
    opt_stats = get_optimal_f1_stats(y_true, y_scores)

    summary = {
        "task_name": task_name,
        "auroc": auroc_stats, # [Mean, Low, High]
        "auprc": auprc_stats, # [Mean, Low, High]
        **opt_stats,
        **run_stats
    }
    
    with open(os.path.join(folder_path, "summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", type=str, required=True, help="Root folder of experiment")
    args = parser.parse_args()

    # Create DataFrames for the 4x4 comparison tables
    auroc_matrix = pd.DataFrame(index=TASKS, columns=TASKS)
    auprc_matrix = pd.DataFrame(index=TASKS, columns=TASKS)

    print(f"Processing results in: {args.eval_root}")

    for source_task in TASKS:
        source_dir = os.path.join(args.eval_root, source_task)
        
        if not os.path.exists(source_dir):
            print(f"Skipping missing source model folder: {source_task}")
            continue

        for target_task in TASKS:
            target_dir = os.path.join(source_dir, target_task)
            
            summary = process_single_task(target_dir, target_task)
            
            if summary:
                roc = summary['auroc']
                prc = summary['auprc']
                
                # Format: "0.850 (0.840-0.860)"
                auroc_str = f"{roc[0]:.3f} ({roc[1]:.3f}-{roc[2]:.3f})"
                auprc_str = f"{prc[0]:.3f} ({prc[1]:.3f}-{prc[2]:.3f})"
                
                auroc_matrix.loc[source_task, target_task] = auroc_str
                auprc_matrix.loc[source_task, target_task] = auprc_str
            else:
                auroc_matrix.loc[source_task, target_task] = "N/A"
                auprc_matrix.loc[source_task, target_task] = "N/A"

    print("\n=== AUROC Table (Rows: Model, Cols: Task) ===")
    print(auroc_matrix)
    auroc_matrix.to_csv(os.path.join(args.eval_root, "final_auroc_table.csv"))

    print("\n=== AUPRC Table (Rows: Model, Cols: Task) ===")
    print(auprc_matrix)
    auprc_matrix.to_csv(os.path.join(args.eval_root, "final_auprc_table.csv"))

if __name__ == "__main__":
    main()