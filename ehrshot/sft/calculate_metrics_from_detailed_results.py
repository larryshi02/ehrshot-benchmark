#!/usr/bin/env python3
"""
Calculate AUROC, precision, recall, and F1 score from detailed results JSON file.
This script processes files like base_model_multi_task_results/acute_mi_detailed_results.json
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_curve
)
from pathlib import Path


def calculate_auroc_ci_bootstrap(ground_truths, scores, n_bootstrap=10000, confidence_level=0.95, random_seed=42):
    """
    Calculate AUROC with confidence intervals using bootstrap resampling.
    
    Args:
        ground_truths: Ground truth labels (binary)
        scores: Prediction scores/probabilities
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level (default 0.95 for 95% CI)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with AUROC, lower bound, upper bound, and bootstrap samples
    """
    np.random.seed(random_seed)
    
    n_samples = len(ground_truths)
    bootstrap_aurocs = []
    
    print(f"Computing {n_bootstrap} bootstrap iterations for {confidence_level*100:.0f}% CI...")
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_y_true = ground_truths[indices]
        bootstrap_y_score = scores[indices]
        
        # Check if both classes are present in bootstrap sample
        unique_classes = np.unique(bootstrap_y_true)
        if len(unique_classes) < 2:
            # Skip if only one class is present
            continue
        
        # Calculate AUROC for this bootstrap sample
        try:
            bootstrap_auroc = roc_auc_score(bootstrap_y_true, bootstrap_y_score)
            bootstrap_aurocs.append(bootstrap_auroc)
        except ValueError:
            # Skip if AUROC cannot be calculated
            continue
        
        # Print progress every 1000 iterations
        if (i + 1) % 1000 == 0:
            print(f"  Completed {i + 1}/{n_bootstrap} bootstrap iterations...")
    
    if len(bootstrap_aurocs) == 0:
        print("Warning: No valid bootstrap samples. Returning None for CI.")
        # Calculate actual AUROC on full data even if bootstrap failed
        try:
            actual_auroc = roc_auc_score(ground_truths, scores)
        except ValueError:
            actual_auroc = 0.5  # Default to 0.5 if only one class
        return {
            'auroc': float(actual_auroc),
            'ci_lower': None,
            'ci_upper': None,
            'n_valid_bootstrap': 0,
            'bootstrap_mean': None,
            'bootstrap_std': None
        }
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_aurocs, lower_percentile)
    ci_upper = np.percentile(bootstrap_aurocs, upper_percentile)
    
    # Calculate actual AUROC on full data
    try:
        actual_auroc = roc_auc_score(ground_truths, scores)
    except ValueError:
        actual_auroc = 0.5  # Default to 0.5 if only one class
    
    return {
        'auroc': float(actual_auroc),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_valid_bootstrap': len(bootstrap_aurocs),
        'bootstrap_mean': float(np.mean(bootstrap_aurocs)),
        'bootstrap_std': float(np.std(bootstrap_aurocs))
    }


def plot_score_distribution(ground_truths, scores, task_name, output_file=None, optimal_threshold=None):
    """
    Plot the distribution of continuous scores, stratified by ground truth labels.
    
    Args:
        ground_truths: Ground truth labels (binary)
        scores: Prediction scores/probabilities
        task_name: Name of the task for the plot title
        output_file: Path to save the plot. If None, auto-generates based on task_name
        optimal_threshold: Optional threshold to draw as a vertical line
    """
    # Separate scores by ground truth
    positive_scores = scores[ground_truths == 1]
    negative_scores = scores[ground_truths == 0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histograms with transparency
    bins = np.linspace(0, 1, 51)  # 50 bins from 0 to 1
    
    ax.hist(negative_scores, bins=bins, alpha=0.6, label=f'Negative (n={len(negative_scores)})', 
            color='blue', edgecolor='black', linewidth=0.5)
    ax.hist(positive_scores, bins=bins, alpha=0.6, label=f'Positive (n={len(positive_scores)})', 
            color='red', edgecolor='black', linewidth=0.5)
    
    # Add optimal threshold line if provided
    if optimal_threshold is not None:
        ax.axvline(x=optimal_threshold, color='green', linestyle='--', linewidth=2, 
                  label=f'Optimal Threshold ({optimal_threshold:.3f})')
    
    # Labels and formatting
    ax.set_xlabel('Prediction Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Score Distribution: {task_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        output_file = f"{task_name}_score_distribution.png"
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Score distribution plot saved to: {output_file}")
    
    return output_file


def calculate_metrics_from_detailed_results(input_file: str, output_file: str = None, n_bootstrap=10000, plot_distribution=True):
    """
    Calculate AUROC, precision, recall, and F1 score from detailed results JSON file.
    
    Args:
        input_file: Path to the detailed results JSON file (e.g., acute_mi_detailed_results.json)
        output_file: Optional path to save metrics JSON file. If None, saves next to input file.
        n_bootstrap: Number of bootstrap iterations for confidence interval calculation
        plot_distribution: Whether to plot and save the score distribution
    
    Returns:
        Dictionary containing the calculated metrics
    """
    # Load the detailed results
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples from {input_file}")
    
    # Filter out entries with 0 valid predictions
    data_filtered = []
    excluded_count = 0
    for entry in data:
        num_valid = entry.get('num_valid_predictions', 0)
        if num_valid > 0:
            data_filtered.append(entry)
        else:
            excluded_count += 1
    
    if excluded_count > 0:
        print(f"Excluded {excluded_count} data points with 0 valid predictions")
    
    print(f"Using {len(data_filtered)} examples for calculation")
    
    # Extract ground truth and scores
    ground_truths = []
    scores = []
    
    for entry in data_filtered:
        # Convert ground_truth to int (handle both bool and int)
        if isinstance(entry['ground_truth'], bool):
            gt = 1 if entry['ground_truth'] else 0
        else:
            gt = int(entry['ground_truth'])
        ground_truths.append(gt)
        
        # Extract the aggregated score (probability)
        score = float(entry['score'])
        scores.append(score)
    
    # Convert to numpy arrays
    ground_truths = np.array(ground_truths)
    scores = np.array(scores)
    
    print(f"Ground truth distribution: {np.bincount(ground_truths)}")
    print(f"Average score: {np.mean(scores):.4f}")
    print(f"Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
    
    # Calculate AUROC with bootstrap confidence intervals
    auroc_results = calculate_auroc_ci_bootstrap(ground_truths, scores, n_bootstrap=n_bootstrap)
    auroc = auroc_results['auroc']
    
    print(f"\nAUROC: {auroc:.4f}")
    if auroc_results['ci_lower'] is not None and auroc_results['ci_upper'] is not None:
        print(f"AUROC 95% CI: [{auroc_results['ci_lower']:.4f}, {auroc_results['ci_upper']:.4f}]")
    
    # Calculate precision-recall curve and ROC curve
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(ground_truths, scores)
    fpr, tpr, roc_thresholds = roc_curve(ground_truths, scores)
    
    # Find optimal threshold using Youden's J statistic (maximize TPR - FPR)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = roc_thresholds[optimal_idx]
    
    # Binary predictions at optimal threshold
    binary_predictions = (scores >= optimal_threshold).astype(int)
    
    # Count predicted positives and negatives
    n_predicted_positive = int(np.sum(binary_predictions == 1))
    n_predicted_negative = int(np.sum(binary_predictions == 0))
    
    # Calculate metrics at optimal threshold
    precision = precision_score(ground_truths, binary_predictions, zero_division=0)
    recall = recall_score(ground_truths, binary_predictions, zero_division=0)
    f1 = f1_score(ground_truths, binary_predictions, zero_division=0)
    
    # Print results
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Predicted Positive: {n_predicted_positive}")
    print(f"Predicted Negative: {n_predicted_negative}")
    
    # Get task name and input directory for consistent output file naming
    input_path = Path(input_file)
    # Extract task name from filename (remove _detailed_results suffix if present)
    task_name = input_path.stem
    if task_name.endswith('_detailed_results'):
        task_name = task_name[:-len('_detailed_results')]
    elif task_name.endswith('_detailed'):
        task_name = task_name[:-len('_detailed')]
    input_dir = input_path.parent
    
    # Plot score distribution if requested (always in same directory as input)
    plot_file = None
    if plot_distribution:
        plot_output_file = input_dir / f"{task_name}_score_distribution.png"
        plot_file = plot_score_distribution(
            ground_truths, 
            scores, 
            task_name, 
            output_file=str(plot_output_file),
            optimal_threshold=optimal_threshold
        )
    
    # Prepare metrics dictionary
    metrics = {
        'task_name': task_name,
        'n_examples': len(data_filtered),
        'n_excluded': excluded_count,
        'auroc': float(auroc),
        'auroc_ci_95': {
            'lower': float(auroc_results['ci_lower']) if auroc_results['ci_lower'] is not None else None,
            'upper': float(auroc_results['ci_upper']) if auroc_results['ci_upper'] is not None else None
        },
        'bootstrap_stats': {
            'n_valid_bootstrap': int(auroc_results['n_valid_bootstrap']),
            'bootstrap_mean': float(auroc_results['bootstrap_mean']) if auroc_results['bootstrap_mean'] is not None else None,
            'bootstrap_std': float(auroc_results['bootstrap_std']) if auroc_results['bootstrap_std'] is not None else None
        },
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'optimal_threshold': float(optimal_threshold),
        'predicted_distribution': {
            'positive': n_predicted_positive,
            'negative': n_predicted_negative
        },
        'ground_truth_distribution': {
            'positive': int(np.sum(ground_truths == 1)),
            'negative': int(np.sum(ground_truths == 0))
        },
        'score_distribution_plot': plot_file if plot_distribution else None
    }
    
    # Save metrics to JSON file (always in same directory as input file)
    # Name it according to the task name: {task_name}_metrics.json
    if output_file is None:
        # Auto-generate filename based on task name
        output_file = str(input_dir / f"{task_name}_metrics.json")
    else:
        # If output_file is specified, still name it according to task but in same directory
        # This ensures consistency - metrics files are always named by task
        output_file = str(input_dir / f"{task_name}_metrics.json")
    
    # Ensure output_file is a string
    output_file = str(output_file)
    
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to: {output_file}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Calculate AUROC, precision, recall, and F1 score from detailed results JSON file"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to detailed results JSON file (e.g., base_model_multi_task_results/acute_mi_detailed_results.json)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path to save metrics JSON file. If not specified, saves next to input file."
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap iterations for confidence interval calculation (default: 10000)"
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable plotting of score distribution"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}")
        return
    
    # Calculate metrics
    plot_distribution = not args.no_plot
    metrics = calculate_metrics_from_detailed_results(args.input_file, args.output_file, args.n_bootstrap, plot_distribution)
    
    # Print summary
    print("\n" + "=" * 70)
    print("METRICS SUMMARY")
    print("=" * 70)
    print(f"Task: {metrics['task_name']}")
    print(f"Number of examples: {metrics['n_examples']}")
    if metrics['n_excluded'] > 0:
        print(f"Number of excluded examples (0 valid predictions): {metrics['n_excluded']}")
    print(f"AUROC: {metrics['auroc']:.4f}")
    if metrics['auroc_ci_95']['lower'] is not None and metrics['auroc_ci_95']['upper'] is not None:
        print(f"AUROC 95% CI: [{metrics['auroc_ci_95']['lower']:.4f}, {metrics['auroc_ci_95']['upper']:.4f}]")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    print(f"Predicted - Positive: {metrics['predicted_distribution']['positive']}")
    print(f"Predicted - Negative: {metrics['predicted_distribution']['negative']}")
    print(f"Ground truth - Positive: {metrics['ground_truth_distribution']['positive']}")
    print(f"Ground truth - Negative: {metrics['ground_truth_distribution']['negative']}")
    if metrics['score_distribution_plot']:
        print(f"Score distribution plot: {metrics['score_distribution_plot']}")
    print("=" * 70)


if __name__ == "__main__":
    main()

