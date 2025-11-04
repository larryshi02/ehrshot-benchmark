#!/usr/bin/env python3
"""
Calculate AUROC from existing multi-sample outputs JSON files
"""

import json
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
import argparse

def calculate_auroc_from_outputs(output_file, model_type="fine_tuned"):
    """Calculate AUROC from the multi-sample outputs JSON file"""
    
    # Load the outputs
    with open(output_file, 'r') as f:
        outputs = json.load(f)
    
    print(f"Loaded {len(outputs)} examples from {output_file}")
    print(f"Model type: {model_type}")
    
    # Extract ground truth and aggregated predictions
    ground_truths = []
    aggregated_predictions = []
    all_confidences = []
    all_consistencies = []
    
    for output in outputs:
        # Convert ground_truth to int if needed
        if isinstance(output['ground_truth'], bool):
            gt = 1 if output['ground_truth'] else 0
        else:
            gt = int(output['ground_truth'])
        ground_truths.append(gt)
        
        # Use the aggregated prediction (probability between 0 and 1)
        aggregated_predictions.append(output['aggregated_prediction'])
        
        # Extract confidence and consistency if available
        if 'prediction_confidence' in output:
            all_confidences.append(output['prediction_confidence'])
        if 'prediction_consistency' in output:
            all_consistencies.append(output['prediction_consistency'])
    
    ground_truths = np.array(ground_truths)
    aggregated_predictions = np.array(aggregated_predictions)
    
    print(f"Ground truth distribution: {np.bincount(ground_truths)}")
    print(f"Average aggregated prediction: {np.mean(aggregated_predictions):.4f}")
    
    # Calculate AUROC
    try:
        auroc = roc_auc_score(ground_truths, aggregated_predictions)
        print(f"AUROC: {auroc:.4f}")
    except ValueError as e:
        print(f"Error calculating AUROC: {e}")
        return None
    
    # Calculate additional metrics
    precision, recall, thresholds = precision_recall_curve(ground_truths, aggregated_predictions)
    fpr, tpr, roc_thresholds = roc_curve(ground_truths, aggregated_predictions)
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = roc_thresholds[optimal_idx]
    
    # Binary predictions at optimal threshold
    binary_predictions = (aggregated_predictions >= optimal_threshold).astype(int)
    
    # Additional metrics
    accuracy = np.mean(binary_predictions == ground_truths)
    precision_at_optimal = precision[optimal_idx] if optimal_idx < len(precision) else 0.0
    recall_at_optimal = recall[optimal_idx] if optimal_idx < len(recall) else 0.0
    
    # Multi-sample specific metrics
    avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
    avg_consistency = np.mean(all_consistencies) if all_consistencies else 0.0
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision_at_optimal:.4f}")
    print(f"Recall: {recall_at_optimal:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Average Confidence: {avg_confidence:.4f}")
    print(f"Average Consistency: {avg_consistency:.4f}")
    
    return {
        'model_type': model_type,
        'auroc': auroc,
        'accuracy': accuracy,
        'precision': precision_at_optimal,
        'recall': recall_at_optimal,
        'optimal_threshold': optimal_threshold,
        'avg_confidence': avg_confidence,
        'avg_consistency': avg_consistency,
        'n_examples': len(outputs)
    }

def compare_models(fine_tuned_file, baseline_file):
    """Compare fine-tuned and baseline models"""
    print("=" * 70)
    print("COMPARING MULTI-SAMPLE EVALUATION RESULTS")
    print("=" * 70)
    
    # Calculate metrics for both models
    fine_tuned_results = calculate_auroc_from_outputs(fine_tuned_file, "fine_tuned")
    print("\n")
    baseline_results = calculate_auroc_from_outputs(baseline_file, "baseline")
    
    if not fine_tuned_results or not baseline_results:
        print("Error: Could not calculate metrics for one or both models")
        return
    
    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<20} {'Fine-tuned':<15} {'Baseline':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'AUROC':<20} {fine_tuned_results['auroc']:<15.4f} {baseline_results['auroc']:<15.4f} {fine_tuned_results['auroc'] - baseline_results['auroc']:<15.4f}")
    print(f"{'Accuracy':<20} {fine_tuned_results['accuracy']:<15.4f} {baseline_results['accuracy']:<15.4f} {fine_tuned_results['accuracy'] - baseline_results['accuracy']:<15.4f}")
    print(f"{'Precision':<20} {fine_tuned_results['precision']:<15.4f} {baseline_results['precision']:<15.4f} {fine_tuned_results['precision'] - baseline_results['precision']:<15.4f}")
    print(f"{'Recall':<20} {fine_tuned_results['recall']:<15.4f} {baseline_results['recall']:<15.4f} {fine_tuned_results['recall'] - baseline_results['recall']:<15.4f}")
    print(f"{'Avg Confidence':<20} {fine_tuned_results['avg_confidence']:<15.4f} {baseline_results['avg_confidence']:<15.4f} {fine_tuned_results['avg_confidence'] - baseline_results['avg_confidence']:<15.4f}")
    print(f"{'Avg Consistency':<20} {fine_tuned_results['avg_consistency']:<15.4f} {baseline_results['avg_consistency']:<15.4f} {fine_tuned_results['avg_consistency'] - baseline_results['avg_consistency']:<15.4f}")
    print("=" * 70)
    
    # Calculate percentage improvements
    print("\nPERCENTAGE IMPROVEMENTS:")
    print(f"AUROC: {((fine_tuned_results['auroc'] - baseline_results['auroc']) / baseline_results['auroc'] * 100):.2f}%")
    print(f"Accuracy: {((fine_tuned_results['accuracy'] - baseline_results['accuracy']) / baseline_results['accuracy'] * 100):.2f}%")
    print(f"Precision: {((fine_tuned_results['precision'] - baseline_results['precision']) / baseline_results['precision'] * 100):.2f}%")
    print(f"Recall: {((fine_tuned_results['recall'] - baseline_results['recall']) / baseline_results['recall'] * 100):.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Calculate AUROC from multi-sample outputs JSON files")
    parser.add_argument("--output_file", type=str, 
                       help="Path to multi-sample outputs JSON file (either fine_tuned or baseline)")
    parser.add_argument("--model_type", type=str, 
                       choices=["fine_tuned", "baseline"],
                       default="fine_tuned",
                       help="Model type for the output file")
    parser.add_argument("--fine_tuned_file", type=str,
                       help="Path to fine_tuned_multi_sample_outputs.json")
    parser.add_argument("--baseline_file", type=str,
                       help="Path to baseline_multi_sample_outputs.json")
    parser.add_argument("--compare", action="store_true",
                       help="Compare fine-tuned and baseline models")
    
    args = parser.parse_args()
    
    if args.compare:
        if not args.fine_tuned_file or not args.baseline_file:
            print("Error: --fine_tuned_file and --baseline_file are required for comparison")
            return
        compare_models(args.fine_tuned_file, args.baseline_file)
    else:
        if not args.output_file:
            print("Error: --output_file is required")
            return
        results = calculate_auroc_from_outputs(args.output_file, args.model_type)
        
        if results:
            print("\n=== SUMMARY ===")
            print(f"Model Type: {results['model_type']}")
            print(f"AUROC: {results['auroc']:.4f}")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
            print(f"Average Confidence: {results['avg_confidence']:.4f}")
            print(f"Average Consistency: {results['avg_consistency']:.4f}")
            print(f"Number of examples: {results['n_examples']}")

if __name__ == "__main__":
    main()


