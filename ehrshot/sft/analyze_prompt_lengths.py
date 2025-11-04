#!/usr/bin/env python3
"""
Analyze and plot the distribution of prompt lengths in the SFT dataset.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from loguru import logger
import argparse
from pathlib import Path


def load_dataset(dataset_path: str) -> list:
    """Load the SFT dataset"""
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    logger.info(f"Loaded {len(dataset)} examples")
    return dataset


def extract_prompt_content(example: dict) -> str:
    """Extract the prompt content from an example (same as format_prompt)"""
    # Extract patient history from the conversation
    user_content = example['conversations'][0]['content']
    
    # Return as single completion prompt (no chat format)
    return user_content


def extract_output_content(example: dict) -> str:
    """Extract the assistant response content from an example"""
    # Find the assistant response in the conversation
    for conversation in example['conversations']:
        if conversation['role'] == 'assistant':
            return conversation['content']
    
    # If no assistant response found, return empty string
    return ""


def analyze_prompt_lengths(dataset: list, tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct") -> dict:
    """Analyze prompt and output lengths in characters and tokens"""
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    prompt_char_lengths = []
    prompt_token_lengths = []
    output_char_lengths = []
    output_token_lengths = []
    
    logger.info("Analyzing prompt and output lengths...")
    for i, example in enumerate(dataset):
        if i % 100 == 0:
            logger.info(f"Processing example {i}/{len(dataset)}")
        
        # Extract prompt content
        prompt = extract_prompt_content(example)
        
        # Calculate prompt character length
        prompt_char_length = len(prompt)
        prompt_char_lengths.append(prompt_char_length)
        
        # Calculate prompt token length
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_token_length = len(prompt_tokens)
        prompt_token_lengths.append(prompt_token_length)
        
        # Extract output content
        output = extract_output_content(example)
        
        # Calculate output character length
        output_char_length = len(output)
        output_char_lengths.append(output_char_length)
        
        # Calculate output token length
        output_tokens = tokenizer.encode(output, add_special_tokens=False)
        output_token_length = len(output_tokens)
        output_token_lengths.append(output_token_length)
    
    return {
        'prompt_char_lengths': prompt_char_lengths,
        'prompt_token_lengths': prompt_token_lengths,
        'output_char_lengths': output_char_lengths,
        'output_token_lengths': output_token_lengths,
        'prompt_char_stats': {
            'mean': np.mean(prompt_char_lengths),
            'median': np.median(prompt_char_lengths),
            'std': np.std(prompt_char_lengths),
            'min': np.min(prompt_char_lengths),
            'max': np.max(prompt_char_lengths),
            'q25': np.percentile(prompt_char_lengths, 25),
            'q75': np.percentile(prompt_char_lengths, 75),
            'q90': np.percentile(prompt_char_lengths, 90),
            'q95': np.percentile(prompt_char_lengths, 95),
            'q99': np.percentile(prompt_char_lengths, 99)
        },
        'prompt_token_stats': {
            'mean': np.mean(prompt_token_lengths),
            'median': np.median(prompt_token_lengths),
            'std': np.std(prompt_token_lengths),
            'min': np.min(prompt_token_lengths),
            'max': np.max(prompt_token_lengths),
            'q25': np.percentile(prompt_token_lengths, 25),
            'q75': np.percentile(prompt_token_lengths, 75),
            'q90': np.percentile(prompt_token_lengths, 90),
            'q95': np.percentile(prompt_token_lengths, 95),
            'q99': np.percentile(prompt_token_lengths, 99)
        },
        'output_char_stats': {
            'mean': np.mean(output_char_lengths),
            'median': np.median(output_char_lengths),
            'std': np.std(output_char_lengths),
            'min': np.min(output_char_lengths),
            'max': np.max(output_char_lengths),
            'q25': np.percentile(output_char_lengths, 25),
            'q75': np.percentile(output_char_lengths, 75),
            'q90': np.percentile(output_char_lengths, 90),
            'q95': np.percentile(output_char_lengths, 95),
            'q99': np.percentile(output_char_lengths, 99)
        },
        'output_token_stats': {
            'mean': np.mean(output_token_lengths),
            'median': np.median(output_token_lengths),
            'std': np.std(output_token_lengths),
            'min': np.min(output_token_lengths),
            'max': np.max(output_token_lengths),
            'q25': np.percentile(output_token_lengths, 25),
            'q75': np.percentile(output_token_lengths, 75),
            'q90': np.percentile(output_token_lengths, 90),
            'q95': np.percentile(output_token_lengths, 95),
            'q99': np.percentile(output_token_lengths, 99)
        }
    }


def plot_distributions(analysis_results: dict, output_dir: str = "./prompt_analysis"):
    """Create plots of the prompt and output length distributions"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    prompt_char_lengths = analysis_results['prompt_char_lengths']
    prompt_token_lengths = analysis_results['prompt_token_lengths']
    output_char_lengths = analysis_results['output_char_lengths']
    output_token_lengths = analysis_results['output_token_lengths']
    prompt_char_stats = analysis_results['prompt_char_stats']
    prompt_token_stats = analysis_results['prompt_token_stats']
    output_char_stats = analysis_results['output_char_stats']
    output_token_stats = analysis_results['output_token_stats']
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots for prompt and output lengths
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('Prompt and Output Length Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Prompt character length histogram
    axes[0, 0].hist(prompt_char_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(prompt_char_stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {prompt_char_stats["mean"]:.0f}')
    axes[0, 0].axvline(prompt_char_stats['median'], color='orange', linestyle='--', linewidth=2, label=f'Median: {prompt_char_stats["median"]:.0f}')
    axes[0, 0].axvline(prompt_char_stats['q95'], color='green', linestyle='--', linewidth=2, label=f'95th percentile: {prompt_char_stats["q95"]:.0f}')
    axes[0, 0].set_xlabel('Character Length')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Prompt Character Lengths')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Prompt token length histogram
    axes[0, 1].hist(prompt_token_lengths, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].axvline(prompt_token_stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {prompt_token_stats["mean"]:.0f}')
    axes[0, 1].axvline(prompt_token_stats['median'], color='orange', linestyle='--', linewidth=2, label=f'Median: {prompt_token_stats["median"]:.0f}')
    axes[0, 1].axvline(prompt_token_stats['q95'], color='green', linestyle='--', linewidth=2, label=f'95th percentile: {prompt_token_stats["q95"]:.0f}')
    axes[0, 1].set_xlabel('Token Length')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Prompt Token Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Output character length histogram
    axes[0, 2].hist(output_char_lengths, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 2].axvline(output_char_stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {output_char_stats["mean"]:.0f}')
    axes[0, 2].axvline(output_char_stats['median'], color='orange', linestyle='--', linewidth=2, label=f'Median: {output_char_stats["median"]:.0f}')
    axes[0, 2].axvline(output_char_stats['q95'], color='green', linestyle='--', linewidth=2, label=f'95th percentile: {output_char_stats["q95"]:.0f}')
    axes[0, 2].set_xlabel('Character Length')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distribution of Output Character Lengths')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Output token length histogram
    axes[0, 3].hist(output_token_lengths, bins=50, alpha=0.7, color='gold', edgecolor='black')
    axes[0, 3].axvline(output_token_stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {output_token_stats["mean"]:.0f}')
    axes[0, 3].axvline(output_token_stats['median'], color='orange', linestyle='--', linewidth=2, label=f'Median: {output_token_stats["median"]:.0f}')
    axes[0, 3].axvline(output_token_stats['q95'], color='green', linestyle='--', linewidth=2, label=f'95th percentile: {output_token_stats["q95"]:.0f}')
    axes[0, 3].set_xlabel('Token Length')
    axes[0, 3].set_ylabel('Frequency')
    axes[0, 3].set_title('Distribution of Output Token Lengths')
    axes[0, 3].legend()
    axes[0, 3].grid(True, alpha=0.3)
    
    # 5. Prompt character length box plot
    axes[1, 0].boxplot(prompt_char_lengths, vert=True, patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1, 0].set_ylabel('Character Length')
    axes[1, 0].set_title('Prompt Character Length Box Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 6. Prompt token length box plot
    axes[1, 1].boxplot(prompt_token_lengths, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightcoral', alpha=0.7))
    axes[1, 1].set_ylabel('Token Length')
    axes[1, 1].set_title('Prompt Token Length Box Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 7. Output character length box plot
    axes[1, 2].boxplot(output_char_lengths, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7))
    axes[1, 2].set_ylabel('Character Length')
    axes[1, 2].set_title('Output Character Length Box Plot')
    axes[1, 2].grid(True, alpha=0.3)
    
    # 8. Output token length box plot
    axes[1, 3].boxplot(output_token_lengths, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='gold', alpha=0.7))
    axes[1, 3].set_ylabel('Token Length')
    axes[1, 3].set_title('Output Token Length Box Plot')
    axes[1, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_path / "prompt_output_length_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {plot_path}")
    
    # Create a detailed statistics plot
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create a comparison of percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    prompt_char_percentiles = [prompt_char_stats[f'q{p}' if p != 50 else 'median'] for p in percentiles]
    prompt_token_percentiles = [prompt_token_stats[f'q{p}' if p != 50 else 'median'] for p in percentiles]
    output_char_percentiles = [output_char_stats[f'q{p}' if p != 50 else 'median'] for p in percentiles]
    output_token_percentiles = [output_token_stats[f'q{p}' if p != 50 else 'median'] for p in percentiles]
    
    x = np.arange(len(percentiles))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, prompt_char_percentiles, width, label='Prompt Character Length', alpha=0.7, color='skyblue')
    bars2 = ax.bar(x - 0.5*width, prompt_token_percentiles, width, label='Prompt Token Length', alpha=0.7, color='lightcoral')
    bars3 = ax.bar(x + 0.5*width, output_char_percentiles, width, label='Output Character Length', alpha=0.7, color='lightgreen')
    bars4 = ax.bar(x + 1.5*width, output_token_percentiles, width, label='Output Token Length', alpha=0.7, color='gold')
    
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Length')
    ax.set_title('Prompt and Output Length Percentiles Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{p}th' for p in percentiles])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=6)
    
    plt.tight_layout()
    
    # Save the percentiles plot
    percentiles_path = output_path / "prompt_output_length_percentiles.png"
    plt.savefig(percentiles_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved percentiles plot to {percentiles_path}")
    
    plt.show()


def print_statistics(analysis_results: dict):
    """Print detailed statistics"""
    prompt_char_stats = analysis_results['prompt_char_stats']
    prompt_token_stats = analysis_results['prompt_token_stats']
    output_char_stats = analysis_results['output_char_stats']
    output_token_stats = analysis_results['output_token_stats']
    
    print("\n" + "="*80)
    print("PROMPT AND OUTPUT LENGTH ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\n📊 PROMPT CHARACTER LENGTH STATISTICS:")
    print(f"   Count: {len(analysis_results['prompt_char_lengths'])}")
    print(f"   Mean: {prompt_char_stats['mean']:.1f}")
    print(f"   Median: {prompt_char_stats['median']:.1f}")
    print(f"   Std Dev: {prompt_char_stats['std']:.1f}")
    print(f"   Min: {prompt_char_stats['min']:.0f}")
    print(f"   Max: {prompt_char_stats['max']:.0f}")
    print(f"   25th percentile: {prompt_char_stats['q25']:.0f}")
    print(f"   75th percentile: {prompt_char_stats['q75']:.0f}")
    print(f"   90th percentile: {prompt_char_stats['q90']:.0f}")
    print(f"   95th percentile: {prompt_char_stats['q95']:.0f}")
    print(f"   99th percentile: {prompt_char_stats['q99']:.0f}")
    
    print(f"\n🔤 PROMPT TOKEN LENGTH STATISTICS:")
    print(f"   Count: {len(analysis_results['prompt_token_lengths'])}")
    print(f"   Mean: {prompt_token_stats['mean']:.1f}")
    print(f"   Median: {prompt_token_stats['median']:.1f}")
    print(f"   Std Dev: {prompt_token_stats['std']:.1f}")
    print(f"   Min: {prompt_token_stats['min']:.0f}")
    print(f"   Max: {prompt_token_stats['max']:.0f}")
    print(f"   25th percentile: {prompt_token_stats['q25']:.0f}")
    print(f"   75th percentile: {prompt_token_stats['q75']:.0f}")
    print(f"   90th percentile: {prompt_token_stats['q90']:.0f}")
    print(f"   95th percentile: {prompt_token_stats['q95']:.0f}")
    print(f"   99th percentile: {prompt_token_stats['q99']:.0f}")
    
    print(f"\n📊 OUTPUT CHARACTER LENGTH STATISTICS:")
    print(f"   Count: {len(analysis_results['output_char_lengths'])}")
    print(f"   Mean: {output_char_stats['mean']:.1f}")
    print(f"   Median: {output_char_stats['median']:.1f}")
    print(f"   Std Dev: {output_char_stats['std']:.1f}")
    print(f"   Min: {output_char_stats['min']:.0f}")
    print(f"   Max: {output_char_stats['max']:.0f}")
    print(f"   25th percentile: {output_char_stats['q25']:.0f}")
    print(f"   75th percentile: {output_char_stats['q75']:.0f}")
    print(f"   90th percentile: {output_char_stats['q90']:.0f}")
    print(f"   95th percentile: {output_char_stats['q95']:.0f}")
    print(f"   99th percentile: {output_char_stats['q99']:.0f}")
    
    print(f"\n🔤 OUTPUT TOKEN LENGTH STATISTICS:")
    print(f"   Count: {len(analysis_results['output_token_lengths'])}")
    print(f"   Mean: {output_token_stats['mean']:.1f}")
    print(f"   Median: {output_token_stats['median']:.1f}")
    print(f"   Std Dev: {output_token_stats['std']:.1f}")
    print(f"   Min: {output_token_stats['min']:.0f}")
    print(f"   Max: {output_token_stats['max']:.0f}")
    print(f"   25th percentile: {output_token_stats['q25']:.0f}")
    print(f"   75th percentile: {output_token_stats['q75']:.0f}")
    print(f"   90th percentile: {output_token_stats['q90']:.0f}")
    print(f"   95th percentile: {output_token_stats['q95']:.0f}")
    print(f"   99th percentile: {output_token_stats['q99']:.0f}")
    
    # Analysis for token limits
    print(f"\n🎯 TOKEN LIMIT ANALYSIS:")
    current_prompt_limit = 16384 # Current max_prompt_tokens
    current_output_limit = 4096  # Typical max_output_tokens
    
    prompt_over_limit = sum(1 for length in analysis_results['prompt_token_lengths'] if length > current_prompt_limit)
    prompt_over_limit_pct = (prompt_over_limit / len(analysis_results['prompt_token_lengths'])) * 100
    
    output_over_limit = sum(1 for length in analysis_results['output_token_lengths'] if length > current_output_limit)
    output_over_limit_pct = (output_over_limit / len(analysis_results['output_token_lengths'])) * 100
    
    print(f"   Current prompt token limit: {current_prompt_limit}")
    print(f"   Prompt examples over limit: {prompt_over_limit} ({prompt_over_limit_pct:.1f}%)")
    print(f"   Prompt examples under limit: {len(analysis_results['prompt_token_lengths']) - prompt_over_limit} ({100-prompt_over_limit_pct:.1f}%)")
    
    print(f"   Current output token limit: {current_output_limit}")
    print(f"   Output examples over limit: {output_over_limit} ({output_over_limit_pct:.1f}%)")
    print(f"   Output examples under limit: {len(analysis_results['output_token_lengths']) - output_over_limit} ({100-output_over_limit_pct:.1f}%)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    # Prompt recommendations
    if prompt_over_limit_pct < 1:
        print(f"   ✅ Prompt limit ({current_prompt_limit}) handles {100-prompt_over_limit_pct:.1f}% of examples well")
    elif prompt_over_limit_pct < 5:
        print(f"   ⚠️  Consider increasing prompt limit slightly - {prompt_over_limit_pct:.1f}% of examples exceed current limit")
    else:
        print(f"   ❌ Consider significantly increasing prompt limit - {prompt_over_limit_pct:.1f}% of examples exceed current limit")
    
    # Output recommendations
    if output_over_limit_pct < 1:
        print(f"   ✅ Output limit ({current_output_limit}) handles {100-output_over_limit_pct:.1f}% of examples well")
    elif output_over_limit_pct < 5:
        print(f"   ⚠️  Consider increasing output limit slightly - {output_over_limit_pct:.1f}% of examples exceed current limit")
    else:
        print(f"   ❌ Consider significantly increasing output limit - {output_over_limit_pct:.1f}% of examples exceed current limit")
    
    # Suggested limits
    suggested_prompt_95 = int(prompt_token_stats['q95'])
    suggested_prompt_99 = int(prompt_token_stats['q99'])
    suggested_output_95 = int(output_token_stats['q95'])
    suggested_output_99 = int(output_token_stats['q99'])
    
    print(f"   📈 Suggested prompt limit for 95% coverage: {suggested_prompt_95} tokens")
    print(f"   📈 Suggested prompt limit for 99% coverage: {suggested_prompt_99} tokens")
    print(f"   📈 Suggested output limit for 95% coverage: {suggested_output_95} tokens")
    print(f"   📈 Suggested output limit for 99% coverage: {suggested_output_99} tokens")


def main():
    parser = argparse.ArgumentParser(description="Analyze prompt lengths in SFT dataset")
    parser.add_argument("--dataset_path", type=str, 
                       default="/home/lrshi/llm/ehrshot-benchmark/ehrshot/sft/data/acute_mi_sft_dataset.json",
                       help="Path to the SFT dataset JSON file")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Tokenizer model name")
    parser.add_argument("--output_dir", type=str, default="./prompt_analysis",
                       help="Output directory for plots")
    parser.add_argument("--no_plots", action="store_true",
                       help="Skip generating plots, only print statistics")
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_dataset(args.dataset_path)
    
    # Analyze prompt lengths
    analysis_results = analyze_prompt_lengths(dataset, args.tokenizer_name)
    
    # Print statistics
    print_statistics(analysis_results)
    
    # Generate plots
    if not args.no_plots:
        plot_distributions(analysis_results, args.output_dir)
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()


