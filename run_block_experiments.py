import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time
import shutil
from chain_blocks_dataset import ChainBlocksDataset
from train import run_experiment, set_seed
import copy


def run_block_count_experiments(
        min_blocks=2,
        max_blocks=20,
        runs_per_block=3,
        dataset_config_path='config/config.yaml',
        train_config_path='config/config_weighted_sequence.yaml'
):
    """
    Run multiple experiments with varying number of blocks in the dataset,
    comparing MAE for normal vs. oracle edges.

    Args:
        min_blocks: Minimum number of blocks to use
        max_blocks: Maximum number of blocks to use
        runs_per_block: Number of runs for each block count
        dataset_config_path: Path to dataset configuration file
        train_config_path: Path to training configuration file
    """
    # Load configurations
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)

    with open(train_config_path, 'r') as f:
        train_config = yaml.safe_load(f)

    # Set up storage for results
    results = {
        'block_count': [],
        'normal_mae_mean': [],
        'normal_mae_std': [],
        'oracle_mae_mean': [],
        'oracle_mae_std': [],
        'improvement_percent': []
    }

    # Extract task type for naming
    task_type = dataset_config.get('task', {}).get('type', 'default')

    # Create a timestamped directory for this meta-experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_exp_dir = os.path.join("experiments", f"{timestamp}_{task_type}_blocks_{min_blocks}_to_{max_blocks}")
    os.makedirs(meta_exp_dir, exist_ok=True)

    # Save the experiment configuration
    meta_config = {
        'min_blocks': min_blocks,
        'max_blocks': max_blocks,
        'runs_per_block': runs_per_block,
        'dataset_config_path': dataset_config_path,
        'train_config_path': train_config_path,
        'timestamp': timestamp,
        'task_type': task_type
    }

    with open(os.path.join(meta_exp_dir, 'meta_config.yaml'), 'w') as f:
        yaml.dump(meta_config, f)

    # Save original configs
    shutil.copy(dataset_config_path, os.path.join(meta_exp_dir, 'original_dataset_config.yaml'))
    shutil.copy(train_config_path, os.path.join(meta_exp_dir, 'original_train_config.yaml'))

    # Create temporary config paths for modified configs
    temp_dataset_config_path = os.path.join(meta_exp_dir, 'temp_dataset_config.yaml')
    temp_train_config_path = os.path.join(meta_exp_dir, 'temp_train_config.yaml')

    # Set runs in train config
    train_config['experiment']['num_runs'] = runs_per_block

    # Set seed for reproducibility
    base_seed = train_config['experiment']['seed']
    set_seed(base_seed)

    # Store start time
    overall_start_time = time.time()

    # Create a file to list all experiment paths
    exp_index_path = os.path.join(meta_exp_dir, 'experiment_index.txt')
    with open(exp_index_path, 'w') as f:
        f.write(f"Block Count Experiments Index\n")
        f.write(f"Task Type: {task_type}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Run experiments for each block count
    for block_count in range(min_blocks, max_blocks + 1):
        print(f"\n{'=' * 60}")
        print(f"RUNNING EXPERIMENTS FOR {block_count} BLOCKS")
        print(f"{'=' * 60}\n")

        # Modify dataset config for this block count
        modified_dataset_config = copy.deepcopy(dataset_config)
        modified_dataset_config['graph']['num_blocks'] = block_count

        # Save the modified dataset config
        with open(temp_dataset_config_path, 'w') as f:
            yaml.dump(modified_dataset_config, f)

        # Update train config to point to the modified dataset config
        modified_train_config = copy.deepcopy(train_config)
        modified_train_config['dataset']['config_path'] = temp_dataset_config_path

        # Also ensure we're using a different output directory for each block count
        modified_train_config['experiment']['block_count'] = block_count

        # Save the modified train config
        with open(temp_train_config_path, 'w') as f:
            yaml.dump(modified_train_config, f)

        # Run the experiment with the modified configs
        start_time = time.time()
        exp_results, exp_dir = run_experiment(temp_train_config_path)
        elapsed_time = time.time() - start_time

        # Create a reference file instead of symlink
        exp_basename = os.path.basename(exp_dir)
        ref_file_path = os.path.join(meta_exp_dir, f"blocks_{block_count}_reference.txt")
        with open(ref_file_path, 'w') as f:
            f.write(f"Block Count: {block_count}\n")
            f.write(f"Task Type: {task_type}\n")
            f.write(f"Experiment Directory: {os.path.abspath(exp_dir)}\n")
            f.write(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Also update the master index
        with open(exp_index_path, 'a') as f:
            f.write(f"Block Count {block_count}: {os.path.abspath(exp_dir)}\n")

        # Extract MAE results
        normal_maes = [run['test_metrics']['mae'] for run in exp_results['normal_edges']['runs']]
        normal_mae_mean = np.mean(normal_maes)
        normal_mae_std = np.std(normal_maes)

        oracle_maes = [run['test_metrics']['mae'] for run in exp_results['oracle_edges']['runs']] if exp_results[
            'oracle_edges'] else [0]
        oracle_mae_mean = np.mean(oracle_maes) if exp_results['oracle_edges'] else 0
        oracle_mae_std = np.std(oracle_maes) if exp_results['oracle_edges'] else 0

        # Calculate improvement
        if exp_results['oracle_edges']:
            improvement = (normal_mae_mean - oracle_mae_mean) / normal_mae_mean * 100
        else:
            improvement = 0

        # Store results
        results['block_count'].append(block_count)
        results['normal_mae_mean'].append(normal_mae_mean)
        results['normal_mae_std'].append(normal_mae_std)
        results['oracle_mae_mean'].append(oracle_mae_mean)
        results['oracle_mae_std'].append(oracle_mae_std)
        results['improvement_percent'].append(improvement)

        print(f"\nCompleted experiments for {block_count} blocks in {elapsed_time:.2f} seconds")
        print(f"Normal MAE: {normal_mae_mean:.4f} ± {normal_mae_std:.4f}")
        if exp_results['oracle_edges']:
            print(f"Oracle MAE: {oracle_mae_mean:.4f} ± {oracle_mae_std:.4f}")
            print(f"Improvement: {improvement:.2f}%")

        # Save intermediate results to CSV in case of interruption
        interim_df = pd.DataFrame({
            'block_count': results['block_count'],
            'normal_mae_mean': results['normal_mae_mean'],
            'normal_mae_std': results['normal_mae_std'],
            'oracle_mae_mean': results['oracle_mae_mean'],
            'oracle_mae_std': results['oracle_mae_std'],
            'improvement_percent': results['improvement_percent']
        })
        interim_df.to_csv(os.path.join(meta_exp_dir, 'interim_results.csv'), index=False)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    csv_path = os.path.join(meta_exp_dir, 'block_count_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Create visualizations
    create_visualizations(results_df, meta_exp_dir, task_type)

    # Print total time
    total_time = time.time() - overall_start_time
    print(f"\nCompleted all experiments in {total_time:.2f} seconds")
    print(f"All results saved to {meta_exp_dir}")

    return results_df, meta_exp_dir


def create_visualizations(results_df, output_dir, task_type="default"):
    """Create visualizations for the block count experiments"""
    # 1. Line plot with error bars for MAE vs block count
    plt.figure(figsize=(12, 8))

    # Plot normal edges
    plt.errorbar(
        results_df['block_count'],
        results_df['normal_mae_mean'],
        yerr=results_df['normal_mae_std'],
        fmt='o-',
        capsize=5,
        label='Normal Edges',
        color='blue'
    )

    # Plot oracle edges
    plt.errorbar(
        results_df['block_count'],
        results_df['oracle_mae_mean'],
        yerr=results_df['oracle_mae_std'],
        fmt='o-',
        capsize=5,
        label='Oracle Edges (with shortcut)',
        color='green'
    )

    plt.xlabel('Number of Blocks')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title(f'MAE vs. Block Count: Normal vs. Oracle Edges\nTask: {task_type}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(output_dir, 'mae_vs_blocks.png'), dpi=300)
    plt.close()

    # 2. Bar chart for each block count
    plt.figure(figsize=(15, 10))

    x = np.arange(len(results_df['block_count']))
    width = 0.35

    # Plot bars
    plt.bar(
        x - width / 2,
        results_df['normal_mae_mean'],
        width,
        yerr=results_df['normal_mae_std'],
        label='Normal Edges',
        capsize=5,
        color='blue'
    )

    plt.bar(
        x + width / 2,
        results_df['oracle_mae_mean'],
        width,
        yerr=results_df['oracle_mae_std'],
        label='Oracle Edges (with shortcut)',
        capsize=5,
        color='green'
    )

    plt.xlabel('Number of Blocks')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title(f'MAE Comparison for Different Block Counts\nTask: {task_type}')
    plt.xticks(x, results_df['block_count'])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(output_dir, 'mae_comparison_bars.png'), dpi=300)
    plt.close()

    # 3. Percentage improvement plot
    plt.figure(figsize=(12, 8))

    plt.plot(
        results_df['block_count'],
        results_df['improvement_percent'],
        'o-',
        color='purple',
        linewidth=2
    )

    plt.xlabel('Number of Blocks')
    plt.ylabel('Improvement (%)')
    plt.title(f'Oracle Edge Improvement Over Normal Edges\nTask: {task_type}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(output_dir, 'improvement_percentage.png'), dpi=300)
    plt.close()

    # 4. Heatmap-like visualization showing relative performance
    plt.figure(figsize=(14, 6))

    # Create data for the heatmap
    data = np.array([
        results_df['normal_mae_mean'],
        results_df['oracle_mae_mean']
    ])

    # Create a heatmap using imshow
    im = plt.imshow(data, cmap='coolwarm_r', aspect='auto')

    # Add colorbar
    plt.colorbar(im, label='MAE')

    # Set tick labels
    plt.yticks([0, 1], ['Normal Edges', 'Oracle Edges'])
    plt.xticks(np.arange(len(results_df['block_count'])), results_df['block_count'])

    plt.xlabel('Number of Blocks')
    plt.title(f'MAE Heatmap: Normal vs. Oracle Edges\nTask: {task_type}')

    # Add text annotations
    for i in range(2):
        for j in range(len(results_df['block_count'])):
            text = plt.text(j, i, f"{data[i, j]:.4f}",
                            ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(output_dir, 'mae_heatmap.png'), dpi=300)
    plt.close()

    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    # Default settings
    MIN_BLOCKS = 5
    MAX_BLOCKS = 10
    RUNS_PER_BLOCK = 2

    import argparse

    parser = argparse.ArgumentParser(description='Run experiments with varying block counts')
    parser.add_argument('--min-blocks', type=int, default=MIN_BLOCKS, help='Minimum number of blocks')
    parser.add_argument('--max-blocks', type=int, default=MAX_BLOCKS, help='Maximum number of blocks')
    parser.add_argument('--runs', type=int, default=RUNS_PER_BLOCK, help='Number of runs per block count')
    parser.add_argument('--dataset-config', default='config/config_polynomial.yaml', help='Path to dataset config')
    parser.add_argument('--train-config', default='config/train_config.yaml', help='Path to training config')

    args = parser.parse_args()

    print(f"Running experiments with block counts from {args.min_blocks} to {args.max_blocks}...")
    print(f"{args.runs} runs will be performed for each block count")

    # Run the experiments
    results_df, meta_exp_dir = run_block_count_experiments(
        min_blocks=args.min_blocks,
        max_blocks=args.max_blocks,
        runs_per_block=args.runs,
        dataset_config_path=args.dataset_config,
        train_config_path=args.train_config
    )
