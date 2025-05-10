# block_experiments.py

import os
import yaml
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import subprocess
from pathlib import Path
import shutil
from itertools import product
from tqdm import tqdm


def run_block_experiment(param_name, param_values, num_runs=5, base_config_path="config.yaml"):
    """
    Run a block experiment by varying a specific parameter across multiple values and runs.

    Args:
        param_name: Name of the parameter to vary (e.g., "model.num_layers")
        param_values: List of values to try for the parameter
        num_runs: Number of runs for each parameter value
        base_config_path: Path to the base configuration file
    """
    # Create block experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    block_dir = os.path.join("block_experiments", f"{timestamp}_{param_name.replace('.', '_')}")
    os.makedirs(block_dir, exist_ok=True)

    # Copy base config
    shutil.copy(base_config_path, os.path.join(block_dir, "base_config.yaml"))

    # Load base config
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Store results
    results = []
    all_experiment_dirs = []

    # Calculate total number of experiments
    total_experiments = len(param_values) * num_runs

    # Create unified progress bar
    progress_bar = tqdm(total=total_experiments, desc="Block Experiment Progress")

    # For each parameter value
    for param_value in param_values:
        print(f"\n{'-' * 80}")
        print(f"Setting {param_name} = {param_value}")

        value_results = []
        value_dirs = []

        # For each run
        for run in range(1, num_runs + 1):
            # Update progress bar description
            progress_bar.set_description(f"Parameter: {param_name}={param_value}, Run: {run}/{num_runs}")

            # Create modified config
            modified_config = update_nested_dict(base_config.copy(), param_name, param_value)

            # Create run directory
            run_dir = os.path.join(block_dir, f"{param_name.replace('.', '_')}_{param_value}_run{run}")
            os.makedirs(run_dir, exist_ok=True)

            # Save config
            config_path = os.path.join(run_dir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(modified_config, f)

            # Run experiment
            exp_dir = run_experiment(config_path)
            value_dirs.append(exp_dir)

            # Collect metrics
            metrics = load_experiment_metrics(exp_dir)
            metrics["run"] = run
            metrics[param_name] = param_value
            value_results.append(metrics)

            # Update progress bar
            progress_bar.update(1)

        # Collect all runs for this parameter value
        results.extend(value_results)
        all_experiment_dirs.append((param_value, value_dirs))

    # Close progress bar
    progress_bar.close()

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Save results
    df.to_csv(os.path.join(block_dir, "results.csv"), index=False)

    # Generate visualizations
    generate_visualizations(df, param_name, block_dir)

    # Generate summary report
    generate_summary(df, param_name, block_dir, all_experiment_dirs)

    return block_dir, df


def update_nested_dict(d, key_path, value):
    """Update a nested dictionary with a value at the specified key path"""
    keys = key_path.split(".")
    current = d
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return d


def run_experiment(config_path):
    """Run an experiment with the given config file and return the experiment directory"""
    # Set environment variable to use the specific config
    env = os.environ.copy()
    env["CONFIG_PATH"] = config_path

    # Run the main.py script with the specific config
    cmd = ["python", "main.py"]
    output = subprocess.check_output(cmd, env=env).decode()

    # Extract experiment directory from output
    for line in output.splitlines():
        if "Experiment results saved to:" in line:
            return line.split(":")[-1].strip()

    raise ValueError("Could not determine experiment directory from output")


def load_experiment_metrics(exp_dir):
    """Load metrics from an experiment directory"""
    metrics = {}

    # Load training history
    with open(os.path.join(exp_dir, "training_history.json"), "r") as f:
        history = json.load(f)

    # Extract key metrics
    metrics["best_val_accuracy"] = history.get("best_val_accuracy")
    metrics["best_test_accuracy"] = history.get("best_test_accuracy")
    metrics["epochs"] = len(history.get("train_losses", []))
    metrics["best_epoch"] = history.get("best_epoch")

    final_metrics = history.get("final_test_metrics", {})
    metrics["test_accuracy"] = final_metrics.get("accuracy")
    metrics["test_precision"] = final_metrics.get("precision")
    metrics["test_recall"] = final_metrics.get("recall")
    metrics["test_f1"] = final_metrics.get("f1")

    return metrics


def generate_visualizations(df, param_name, block_dir):
    """Generate visualizations for the block experiment"""
    # Create visualization directory
    viz_dir = os.path.join(block_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Set Seaborn style
    sns.set(style="whitegrid", context="talk")

    # Define metrics to visualize
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1"]

    # Calculate mean and std for each parameter value and metric
    agg_df = df.groupby(param_name)[metrics].agg(['mean', 'std']).reset_index()

    # Plot each metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Extract mean and std for the metric
        means = agg_df[(metric, 'mean')].values
        stds = agg_df[(metric, 'std')].values
        x_values = agg_df[param_name].values

        # Plot with error bars
        ax.errorbar(x_values, means, yerr=stds, fmt='o-', capsize=5,
                    linewidth=2, markersize=8, label=f'{metric} ± std')

        # Add raw data points
        for val in x_values:
            points = df[df[param_name] == val][metric].values
            ax.scatter([val] * len(points), points, alpha=0.4, color='gray')

        # Format plot
        ax.set_title(f"{metric.replace('test_', '').title()}")
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric.replace('test_', '').title())

        # Set the x-axis to show all parameter values
        ax.set_xticks(x_values)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{param_name}_metrics.png"), dpi=300, bbox_inches='tight')

    # Create a combined plot
    plt.figure(figsize=(12, 8))

    for metric in metrics:
        means = agg_df[(metric, 'mean')].values
        x_values = agg_df[param_name].values
        plt.plot(x_values, means, 'o-', linewidth=2, markersize=8, label=metric.replace('test_', '').title())

    plt.title(f"Performance Metrics vs {param_name}")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(x_values)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{param_name}_combined.png"), dpi=300, bbox_inches='tight')

    # Generate heatmap for all runs
    plt.figure(figsize=(10, 6))
    pivot_data = df.pivot_table(
        index='run',
        columns=param_name,
        values='test_accuracy'
    )

    sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.3f', cbar_kws={'label': 'Test Accuracy'})
    plt.title(f'Test Accuracy for each Run and {param_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{param_name}_heatmap.png"), dpi=300, bbox_inches='tight')


def generate_summary(df, param_name, block_dir, experiment_dirs):
    """Generate a summary of the block experiment"""
    summary_path = os.path.join(block_dir, "summary.md")

    # Calculate statistics
    stats = df.groupby(param_name)[['test_accuracy', 'test_precision', 'test_recall', 'test_f1']].agg(['mean', 'std'])

    # Format the table for markdown
    stats_table = stats.to_markdown(floatfmt=".4f")

    with open(summary_path, "w") as f:
        f.write(f"# Block Experiment: {param_name}\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Experiment Settings\n\n")
        f.write(f"- Parameter: `{param_name}`\n")
        f.write(f"- Values: {', '.join(map(str, df[param_name].unique()))}\n")
        f.write(f"- Runs per value: {df.groupby(param_name).size().iloc[0]}\n\n")

        f.write("## Results Summary\n\n")
        f.write(stats_table + "\n\n")

        f.write("## Best Configuration\n\n")
        best_idx = stats[('test_accuracy', 'mean')].idxmax()
        f.write(f"- Best {param_name}: {best_idx}\n")
        f.write(
            f"- Mean Accuracy: {stats.loc[best_idx, ('test_accuracy', 'mean')]:.4f} ± {stats.loc[best_idx, ('test_accuracy', 'std')]:.4f}\n")
        f.write(
            f"- Mean F1 Score: {stats.loc[best_idx, ('test_f1', 'mean')]:.4f} ± {stats.loc[best_idx, ('test_f1', 'std')]:.4f}\n\n")

        f.write("## Experiment Directories\n\n")
        for param_value, dirs in experiment_dirs:
            f.write(f"### {param_name} = {param_value}\n\n")
            for i, d in enumerate(dirs, 1):
                f.write(f"- Run {i}: `{d}`\n")
            f.write("\n")


def multiple_block_experiments(experiments_config):
    """
    Run multiple block experiments as specified in the config.

    Args:
        experiments_config: Dictionary mapping parameter names to lists of values
    """
    results = {}

    # Calculate total number of experiments
    total_experiments = sum(len(values) * 5 for values in experiments_config.values())

    # Create a master progress bar
    with tqdm(total=total_experiments, desc="Overall Progress") as master_bar:
        for param_name, param_values in experiments_config.items():
            print(f"\n{'=' * 80}")
            print(f"Starting block experiment for {param_name}")
            print(f"{'=' * 80}")

            # Create a custom wrapper to update both progress bars
            def run_with_progress_tracking():
                block_dir, df = run_block_experiment(param_name, param_values)
                # The individual experiment's progress bar will handle its own updates
                master_bar.update(len(param_values) * 5)
                return block_dir, df

            block_dir, df = run_with_progress_tracking()
            results[param_name] = (block_dir, df)

    return results


def grid_search_experiment(param_grid, num_runs=3, base_config_path="config.yaml"):
    """
    Run a grid search experiment by trying all combinations of specified parameters.

    Args:
        param_grid: Dictionary mapping parameter names to lists of values
        num_runs: Number of runs for each parameter combination
        base_config_path: Path to the base configuration file
    """
    # Create block experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_names = "_".join(k.replace('.', '_') for k in param_grid.keys())
    block_dir = os.path.join("block_experiments", f"{timestamp}_grid_search_{param_names}")
    os.makedirs(block_dir, exist_ok=True)

    # Copy base config
    shutil.copy(base_config_path, os.path.join(block_dir, "base_config.yaml"))

    # Load base config
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    # Store results
    results = []
    all_experiment_dirs = []

    # Calculate total number of experiments
    total_experiments = len(param_combinations) * num_runs

    # Create unified progress bar
    progress_bar = tqdm(total=total_experiments, desc="Grid Search Progress")

    # For each parameter combination
    for i, combination in enumerate(param_combinations):
        param_dict = dict(zip(param_names, combination))

        print(f"\n{'-' * 80}")
        print(f"Testing combination {i + 1}/{len(param_combinations)}: {param_dict}")

        combo_results = []
        combo_dirs = []

        # For each run
        for run in range(1, num_runs + 1):
            # Update progress bar description
            param_str = ", ".join([f"{k}={v}" for k, v in param_dict.items()])
            progress_bar.set_description(f"Combo {i + 1}/{len(param_combinations)}: {param_str}, Run {run}/{num_runs}")

            # Create modified config
            modified_config = base_config.copy()
            for param_name, param_value in param_dict.items():
                modified_config = update_nested_dict(modified_config, param_name, param_value)

            # Create directory name based on parameters
            dir_name = "_".join([f"{p.split('.')[-1]}_{v}" for p, v in param_dict.items()])
            run_dir = os.path.join(block_dir, f"{dir_name}_run{run}")
            os.makedirs(run_dir, exist_ok=True)

            # Save config
            config_path = os.path.join(run_dir, "config.yaml")
            with open(config_path, "w") as f:
                yaml.dump(modified_config, f)

            # Run experiment
            exp_dir = run_experiment(config_path)
            combo_dirs.append(exp_dir)

            # Collect metrics
            metrics = load_experiment_metrics(exp_dir)
            metrics["run"] = run
            for param_name, param_value in param_dict.items():
                metrics[param_name] = param_value
            combo_results.append(metrics)

            # Update progress bar
            progress_bar.update(1)

        # Collect all runs for this parameter combination
        results.extend(combo_results)
        all_experiment_dirs.append((param_dict, combo_dirs))

    # Close progress bar
    progress_bar.close()

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Save results
    df.to_csv(os.path.join(block_dir, "results.csv"), index=False)

    # Generate grid search visualizations
    generate_grid_visualizations(df, param_grid, block_dir)

    # Generate summary report
    generate_grid_summary(df, param_grid, block_dir, all_experiment_dirs)

    return block_dir, df


def generate_grid_visualizations(df, param_grid, block_dir):
    """Generate visualizations for the grid search experiment"""
    # Create visualization directory
    viz_dir = os.path.join(block_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Set Seaborn style
    sns.set(style="whitegrid", context="talk")

    # Define metrics to visualize
    metrics = ["test_accuracy", "test_precision", "test_recall", "test_f1"]

    # If we have a 2D parameter grid, create heatmaps
    if len(param_grid) == 2:
        param_names = list(param_grid.keys())

        for metric in metrics:
            plt.figure(figsize=(10, 8))

            # Calculate mean for each parameter combination
            pivot_data = df.pivot_table(
                index=param_names[0],
                columns=param_names[1],
                values=metric,
                aggfunc='mean'
            )

            # Create heatmap
            sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.3f', cbar_kws={'label': metric.title()})
            plt.title(f'Mean {metric.replace("test_", "").title()} for Grid Search')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"grid_{metric}_heatmap.png"), dpi=300, bbox_inches='tight')

    # For any parameter count, create parallel coordinates plot
    plt.figure(figsize=(12, 8))

    # Normalize the columns for better visualization
    columns_to_plot = list(param_grid.keys()) + ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
    plot_df = df.groupby(list(param_grid.keys()))[metrics].mean().reset_index()

    # Create parallel coordinates plot
    pd.plotting.parallel_coordinates(
        plot_df, 'test_accuracy',
        cols=[col for col in columns_to_plot if col != 'test_accuracy'],
        colormap=plt.cm.viridis
    )

    plt.title('Parameter Combinations and Performance Metrics')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "grid_parallel_coords.png"), dpi=300, bbox_inches='tight')


def generate_grid_summary(df, param_grid, block_dir, experiment_dirs):
    """Generate a summary of the grid search experiment"""
    summary_path = os.path.join(block_dir, "summary.md")

    # Calculate statistics grouped by all parameters
    group_cols = list(param_grid.keys())
    stats = df.groupby(group_cols)[['test_accuracy', 'test_precision', 'test_recall', 'test_f1']].agg(['mean', 'std'])

    # Format the table for markdown
    stats_table = stats.to_markdown(floatfmt=".4f")

    with open(summary_path, "w") as f:
        f.write(f"# Grid Search Experiment\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Experiment Settings\n\n")
        for param_name, values in param_grid.items():
            f.write(f"- Parameter: `{param_name}`, Values: {', '.join(map(str, values))}\n")
        f.write(f"- Runs per combination: {df.groupby(group_cols).size().iloc[0]}\n\n")

        f.write("## Results Summary\n\n")
        f.write(stats_table + "\n\n")

        f.write("## Best Configuration\n\n")
        best_idx = stats[('test_accuracy', 'mean')].idxmax()
        if not isinstance(best_idx, tuple):
            best_idx = (best_idx,)

        param_dict = dict(zip(group_cols, best_idx))
        f.write("Best parameter combination:\n")
        for param, value in param_dict.items():
            f.write(f"- {param}: {value}\n")

        best_row = stats.loc[best_idx]
        f.write(f"\nPerformance:\n")
        f.write(
            f"- Mean Accuracy: {best_row[('test_accuracy', 'mean')]:.4f} ± {best_row[('test_accuracy', 'std')]:.4f}\n")
        f.write(
            f"- Mean Precision: {best_row[('test_precision', 'mean')]:.4f} ± {best_row[('test_precision', 'std')]:.4f}\n")
        f.write(f"- Mean Recall: {best_row[('test_recall', 'mean')]:.4f} ± {best_row[('test_recall', 'std')]:.4f}\n")
        f.write(f"- Mean F1 Score: {best_row[('test_f1', 'mean')]:.4f} ± {best_row[('test_f1', 'std')]:.4f}\n\n")


if __name__ == "__main__":
    # Create block experiments directory if it doesn't exist
    if not os.path.exists("block_experiments"):
        os.makedirs("block_experiments")

    # Example usage:
    # Single parameter experiment
    run_block_experiment("model.num_layers", [1, 2, 3, 4, 5], num_runs=3)

    # Multiple parameter experiments
    # multiple_block_experiments({
    #     "model.hidden_dim": [32, 64, 128, 256],
    #     "model.dropout": [0.0, 0.1, 0.3, 0.5]
    # })

    # Grid search example
    # grid_search_experiment({
    #     "model.num_layers": [1, 2, 3, 4],
    #     "model.hidden_dim": [64, 128, 256]
    # }, num_runs=3)
