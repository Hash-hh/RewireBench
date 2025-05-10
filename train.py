import os
import yaml
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import time
import random
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any, Optional

from chain_blocks_dataset import ChainBlocksDataset
from gnn_models import GNN


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, device, use_oracle_edges=False):
    """Train model for one epoch"""
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Use oracle edges if specified
        edge_index = data.oracle_edge_index if use_oracle_edges else data.edge_index

        # Forward pass
        out = model(data.x, edge_index, data.batch)
        loss = F.mse_loss(out, data.y)

        # Backward pass
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, use_oracle_edges=False):
    """Evaluate model performance"""
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # Use oracle edges if specified
            edge_index = data.oracle_edge_index if use_oracle_edges else data.edge_index

            # Forward pass
            out = model(data.x, edge_index, data.batch)

            y_true.append(data.y.cpu().numpy())
            y_pred.append(out.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'r2': r2
    }


def train_and_evaluate(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        scheduler,
        device,
        config,
        run_dir,
        use_oracle_edges=False
):
    """Complete training and evaluation process"""
    edge_type = "oracle" if use_oracle_edges else "normal"
    epochs = config['training']['epochs']
    patience = config['training']['patience']

    # Create edge-specific directory
    edge_dir = os.path.join(run_dir, f"{edge_type}_edges")
    os.makedirs(edge_dir, exist_ok=True)

    best_val_metric = float('inf')  # For MSE/RMSE - lower is better
    best_epoch = 0
    best_model_state = None
    no_improve_count = 0

    train_loss_history = []
    val_metrics_history = []

    for epoch in tqdm(range(epochs), desc=f"Training ({edge_type} edges)"):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device, use_oracle_edges)
        train_loss_history.append(train_loss)

        # Validation
        val_metrics = evaluate(model, val_loader, device, use_oracle_edges)
        val_metrics_history.append(val_metrics)

        # Learning rate scheduler step
        if scheduler is not None:
            scheduler.step(val_metrics['rmse'])  # Use RMSE as the metric for scheduler

        # Early stopping check
        val_metric = val_metrics['rmse']  # Primary metric for model selection
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_count = 0

            # Save best model
            torch.save(best_model_state, os.path.join(edge_dir, 'best_model.pt'))
        else:
            no_improve_count += 1

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
                  f"Val RMSE: {val_metrics['rmse']:.4f}, Val R²: {val_metrics['r2']:.4f}")

        # Check early stopping
        if no_improve_count >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)
    print(f"Best model from epoch {best_epoch + 1} with val RMSE: {best_val_metric:.4f}")

    # Final evaluation on test set
    test_metrics = evaluate(model, test_loader, device, use_oracle_edges)
    print(
        f"Test Results - RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}, R²: {test_metrics['r2']:.4f}")

    # Save training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, 'b-', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss ({edge_type} edges)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([m['rmse'] for m in val_metrics_history], 'r-', label='RMSE')
    plt.plot([m['mae'] for m in val_metrics_history], 'g-', label='MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title(f'Validation Metrics ({edge_type} edges)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(edge_dir, 'training_curves.png'))
    plt.close()

    # Save training history as JSON
    history = {
        'train_loss': train_loss_history,
        'val_metrics': val_metrics_history,
        'best_epoch': best_epoch,
        'test_metrics': test_metrics
    }

    # Convert numpy values to Python native types for JSON serialization
    history_serializable = {}
    for key, value in history.items():
        if isinstance(value, list):
            if isinstance(value[0], dict):  # For val_metrics_history
                history_serializable[key] = [{k: float(v) for k, v in d.items()} for d in value]
            else:  # For train_loss_history
                history_serializable[key] = [float(x) for x in value]
        elif isinstance(value, dict):  # For test_metrics
            history_serializable[key] = {k: float(v) for k, v in value.items()}
        else:  # For best_epoch
            history_serializable[key] = value

    with open(os.path.join(edge_dir, 'training_history.json'), 'w') as f:
        json.dump(history_serializable, f, indent=2)

    return {
        'train_loss_history': train_loss_history,
        'val_metrics_history': val_metrics_history,
        'best_epoch': best_epoch,
        'best_val_metrics': {k: float(v) for k, v in val_metrics_history[best_epoch].items()},
        'test_metrics': {k: float(v) for k, v in test_metrics.items()}
    }


def run_experiment(config_path: str):
    """Run the complete experiment based on the configuration file"""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create a unique experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Add block count to experiment name if available
    block_count_str = ""
    if 'experiment' in config and 'block_count' in config['experiment']:
        block_count_str = f"_blocks{config['experiment']['block_count']}"

    exp_name = f"{timestamp}{block_count_str}_{config['model']['gnn_type']}_{config['experiment']['num_runs']}runs"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)


    # Save experiment configuration
    shutil.copy(config_path, os.path.join(exp_dir, 'train_config.yaml'))

    # Also save dataset configuration if available
    dataset_config_path = config['dataset']['config_path']
    if os.path.exists(dataset_config_path):
        shutil.copy(dataset_config_path, os.path.join(exp_dir, 'dataset_config.yaml'))

    # Set seed for reproducibility
    set_seed(config['experiment']['seed'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataset = ChainBlocksDataset(
        root=config['dataset']['root_dir'],
        config_path=config['dataset']['config_path']
    )

    # Split dataset into train/val/test sets
    dataset_size = len(dataset)
    train_size = int(dataset_size * config['dataset']['train_ratio'])
    val_size = int(dataset_size * config['dataset']['val_ratio'])
    test_size = dataset_size - train_size - val_size

    # Save dataset split information
    split_info = {
        'dataset_size': dataset_size,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'train_ratio': config['dataset']['train_ratio'],
        'val_ratio': config['dataset']['val_ratio'],
        'test_ratio': 1 - config['dataset']['train_ratio'] - config['dataset']['val_ratio']
    }

    with open(os.path.join(exp_dir, 'dataset_split.json'), 'w') as f:
        json.dump(split_info, f, indent=2)

    # Generate train/val/test split
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config['experiment']['seed'])
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    # Update model config with input feature dimension
    config['model']['num_features'] = dataset[0].x.size(1)

    # Multiple runs configuration
    num_runs = config['experiment']['num_runs']
    run_results = {
        'normal_edges': {'runs': []},
        'oracle_edges': {'runs': []} if config['experiment']['eval_oracle_edges'] else None
    }

    # Track overall start time
    start_time = time.time()

    # Run multiple experiments
    for run_idx in range(num_runs):
        print(f"\n=== Run {run_idx + 1}/{num_runs} ===")

        # Create a directory for this run
        run_dir = os.path.join(exp_dir, f"run_{run_idx + 1}")
        os.makedirs(run_dir, exist_ok=True)

        # Set a different seed for each run
        run_seed = config['experiment']['seed'] + run_idx
        set_seed(run_seed)

        # Save run-specific metadata
        run_metadata = {
            'run_number': run_idx + 1,
            'seed': run_seed,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(os.path.join(run_dir, 'run_metadata.json'), 'w') as f:
            json.dump(run_metadata, f, indent=2)

        # Normal edges run
        print("\nTraining with normal edges...")
        model = GNN(config).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=config['training']['scheduler_patience'],
            min_lr=1e-6
        )

        normal_results = train_and_evaluate(
            model, train_loader, val_loader, test_loader,
            optimizer, scheduler, device, config, run_dir, use_oracle_edges=False
        )
        run_results['normal_edges']['runs'].append(normal_results)

        # Oracle edges run (if specified)
        if config['experiment']['eval_oracle_edges']:
            print("\nTraining with oracle edges (shortcuts)...")
            model = GNN(config).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=config['training']['scheduler_patience'],
                min_lr=1e-6
            )

            oracle_results = train_and_evaluate(
                model, train_loader, val_loader, test_loader,
                optimizer, scheduler, device, config, run_dir, use_oracle_edges=True
            )
            run_results['oracle_edges']['runs'].append(oracle_results)

        # Create comparison plot for this run
        if config['experiment']['eval_oracle_edges']:
            plt.figure(figsize=(10, 6))

            metrics = ['rmse', 'mae', 'r2']
            normal_values = [normal_results['test_metrics'][m] for m in metrics]
            oracle_values = [oracle_results['test_metrics'][m] for m in metrics]

            x = np.arange(len(metrics))
            width = 0.35

            plt.bar(x - width / 2, normal_values, width, label='Normal Edges')
            plt.bar(x + width / 2, oracle_values, width, label='Oracle Edges (with shortcuts)')

            plt.xlabel('Metrics')
            plt.ylabel('Values')
            plt.title(f'Performance Comparison: Run {run_idx + 1}')
            plt.xticks(x, metrics)
            plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, 'performance_comparison.png'))
            plt.close()

    # Calculate aggregate statistics
    for edge_type in ['normal_edges', 'oracle_edges']:
        if edge_type == 'oracle_edges' and not config['experiment']['eval_oracle_edges']:
            continue

        # FIX: Collect metrics properly across runs
        metrics_keys = ['mse', 'rmse', 'mae', 'r2']
        mean_metrics = {}
        std_metrics = {}

        for metric in metrics_keys:
            # Extract this metric from all runs
            metric_values = [run['test_metrics'][metric] for run in run_results[edge_type]['runs']]
            mean_metrics[metric] = float(np.mean(metric_values))
            std_metrics[metric] = float(np.std(metric_values))

        run_results[edge_type]['mean'] = mean_metrics
        run_results[edge_type]['std'] = std_metrics

    # Print and save summary
    summary_text = "\n" + "=" * 50 + "\n"
    summary_text += f"EXPERIMENT SUMMARY ({num_runs} runs)\n"
    summary_text += "=" * 50 + "\n"

    summary_text += "\nNormal Edges Results:\n"
    for metric, value in run_results['normal_edges']['mean'].items():
        std = run_results['normal_edges']['std'][metric]
        summary_text += f"  {metric}: {value:.4f} ± {std:.4f}\n"

    if config['experiment']['eval_oracle_edges']:
        summary_text += "\nOracle Edges (with shortcuts) Results:\n"
        for metric, value in run_results['oracle_edges']['mean'].items():
            std = run_results['oracle_edges']['std'][metric]
            summary_text += f"  {metric}: {value:.4f} ± {std:.4f}\n"

        # Print relative improvement
        summary_text += "\nRelative Improvement (Oracle vs Normal):\n"
        for metric in run_results['normal_edges']['mean'].keys():
            normal = run_results['normal_edges']['mean'][metric]
            oracle = run_results['oracle_edges']['mean'][metric]

            if metric in ['mse', 'rmse', 'mae']:  # Lower is better
                imp = (normal - oracle) / normal * 100
                summary_text += f"  {metric}: {imp:.2f}% improvement\n"
            else:  # Higher is better (like R²)
                imp = (oracle - normal) / abs(normal) * 100 if normal != 0 else float('inf')
                summary_text += f"  {metric}: {imp:.2f}% improvement\n"

    # Total time
    total_time = time.time() - start_time
    summary_text += f"\nTotal experiment time: {total_time:.2f} seconds\n"

    # Print summary
    print(summary_text)

    # Save summary to file
    with open(os.path.join(exp_dir, 'experiment_summary.txt'), 'w') as f:
        f.write(summary_text)

    # Create final comparison plot across all runs
    if config['experiment']['eval_oracle_edges'] and num_runs > 0:
        plt.figure(figsize=(12, 6))

        metrics = ['rmse', 'mae', 'r2']
        normal_means = [run_results['normal_edges']['mean'][m] for m in metrics]
        normal_stds = [run_results['normal_edges']['std'][m] for m in metrics]
        oracle_means = [run_results['oracle_edges']['mean'][m] for m in metrics]
        oracle_stds = [run_results['oracle_edges']['std'][m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        plt.bar(x - width / 2, normal_means, width, label='Normal Edges', yerr=normal_stds, capsize=5)
        plt.bar(x + width / 2, oracle_means, width, label='Oracle Edges (with shortcuts)', yerr=oracle_stds, capsize=5)

        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title(f'Performance Comparison Across {num_runs} Runs')
        plt.xticks(x, metrics)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'overall_performance_comparison.png'))
        plt.close()

    # Save all results as JSON
    results_serializable = {}
    for edge_type in ['normal_edges', 'oracle_edges']:
        if edge_type == 'oracle_edges' and not config['experiment']['eval_oracle_edges']:
            continue

        results_serializable[edge_type] = {
            'mean': run_results[edge_type]['mean'],
            'std': run_results[edge_type]['std']
        }

    with open(os.path.join(exp_dir, 'experiment_results.json'), 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nExperiment results saved to: {exp_dir}")

    return run_results, exp_dir


if __name__ == "__main__":
    # Run experiment with default config
    config_path = 'config/train_config.yaml'
    results, exp_dir = run_experiment(config_path)
