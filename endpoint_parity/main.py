# main.py

import os
import yaml
import random
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader
import shutil

from endpoint_parity_dataset import EndpointParityDataset
from models import GNN
from train import train_epoch, evaluate
from visualize_endpoint_parity import visualize_samples


def create_experiment_dir(config):
    """Create and return experiment directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config["model"]["conv_type"]
    exp_name = f"{timestamp}_{model_type}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_metrics(metrics_dict, exp_dir, filename):
    """Save metrics dictionary to JSON file"""
    # Convert any numpy/torch values to Python native types
    serializable_dict = {}
    for key, value in metrics_dict.items():
        if isinstance(value, list):
            serializable_dict[key] = [float(x) if hasattr(x, 'item') else x for x in value]
        else:
            serializable_dict[key] = float(value) if hasattr(value, 'item') else value

    with open(os.path.join(exp_dir, filename), 'w') as f:
        json.dump(serializable_dict, f, indent=2)


def main():
    # 1) Load config
    cfg = yaml.safe_load(open("config.yaml"))

    # Create experiment directory
    exp_dir = create_experiment_dir(cfg)

    # Save config
    shutil.copy("config.yaml", os.path.join(exp_dir, "config.yaml"))

    # 2) Generate / load dataset
    ds_cfg = cfg["dataset"]
    dataset = EndpointParityDataset(
        root=ds_cfg["root"],
        L=ds_cfg["L"],
        M=ds_cfg["M"],
        num_graphs=ds_cfg["num_graphs"],
        n_distractors=ds_cfg["n_distractors"],
        force_regen=ds_cfg["force_regen"]
    )

    # 3) Train/val/test split with stratification
    n = len(dataset)
    all_idx = list(range(n))
    ys = [int(dataset[i].y) for i in all_idx]

    test_size = cfg["train"]["test_split"]
    val_size = cfg["train"]["val_split"]

    idx_trainval, idx_test = train_test_split(
        all_idx, test_size=test_size, stratify=ys, random_state=42
    )
    ys_trainval = [int(dataset[i].y) for i in idx_trainval]
    val_rel = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_rel, stratify=ys_trainval, random_state=42
    )

    # Save dataset split information
    split_info = {
        'dataset_size': n,
        'train_size': len(idx_train),
        'val_size': len(idx_val),
        'test_size': len(idx_test),
        'train_ratio': 1 - test_size - val_size,
        'val_ratio': val_size,
        'test_ratio': test_size
    }
    save_metrics(split_info, exp_dir, 'dataset_split.json')

    # 4) DataLoaders
    batch_size = cfg["train"]["batch_size"]
    train_loader = DataLoader(
        dataset, batch_size, sampler=SubsetRandomSampler(idx_train)
    )
    val_loader = DataLoader(
        dataset, batch_size, sampler=SubsetRandomSampler(idx_val)
    )
    test_loader = DataLoader(
        dataset, batch_size, sampler=SubsetRandomSampler(idx_test)
    )

    # 5) Build model & optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_cfg = cfg["model"]
    model = GNN(
        in_dim=2,
        hid_dim=m_cfg["hidden_dim"],
        num_layers=m_cfg["num_layers"],
        conv_type=m_cfg["conv_type"],
        readout=m_cfg["readout"],
        dropout=m_cfg["dropout"]
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"])
    )

    # 6) Training loop with metric tracking
    best_val = 0
    train_losses = []
    val_accs = []
    test_accs = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        loss = train_epoch(model, train_loader, opt, device)
        val_acc = evaluate(model, val_loader, device)
        test_acc = evaluate(model, test_loader, device)

        # Count positive and negative predictions on test set
        model.eval()
        pos_count, neg_count = 0, 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)
                pos_count += (pred == 1).sum().item()
                neg_count += (pred == 0).sum().item()

        total_count = pos_count + neg_count
        pos_percent = pos_count / total_count * 100
        neg_percent = neg_count / total_count * 100

        train_losses.append(loss)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch:03d} | loss {loss:.4f} | val_acc {val_acc:.4f} | test_acc {test_acc:.4f} | "
              f"pred: {pos_count} pos ({pos_percent:.1f}%), {neg_count} neg ({neg_percent:.1f}%)")

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_test_acc = test_acc
            # Save best model to experiment directory
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))

    # 7) Calculate detailed metrics on test set
    model.load_state_dict(torch.load(os.path.join(exp_dir, "best_model.pth")))
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    # Calculate classification metrics
    test_metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='binary'),
        'recall': recall_score(all_labels, all_preds, average='binary'),
        'f1': f1_score(all_labels, all_preds, average='binary')
    }

    # 8) Save training history
    training_history = {
        'train_losses': train_losses,
        'val_accuracies': val_accs,
        'test_accuracies': test_accs,
        'best_epoch': best_epoch,
        'best_val_accuracy': best_val,
        'best_test_accuracy': best_test_acc,
        'final_test_metrics': test_metrics
    }
    save_metrics(training_history, exp_dir, 'training_history.json')

    # 9) Plot and save training curves
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, 'r-', label='Validation')
    plt.plot(test_accs, 'g-', label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'training_curves.png'))
    plt.close()

    # 10) Create experiment summary
    summary = f"""
====== EXPERIMENT SUMMARY ======
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model: {cfg['model']['conv_type']} (layers={cfg['model']['num_layers']}, hidden_dim={cfg['model']['hidden_dim']})
Dataset: EndpointParity (L={ds_cfg['L']}, M={ds_cfg['M']}, size={n})

Best model: epoch {best_epoch}/{cfg['train']['epochs']}
Best validation accuracy: {best_val:.4f}

Final test metrics:
- Accuracy: {test_metrics['accuracy']:.4f}
- Precision: {test_metrics['precision']:.4f}
- Recall: {test_metrics['recall']:.4f}
- F1 Score: {test_metrics['f1']:.4f}
"""

    with open(os.path.join(exp_dir, 'experiment_summary.txt'), 'w') as f:
        f.write(summary)

    print(summary)
    print(f"Experiment results saved to: {exp_dir}")

    # 11) Visualization
    vis_cfg = cfg["visualization"]
    viz_dir = os.path.join(exp_dir, vis_cfg["out_dir"])
    visualize_samples(dataset,
                      vis_cfg["num_samples_per_class"],
                      viz_dir,
                      exp_dir)


if __name__ == "__main__":
    main()
