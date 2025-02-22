from pyg_dataset import SyntheticRewiringDataset
import torch

# Load the dataset
dataset = SyntheticRewiringDataset(root='./synthetic_dataset')

# Get number of predictions per graph
num_preds = dataset[0].y.shape[0]
print(f"Number of predictions per graph: {num_preds}")

# Analyze each prediction separately
for i in range(num_preds):
    pred_i = torch.stack([data.y[i] for data in dataset])

    print(f"\nPrediction {i+1} statistics:")
    print(f"Mean: {pred_i.mean():.4f}")
    print(f"Std: {pred_i.std():.4f}")
    print(f"Min: {pred_i.min():.4f}")
    print(f"Max: {pred_i.max():.4f}")
