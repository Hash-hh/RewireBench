import numpy as np
import random
import torch

from pyg_dataset import SyntheticRewiringDataset
from y_vals import compute_y_values

# Set a random seed for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Define dataset parameters.
root = './synthetic_dataset'
num_graphs = 100
min_nodes = 40
max_nodes = 80
min_clusters = 3
max_clusters = 6
H = 0.8
p_intra = 0.8
p_inter = 0.1
p_inter_remove = 0.9
p_intra_remove = 0.05
p_add = 0.1

# Create the dataset (if not already processed, it will be generated).
dataset = SyntheticRewiringDataset(root=root,
                                   num_graphs=num_graphs,
                                   min_nodes=min_nodes,
                                   max_nodes=max_nodes,
                                   min_clusters=min_clusters,
                                   max_clusters=max_clusters,
                                   H=H,
                                   p_intra=p_intra,
                                   p_inter=p_inter,
                                   p_inter_remove=p_inter_remove,
                                   p_intra_remove=p_intra_remove,
                                   p_add=p_add)

print("Dataset has", len(dataset), "graphs.")

# Compute regression y values using n randomly initialized GNN models.
num_models = 8  # For example, 3 different random models.
dataset = compute_y_values(dataset, num_models=num_models, batch_size=16,
                           in_channels=6, hidden_channels=16, num_layers=3)

# Save the updated dataset (with y values) for later loading in PyG.
torch.save((dataset.data, dataset.slices), dataset.processed_paths[0])

# Example: Print out the y attribute of the first graph.
sample = dataset[0]
print("Sample graph regression targets (y):", sample.y)

# After computing y values
all_y = torch.cat([data.y for data in dataset])
mean_y = all_y.mean()
std_y = all_y.std()

print(f"Y values statistics:")
print(f"Mean: {mean_y:.4f}")
print(f"Std: {std_y:.4f}")
print(f"Min: {all_y.min():.4f}")
print(f"Max: {all_y.max():.4f}")