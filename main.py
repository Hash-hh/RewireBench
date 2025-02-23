import numpy as np
import random
import torch

from pyg_dataset import SyntheticRewiringDataset

# Set a random seed for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Define dataset parameters.
root = './rewire_bench'
num_graphs = 12000
min_nodes = 30
max_nodes = 60
min_clusters = 2
max_clusters = 6
H = 0.8  # Homophily parameter
p_intra = 0.8
p_inter = 0.1
p_inter_remove = 0.9
p_intra_remove = 0.05
p_inter_add = 0.2
p_intra_add = 0.2
metric_list = ['modularity', 'spectral_gap', 'random_walk_stability', 'conductance']


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
                                   p_inter_add=p_inter_add,
                                   p_intra_add=p_intra_add,
                                   metrics_list=metric_list)

print("Dataset has", len(dataset), "graphs.")

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
