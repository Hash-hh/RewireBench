import numpy as np
import random
import torch

from pyg_dataset import SyntheticRewiringDataset

# Set a random seed for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Define dataset parameters.
root = './synthetic_dataset'
num_graphs = 2 #100
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

# Create and process the dataset.
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

# Example: Access a graph and view its attributes.
sample = dataset[0]
print("Sample graph:")
print("  Number of nodes:", sample.num_nodes)
print("  Original edge_index shape:", sample.org_edge_index.shape)
print("  Modified edge_index shape:", sample.edge_index.shape)
