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

# graph generation parameters
num_graphs = 12000
min_nodes = 30
max_nodes = 60
min_clusters = 2
max_clusters = 6
num_features = 1  # Number of features per node. Note: only one feature is used for y metrics. TODO: add more features.
H = 1  # Homophily parameter; with probability H a node gets its preferred one-hot feature. (put 1 to always get the cluster id feature)
p_intra = 0.8  # Intra-cluster connection probability.
p_inter = 0.1  # Inter-cluster connection probability.

# graph rewiring (modification) parameters
p_inter_remove = 0.9  # Probability to remove an inter-cluster edge.
p_intra_remove = 0.05  # Probability to remove an intra-cluster edge.
p_inter_add = 0.2  # Probability to add an inter-cluster edge.
p_intra_add = 0.2  # Probability to add an intra-cluster edge.

# List of metrics to compute.
metric_list = [
                'local_easy1', 'local_easy2', 'local_easy3',
                'local_hard1', 'local_hard2', 'local_hard3',
                # 'modularity', 'spectral_gap', 'random_walk_stability', 'conductance'
               ]

# Create the dataset (if not already processed, it will be generated).
dataset = SyntheticRewiringDataset(root=root,
                                   num_graphs=num_graphs,
                                   min_nodes=min_nodes,
                                   max_nodes=max_nodes,
                                   min_clusters=min_clusters,
                                   max_clusters=max_clusters,
                                   num_features=num_features,
                                   H=H,
                                   p_intra=p_intra,
                                   p_inter=p_inter,
                                   p_inter_remove=p_inter_remove,
                                   p_intra_remove=p_intra_remove,
                                   p_inter_add=p_inter_add,
                                   p_intra_add=p_intra_add,
                                   metrics_list=metric_list)

print("Dataset has", len(dataset), "graphs.")
