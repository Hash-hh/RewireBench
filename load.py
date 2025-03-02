from pyg_dataset import SyntheticRewiringDataset
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
dataset = SyntheticRewiringDataset(root='rewire_bench')

# Access data through data_list instead of standard indexing
data_list = dataset
# data_list = dataset.data_list  # use this if __getitem__ is not overridden in SyntheticRewiringDataset
print(f"Number of graphs in dataset: {len(data_list)}")

# Collect statistics
num_nodes_list = []
num_clusters_list = []
node_features_by_cluster = {}
edge_density_list = []
intra_edge_ratio_list = []

# Process each graph
for graph_data in data_list:
    num_nodes = graph_data.num_nodes
    num_clusters = graph_data.num_clusters

    num_nodes_list.append(num_nodes)
    num_clusters_list.append(num_clusters)

    # Calculate edge density
    num_edges = graph_data.edge_index.shape[1] // 2  # Divide by 2 as edges are stored twice
    max_possible_edges = num_nodes * (num_nodes - 1) // 2
    edge_density = num_edges / max_possible_edges
    edge_density_list.append(edge_density)

    # Count intra vs inter cluster edges
    edge_attr = graph_data.edge_attr
    if edge_attr is not None:
        intra_edges = (edge_attr[:, 0] == 1).sum().item() // 2  # First dimension is for intra-cluster
        intra_ratio = intra_edges / num_edges if num_edges > 0 else 0
        intra_edge_ratio_list.append(intra_ratio)

    # Extract node features by cluster
    # Note: This requires mapping back to original networkx graph structure
    # For now, gather overall feature statistics
    features = graph_data.x.numpy()
    for i in range(num_clusters):
        if i not in node_features_by_cluster:
            node_features_by_cluster[i] = []

# Print statistics
print("\n--- Dataset Statistics ---")
print(f"Total number of graphs: {len(data_list)}")
print(f"Nodes per graph: min={min(num_nodes_list)}, max={max(num_nodes_list)}, avg={np.mean(num_nodes_list):.1f}")
print(
    f"Clusters per graph: min={min(num_clusters_list)}, max={max(num_clusters_list)}, avg={np.mean(num_clusters_list):.1f}")
print(
    f"Edge density: min={min(edge_density_list):.3f}, max={max(edge_density_list):.3f}, avg={np.mean(edge_density_list):.3f}")
if intra_edge_ratio_list:
    print(
        f"Intra-cluster edge ratio: min={min(intra_edge_ratio_list):.3f}, max={max(intra_edge_ratio_list):.3f}, avg={np.mean(intra_edge_ratio_list):.3f}")

# Feature statistics across all graphs
all_features = np.vstack([g.x.numpy() for g in data_list])
print(f"\nNode feature statistics (across all graphs):")
print(f"Mean: {np.mean(all_features):.4f}")
print(f"Std: {np.std(all_features):.4f}")
print(f"Min: {np.min(all_features):.4f}")
print(f"Max: {np.max(all_features):.4f}")

# Y values statistics
if hasattr(data_list[0], 'y') and data_list[0].y is not None:
    num_metrics = data_list[0].y.shape[0]
    print(f"\nNumber of metrics per graph: {num_metrics}")

    for i in range(num_metrics):
        metric_vals = torch.stack([d.y[i] for d in data_list])
        print(f"\nOriginal metric {i + 1} statistics:")
        print(f"Mean: {metric_vals.mean().item():.4f}")
        print(f"Std: {metric_vals.std().item():.4f}")
        print(f"Min: {metric_vals.min().item():.4f}")
        print(f"Max: {metric_vals.max().item():.4f}")

        if hasattr(data_list[0], 'y_rewire') and data_list[0].y_rewire is not None:
            rewire_vals = torch.stack([d.y_rewire[i] for d in data_list])
            print(f"Rewired metric {i + 1} statistics:")
            print(f"Mean: {rewire_vals.mean().item():.4f}")
            print(f"Std: {rewire_vals.std().item():.4f}")
            print(f"Min: {rewire_vals.min().item():.4f}")
            print(f"Max: {rewire_vals.max().item():.4f}")
