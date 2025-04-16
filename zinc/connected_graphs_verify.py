import torch
import networkx as nx
from torch_geometric.datasets import ZINC

# Load dataset
dataset = ZINC(root="data", subset=True)

# Check for disconnected graphs
for i, data in enumerate(dataset):
    edge_index = data.edge_index.numpy()
    num_nodes = data.num_nodes

    # Convert to NetworkX Graph
    G = nx.Graph()
    G.add_edges_from(edge_index.T)
    G.add_nodes_from(range(num_nodes))  # Ensure isolated nodes are included

    # Check if the graph is connected
    if not nx.is_connected(G):
        print(f"Graph {i} is disconnected!")
