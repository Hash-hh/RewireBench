import networkx as nx
import numpy as np
import random
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx


def modify_graph(G, p_inter_remove=0.9, p_intra_remove=0.05, p_add=0.1):
    """
    Rewire a graph by removing and adding edges.

    Parameters:
      G: A networkx graph with node attribute 'block' and edge attribute 'edge_attr'.
      p_inter_remove: Probability to remove an inter-cluster edge.
      p_intra_remove: Probability to remove an intra-cluster edge.
      p_add: Probability to add an edge between two nodes that are not connected.

    Returns:
      A modified copy of the networkx graph.
    """
    # Copy the original graph so we don't modify it in place.
    G_mod = G.copy()

    # --- Removal Phase ---
    edges_to_remove = []
    for u, v in list(G_mod.edges()):
        block_u = G_mod.nodes[u]['block']
        block_v = G_mod.nodes[v]['block']
        if block_u != block_v:
            if random.random() < p_inter_remove:
                edges_to_remove.append((u, v))
        else:
            if random.random() < p_intra_remove:
                edges_to_remove.append((u, v))
    G_mod.remove_edges_from(edges_to_remove)

    # --- Addition Phase ---
    nodes = list(G_mod.nodes())
    for i, u in enumerate(nodes):
        for v in nodes[ i +1:]:
            if not G_mod.has_edge(u, v):
                block_u = G_mod.nodes[u]['block']
                block_v = G_mod.nodes[v]['block']
                # Optionally, treat intra-cluster and inter-cluster candidates differently.
                prob = p_add if block_u == block_v else p_add * 0.5
                if random.random() < prob:
                    edge_feat = np.zeros(4, dtype=np.float32)
                    if block_u == block_v:
                        edge_feat[0] = 1.0  # Intra-cluster
                    else:
                        edge_feat[1] = 1.0  # Inter-cluster
                    G_mod.add_edge(u, v, edge_attr=edge_feat)

    return G_mod
