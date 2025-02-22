import networkx as nx
import numpy as np
import random


def generate_synthetic_graph(num_nodes, num_clusters, H=0.8, p_intra=0.8, p_inter=0.1):
    """
    Generate a synthetic graph using a stochastic block model.

    Parameters:
      num_nodes: Total number of nodes (e.g., between 40 and 80).
      num_clusters: Number of clusters (e.g., between 3 and 6).
      H: Homophily parameter; with probability H a node gets a “preferred” feature.
      p_intra: Probability of connection within clusters.
      p_inter: Probability of connection between clusters.

    Returns:
      A networkx graph with:
        - Each node having a 6D one-hot feature stored in attribute 'x'.
        - Each node having a 'block' attribute indicating its cluster.
        - Each edge having a 4D one-hot attribute 'edge_attr', set according to intra- or inter-cluster.
    """
    # Distribute nodes roughly equally among clusters.
    sizes = [num_nodes // num_clusters] * num_clusters
    for i in range(num_nodes % num_clusters):
        sizes[i] += 1

    # Create probability matrix for clusters.
    probs = [[p_intra if i == j else p_inter for j in range(num_clusters)]
             for i in range(num_clusters)]

    # Generate the graph using the stochastic block model.
    G = nx.stochastic_block_model(sizes, probs, seed=random.randint(0, 10000))

    # --- Assign Node Features ---
    # Each node gets a 6-dimensional one-hot vector.
    for node in G.nodes():
        block = G.nodes[node]['block']  # Cluster membership.
        # Use the block (modulo 6) as the "preferred" feature.
        preferred_feature = block % 6
        feat = np.zeros(6, dtype=np.float32)
        if random.random() < H:
            feat[preferred_feature] = 1.0
        else:
            rand_feature = random.randint(0, 5)
            feat[rand_feature] = 1.0
        G.nodes[node]['x'] = feat

    # --- Assign Edge Features ---
    # Each edge gets a 4D one-hot vector.
    # [1, 0, 0, 0] means intra-cluster; [0, 1, 0, 0] means inter-cluster.
    for u, v in G.edges():
        block_u = G.nodes[u]['block']
        block_v = G.nodes[v]['block']
        edge_feat = np.zeros(4, dtype=np.float32)
        if block_u == block_v:
            edge_feat[0] = 1.0  # Intra-cluster
        else:
            edge_feat[1] = 1.0  # Inter-cluster
        G[u][v]['edge_attr'] = edge_feat

    return G
