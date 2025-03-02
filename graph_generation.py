import networkx as nx
import numpy as np
import random


def generate_synthetic_graph(num_nodes, num_clusters, num_features=1, H=0.8, p_intra=0.8, p_inter=0.5):
    """
    Generate a synthetic graph using a stochastic block model with Gaussian node features.

    Parameters:
      num_nodes: Total number of nodes.
      num_clusters: Number of clusters (max 6).
      num_features: Number of features per node.
      H: Homophily parameter; with probability H a node gets features from its own cluster's distributions.
      p_intra: Intra-cluster connection probability.
      p_inter: Inter-cluster connection probability.

    Returns:
      A networkx graph with:
        - Node features: num_features-dimensional vector in attribute 'x'.
        - Community membership in attribute 'block'.
        - Edge features: 2D one-hot vector in attribute 'edge_attr':
              [1, 0] for intra-cluster, [0, 1] for inter-cluster.
    """
    # Distribute nodes roughly equally among clusters
    sizes = [num_nodes // num_clusters] * num_clusters
    for i in range(num_nodes % num_clusters):
        sizes[i] += 1

    # Create probability matrix
    probs = [[p_intra if i == j else p_inter for j in range(num_clusters)]
             for i in range(num_clusters)]

    # Generate SBM graph
    G = nx.stochastic_block_model(sizes, probs, seed=random.randint(0, 100000))

    # Define Gaussian distributions for each cluster
    # Each cluster has num_features different distributions
    gaussian_params = {}
    for cluster in range(min(num_clusters, 6)):
        gaussian_params[cluster] = []
        for feat_idx in range(num_features):
            # Create distinct distributions for each cluster and feature
            mean = 2.0 * cluster + 0.5 * feat_idx
            std = 0.5
            gaussian_params[cluster].append((mean, std))

    # --- Assign Node Features ---
    for node in G.nodes():
        block = G.nodes[node]['block']
        # Determine source cluster (apply homophily parameter)
        source_cluster = block if random.random() < H else random.randint(0, min(num_clusters, 6) - 1)

        # Sample features from the appropriate distributions
        feat = np.zeros(num_features, dtype=np.float32)
        for f in range(num_features):
            mean, std = gaussian_params[source_cluster % 6][f]
            feat[f] = np.random.normal(mean, std)

        G.nodes[node]['x'] = feat

    # --- Assign Edge Features --- (unchanged)
    for u, v in G.edges():
        block_u = G.nodes[u]['block']
        block_v = G.nodes[v]['block']
        edge_feat = np.zeros(2, dtype=np.float32)
        if block_u == block_v:
            edge_feat[0] = 1.0  # Intra-cluster
        else:
            edge_feat[1] = 1.0  # Inter-cluster
        G[u][v]['edge_attr'] = edge_feat

    return G
