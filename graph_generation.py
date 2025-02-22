import networkx as nx
import numpy as np
import random


def generate_synthetic_graph(num_nodes, num_clusters, H=0.8, p_intra=0.8, p_inter=0.5):
    """
    Generate a synthetic graph using a stochastic block model.

    Parameters:
      num_nodes: Total number of nodes.
      num_clusters: Number of clusters (max 6).
      H: Homophily parameter; with probability H a node gets its preferred one-hot feature. (put 1 to always get the cluster id feature)
      p_intra: Intra-cluster connection probability.
      p_inter: Inter-cluster connection probability.

    Returns:
      A networkx graph with:
        - Node features: 6D one-hot vector in attribute 'x'.
        - Community membership in attribute 'block'.
        - Edge features: 2D one-hot vector in attribute 'edge_attr':
              [1, 0] for intra-cluster, [0, 1] for inter-cluster.
    """
    # Distribute nodes roughly equally among clusters.
    sizes = [num_nodes // num_clusters] * num_clusters
    for i in range(num_nodes % num_clusters):
        sizes[i] += 1

    # Create probability matrix.
    probs = [[p_intra if i == j else p_inter for j in range(num_clusters)]
             for i in range(num_clusters)]

    # Generate SBM graph.
    G = nx.stochastic_block_model(sizes, probs, seed=random.randint(0, 100000))

    # --- Assign Node Features ---
    for node in G.nodes():
        block = G.nodes[node]['block']
        preferred_feature = block % 6
        feat = np.zeros(6, dtype=np.float32)
        if random.random() < H:
            feat[preferred_feature] = 1.0
        else:
            rand_feature = random.randint(0, 5)
            feat[rand_feature] = 1.0
        G.nodes[node]['x'] = feat

    # --- Assign Edge Features ---
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
