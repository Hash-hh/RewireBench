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
    # Distribute nodes with some randomness among clusters
    # Set minimum cluster size to ensure no tiny clusters
    min_cluster_size = max(2, num_nodes // (num_clusters * 5))
    remaining_nodes = num_nodes - min_cluster_size * num_clusters

    # Assign each cluster its minimum size
    sizes = [min_cluster_size] * num_clusters

    # Distribute remaining nodes randomly but proportionally
    if remaining_nodes > 0:
        # Generate random weights for proportional distribution
        weights = np.random.dirichlet(np.ones(num_clusters))
        # Distribute remaining nodes according to weights
        extra_nodes = np.random.multinomial(remaining_nodes, weights)
        sizes = [sizes[i] + extra_nodes[i] for i in range(num_clusters)]

    # Ensure no empty clusters in case of rounding errors
    for i in range(len(sizes)):
        if sizes[i] == 0:
            # Steal a node from the largest cluster
            largest_idx = np.argmax(sizes)
            sizes[largest_idx] -= 1
            sizes[i] += 1

    # Create probability matrix
    probs = [[p_intra if i == j else p_inter for j in range(num_clusters)]
             for i in range(num_clusters)]

    # Generate SBM graph
    G = nx.stochastic_block_model(sizes, probs, seed=random.randint(0, 100000))

    # Define Gaussian distributions - we will have num_clusters+num_features-1 distributions
    # to ensure unique combinations for each cluster
    num_gaussians = min(num_clusters + num_features - 1, 8)
    gaussian_params = []
    for i in range(num_gaussians):
        # Randomize mean within a reasonable range around base position
        base_mean = 2.0 * i + 0.5
        mean = base_mean + np.random.uniform(-0.3, 0.3)

        # Randomize standard deviation between 0.3 and 0.7
        std = np.random.uniform(0.3, 0.7)

        gaussian_params.append((mean, std))

    # Each cluster gets assigned a unique sequence of num_features distributions
    cluster_distributions = {}
    for cluster in range(num_clusters):
        # Start index cycles through available Gaussians
        start_idx = cluster % (num_gaussians - num_features + 1)
        # Assign num_features consecutive Gaussians from start_idx
        cluster_distributions[cluster] = list(range(start_idx, start_idx + num_features))

    # --- Assign Node Features ---
    for node in G.nodes():
        block = G.nodes[node]['block']
        # Determine source cluster (apply homophily parameter)
        source_cluster = block if random.random() < H else random.randint(0, num_clusters - 1)

        # Sample features from the appropriate distributions for this cluster
        feat = np.zeros(num_features, dtype=np.float32)
        for f in range(num_features):
            gaussian_idx = cluster_distributions[source_cluster][f]
            mean, std = gaussian_params[gaussian_idx]
            feat[f] = np.random.normal(mean, std)

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
