import networkx as nx
import numpy as np
from y_metrics.global_metrics import compute_modularity, compute_spectral_gap, compute_random_walk_stability, compute_conductance
from y_metrics.local_easy_metrics import compute_local_easy1, compute_local_easy2, compute_local_easy3
from y_metrics.local_hard_metrics import compute_local_hard1, compute_local_hard2, compute_local_hard3


def compute_all_metrics(G, metrics=None, multiple=1, variability=True, edge_weight=0.3):
    """
    Compute specified metrics and return as a numpy array.

    Parameters:
        G: networkx graph
        metrics: list of strings, valid values are 'modularity', 'spectral_gap',
                'random_walk_stability', 'conductance'. If None, compute all metrics.

    Returns:
        numpy array containing the computed metrics in the same order as requested
    """

    # add multiple aux node features (like polynomial and exponential etc) to the original graph only for the local metrics
    G = add_auxiliary_node_features(G, multiple=multiple, variability=variability)

    metric_funcs = {
        'local_easy1': compute_local_easy1,
        'local_easy2': compute_local_easy2,
        'local_easy3': compute_local_easy3,
        'local_hard1': compute_local_hard1,
        'local_hard2': compute_local_hard2,
        'local_hard3': compute_local_hard3,
        'modularity': compute_modularity,
        'spectral_gap': compute_spectral_gap,
        'random_walk_stability': lambda g, edge_weight: compute_random_walk_stability(g, T=10, num_walks=100,
                                                                                      edge_weight=edge_weight),
        'conductance': compute_conductance
    }

    if metrics is None:
        metrics = list(metric_funcs.keys())

    results = [metric_funcs[m](G, edge_weight) for m in metrics]
    return np.array(results, dtype=np.float32)


def add_auxiliary_node_features(G, multiple=1, variability=True):
    """
    Add auxiliary node features to the graph by applying different transformations.

    Parameters:
        G: networkx graph with node features stored in node attribute 'x'
        multiple: int (0-5), how many times to multiply the feature dimension
                 if 0 or 1, return graph unchanged
        variability: bool, whether to use different parameters for different clusters

    Returns:
        Modified networkx graph with additional node features
    """
    if multiple <= 1:
        return G

    multiple = min(multiple, 5)  # Cap at 5x multiplication

    # Get number of clusters and original feature dimension
    num_clusters = len(set(nx.get_node_attributes(G, 'block').values()))

    # Get the original feature dimension
    sample_node = next(iter(G.nodes()))
    orig_feat_dim = len(G.nodes[sample_node]['x'])

    # Define transformation functions
    def polynomial_transform(x, a, b, c):
        return a * x ** 2 + b * x + c

    def exponential_transform(x, a, b):
        return a * np.exp(b * x)

    def logarithmic_transform(x, a, b):
        # Add small constant to avoid log(0)
        return a * np.log(np.abs(x) + 1.0) + b

    def sinusoidal_transform(x, a, b, c):
        return a * np.sin(b * x + c)

    def modulo_transform(x, a, b):
        return a * (x % b)

    # List of transformation functions
    transforms = [
        polynomial_transform,
        exponential_transform,
        logarithmic_transform,
        sinusoidal_transform,
        modulo_transform
    ]

    # Generate parameters for each transformation
    # If variability=True, generate different parameters per cluster
    transform_params = []
    for t_idx in range(len(transforms)):
        if t_idx == 0:  # Polynomial: (a, b, c)
            if variability:
                cluster_params = [
                    (0.1 + 0.05 * i, 0.5 - 0.1 * i, 0.2 + 0.1 * i)
                    for i in range(num_clusters)
                ]
            else:
                cluster_params = [(0.1, 0.5, 0.2)] * num_clusters

        elif t_idx == 1:  # Exponential: (a, b)
            if variability:
                cluster_params = [
                    (0.2 + 0.05 * i, 0.1 + 0.03 * i)
                    for i in range(num_clusters)
                ]
            else:
                cluster_params = [(0.2, 0.1)] * num_clusters

        elif t_idx == 2:  # Logarithmic: (a, b)
            if variability:
                cluster_params = [
                    (0.5 + 0.1 * i, 0.3 - 0.05 * i)
                    for i in range(num_clusters)
                ]
            else:
                cluster_params = [(0.5, 0.3)] * num_clusters

        elif t_idx == 3:  # Sinusoidal: (a, b, c)
            if variability:
                cluster_params = [
                    (0.3 + 0.05 * i, 0.5 + 0.1 * i, 0.1 * i)
                    for i in range(num_clusters)
                ]
            else:
                cluster_params = [(0.3, 0.5, 0)] * num_clusters

        else:  # Modulo: (a, b)
            if variability:
                cluster_params = [
                    (0.7 + 0.1 * i, 1 + 0.2 * i)
                    for i in range(num_clusters)
                ]
            else:
                cluster_params = [(0.7, 1.0)] * num_clusters

        transform_params.append(cluster_params)

    # Apply transformations to each node
    for node in G.nodes():
        block = G.nodes[node].get('block', 0)  # Default to block 0 if not specified
        orig_features = G.nodes[node]['x']

        # Create new feature vector
        new_features = np.zeros(orig_feat_dim * multiple, dtype=np.float32)
        new_features[:orig_feat_dim] = orig_features  # Keep original features

        # Add transformed features
        for m in range(1, multiple):
            transform_idx = (m - 1) % len(transforms)
            transform_func = transforms[transform_idx]

            # Get parameters for this cluster
            cluster_idx = min(block, len(transform_params[transform_idx]) - 1)
            params = transform_params[transform_idx][cluster_idx]

            # Apply transformation to get new features
            start_idx = m * orig_feat_dim
            end_idx = (m + 1) * orig_feat_dim

            for f_idx in range(orig_feat_dim):
                new_features[start_idx + f_idx] = transform_func(orig_features[f_idx], *params)

        # Update node features
        G.nodes[node]['x'] = new_features

    return G
