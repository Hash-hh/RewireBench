import networkx as nx
import numpy as np
from y_metrics.global_metrics import compute_modularity, compute_spectral_gap, compute_random_walk_stability, compute_conductance
from y_metrics.local_easy_metrics import compute_local_easy1, compute_local_easy2, compute_local_easy3
from y_metrics.local_hard_metrics import compute_local_hard1, compute_local_hard2, compute_local_hard3


def compute_all_metrics(G, metrics=None):
    """
    Compute specified metrics and return as a numpy array.

    Parameters:
        G: networkx graph
        metrics: list of strings, valid values are 'modularity', 'spectral_gap',
                'random_walk_stability', 'conductance'. If None, compute all metrics.

    Returns:
        numpy array containing the computed metrics in the same order as requested
    """
    metric_funcs = {
        'local_easy1': compute_local_easy1,
        'local_easy2': compute_local_easy2,
        'local_easy3': compute_local_easy3,
        'local_hard1': compute_local_hard1,
        'local_hard2': compute_local_hard2,
        'local_hard3': compute_local_hard3,
        'modularity': compute_modularity,
        'spectral_gap': compute_spectral_gap,
        'random_walk_stability': lambda g: compute_random_walk_stability(g, T=10, num_walks=100),
        'conductance': compute_conductance
    }

    if metrics is None:
        metrics = list(metric_funcs.keys())

    results = [metric_funcs[m](G) for m in metrics]
    return np.array(results, dtype=np.float32)
