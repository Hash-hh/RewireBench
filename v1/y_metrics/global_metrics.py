import networkx as nx
import numpy as np
import random


def compute_modularity(G, edge_weight=0.3):
    """
    Compute normalized modularity Q in [0,1].
    Uses edge features to adjust edge weights.
    """
    # Create a copy of the graph to add weights
    G_copy = G.copy()

    # Set edge weights based on edge features
    for u, v in G.edges():
        edge_attr = G[u][v]['edge_attr']
        # Intra-cluster edges get higher weight, inter-cluster lower
        weight = 1.0 + edge_weight * (edge_attr[0] - edge_attr[1])
        G_copy[u][v]['weight'] = weight

    # Create communities from blocks
    blocks = nx.get_node_attributes(G, 'block')
    communities = []
    for b in set(blocks.values()):
        community = {n for n, blk in blocks.items() if blk == b}
        communities.append(community)

    # Use the standard 'weight' attribute
    Q = nx.algorithms.community.modularity(G_copy, communities, weight='weight')
    return (Q + 1) / 2


def compute_spectral_gap(G, edge_weight=0.3):
    """
    Compute the spectral gap defined as the second-smallest eigenvalue
    of the unnormalized Laplacian, normalized by the maximum eigenvalue.
    Incorporates edge features as weights in the Laplacian.
    Returns a value in [0,1] (if the graph is connected; else 0).
    """
    # Build weighted adjacency matrix using edge features
    weight_dict = {}
    for u, v in G.edges():
        edge_attr = G[u][v]['edge_attr']
        # Intra-cluster edges get higher weight
        weight = 1.0 + edge_weight * (edge_attr[0] - edge_attr[1])
        weight_dict[(u, v)] = weight

    try:
        # Use weighted Laplacian
        L = nx.laplacian_matrix(G, weight=lambda u, v: weight_dict.get((u, v), 1.0)).todense()
        eigenvalues = np.linalg.eigvals(L)
        eigenvalues = np.sort(np.real(eigenvalues))  # sort in ascending order
        if len(eigenvalues) < 2:
            return 0.0
        lam2 = eigenvalues[1]
        lam_max = eigenvalues[-1] if eigenvalues[-1] != 0 else 1.0
        return lam2 / lam_max
    except:
        # Fallback if there's an error with eigenvalue calculation
        return 0.0


def compute_random_walk_stability(G, T=10, num_walks=100, edge_weight=0.3):
    """
    Run random walks and compute the average fraction of steps that
    a walker stays in the same community as its starting node.
    Uses edge features to bias the random walk.
    Returns a value in [0,1].
    """
    blocks = nx.get_node_attributes(G, 'block')
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return 0.0

    # Get feature similarity function for node features
    def feature_similarity(feat1, feat2):
        diff = np.mean(np.abs(feat1 - feat2))
        return np.exp(-diff)  # Higher value for similar features

    total_fraction = 0.0
    for _ in range(num_walks):
        start = random.choice(nodes)
        start_block = blocks[start]
        start_features = G.nodes[start]['x']
        current = start
        same_count = 0

        for _ in range(T):
            neighbors = list(G.neighbors(current))
            if not neighbors:
                break

            # Calculate transition probabilities based on edge features and node features
            probs = []
            for neigh in neighbors:
                edge_attr = G[current][neigh]['edge_attr']
                neigh_features = G.nodes[neigh]['x']

                # Edge type weight (higher for intra-cluster)
                edge_factor = 1.0 + edge_weight * (edge_attr[0] - edge_attr[1])

                # Node feature similarity to starting node (higher for similar features)
                feat_sim = feature_similarity(start_features, neigh_features)

                # Combined weight
                weight = edge_factor * feat_sim
                probs.append(max(0.01, weight))  # Ensure positive

            # Normalize to probabilities
            total_weight = sum(probs)
            probs = [p / total_weight for p in probs] if total_weight > 0 else None

            # Choose next step based on weighted probabilities
            current = random.choices(neighbors, weights=probs, k=1)[0]

            if blocks[current] == start_block:
                same_count += 1

        total_fraction += same_count / T

    return total_fraction / num_walks


def compute_conductance(G, edge_weight=0.3):
    """
    Compute a conductance-like measure per community and average over communities.
    Uses edge features to weight the conductance calculation.
    Returns a value in [0,1].
    """
    blocks = nx.get_node_attributes(G, 'block')
    communities = {}
    for n, b in blocks.items():
        communities.setdefault(b, set()).add(n)

    # Compute weighted degrees and edge counts
    weighted_degrees = {}
    for node in G.nodes():
        weighted_degree = 0
        for neigh in G.neighbors(node):
            edge_attr = G[node][neigh]['edge_attr']
            weight = 1.0 + edge_weight * (edge_attr[0] - edge_attr[1])
            weighted_degree += weight
        weighted_degrees[node] = weighted_degree

    phis = []
    for comm in communities.values():
        # Total weighted degree inside community
        d_comm = sum(weighted_degrees[n] for n in comm)

        # Total weighted degree in complement
        comp = set(G.nodes()) - comm
        d_comp = sum(weighted_degrees[n] for n in comp)

        # Count weighted edges leaving the community
        e_out = 0
        for u in comm:
            for v in G.neighbors(u):
                if v not in comm:
                    edge_attr = G[u][v]['edge_attr']
                    weight = 1.0 + edge_weight * (edge_attr[0] - edge_attr[1])
                    e_out += weight

        # Avoid division by zero
        denom = min(d_comm, d_comp) if min(d_comm, d_comp) > 0 else 1
        phi = e_out / denom
        phis.append(phi)

    avg_phi = np.mean(phis) if phis else 1.0

    # Define label: higher is better clustering
    label = 1 - avg_phi
    # Clip to [0,1]
    label = max(0.0, min(1.0, label))
    return label
