import networkx as nx
import numpy as np
import random


def compute_modularity(G):
    """
    Compute normalized modularity Q in [0,1].
    """
    blocks = nx.get_node_attributes(G, 'block')
    communities = []
    for b in set(blocks.values()):
        community = {n for n, blk in blocks.items() if blk == b}
        communities.append(community)
    Q = nx.algorithms.community.modularity(G, communities)
    return (Q + 1) / 2


def compute_spectral_gap(G):
    """
    Compute the spectral gap defined as the second-smallest eigenvalue
    of the unnormalized Laplacian, normalized by the maximum eigenvalue.
    Returns a value in [0,1] (if the graph is connected; else 0).
    """
    L = nx.laplacian_matrix(G).todense()
    eigenvalues = np.linalg.eigvals(L)
    eigenvalues = np.sort(np.real(eigenvalues))  # sort in ascending order
    if len(eigenvalues) < 2:
        return 0.0
    lam2 = eigenvalues[1]
    lam_max = eigenvalues[-1] if eigenvalues[-1] != 0 else 1.0
    return lam2 / lam_max


def compute_random_walk_stability(G, T=10, num_walks=100):
    """
    Run random walks and compute the average fraction of steps that
    a walker stays in the same community as its starting node.
    Returns a value in [0,1].
    """
    blocks = nx.get_node_attributes(G, 'block')
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return 0.0
    total_fraction = 0.0
    for _ in range(num_walks):
        start = random.choice(nodes)
        start_block = blocks[start]
        current = start
        same_count = 0
        for _ in range(T):
            neighbors = list(G.neighbors(current))
            if not neighbors:
                break
            current = random.choice(neighbors)
            if blocks[current] == start_block:
                same_count += 1
        total_fraction += same_count / T
    return total_fraction / num_walks


def compute_conductance(G):
    """
    Compute a conductance-like measure per community and average over communities.
    For each community, phi = (# edges leaving the community) / (min(total degree of community, total degree of complement)).
    Then define the conductance label as 1 - (average phi), clipped to [0,1].
    """
    blocks = nx.get_node_attributes(G, 'block')
    communities = {}
    for n, b in blocks.items():
        communities.setdefault(b, set()).add(n)

    phis = []
    for comm in communities.values():
        # Total degree inside community.
        d_comm = sum(dict(G.degree(comm)).values())
        # Total degree in complement.
        comp = set(G.nodes()) - comm
        d_comp = sum(dict(G.degree(comp)).values())
        # Count edges leaving the community.
        e_out = 0
        for u in comm:
            for v in G.neighbors(u):
                if v not in comm:
                    e_out += 1
        # Avoid division by zero.
        denom = min(d_comm, d_comp) if min(d_comm, d_comp) > 0 else 1
        phi = e_out / denom
        phis.append(phi)
    avg_phi = np.mean(phis) if phis else 1.0
    # Define label: higher is better clustering.
    label = 1 - avg_phi
    # Clip to [0,1]
    label = max(0.0, min(1.0, label))
    return label