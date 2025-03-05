import numpy as np
import networkx as nx

def compute_local_easy1(G):
    """
    Local metric 1: Average difference between node feature and neighbor features.
    Returns a value in [0,1].
    """
    total = 0.0
    count = 0
    for node in G.nodes():
        if G.degree(node) == 0:
            continue

        node_feat = G.nodes[node]['x'][0]  # Using first feature dimension  TODO: Allow for multiple features
        neighbors = list(G.neighbors(node))
        neighbor_feats = [G.nodes[neigh]['x'][0] for neigh in neighbors]

        # Average absolute difference between node and neighbors
        avg_diff = np.mean(np.abs(node_feat - np.array(neighbor_feats)))
        total += avg_diff
        count += 1

    if count == 0:
        return 0.0

    # Normalize to [0,1] range - assuming feature differences typically fall within [0,5]
    normalized = total / count / 5.0
    return min(max(normalized, 0.0), 1.0)

def compute_local_easy2(G):
    """
    Local metric 2: Quadratic relationship between node feature and neighbor features.
    Returns a value in [0,1].
    """
    total = 0.0
    count = 0
    for node in G.nodes():
        if G.degree(node) == 0:
            continue

        node_feat = G.nodes[node]['x'][0]
        neighbors = list(G.neighbors(node))
        avg_neigh_feat = np.mean([G.nodes[neigh]['x'][0] for neigh in neighbors])

        # Quadratic function: normalized squared difference
        quad_diff = (node_feat - avg_neigh_feat )**2
        total += quad_diff
        count += 1

    if count == 0:
        return 0.0

    # Normalize to [0,1] range - assuming squared differences typically fall within [0,25]
    normalized = total / count / 25.0
    return min(max(normalized, 0.0), 1.0)

def compute_local_easy3(G):
    """
    Local metric 3: Weighted sum of 1-hop and 2-hop neighborhood features.
    Returns a value in [0,1].
    """
    result = 0.0
    count = 0

    for node in G.nodes():
        if G.degree(node) == 0:
            continue

        node_feat = G.nodes[node]['x'][0]

        # Get 1-hop neighbors
        neighbors = list(G.neighbors(node))
        hop1_feats = [G.nodes[neigh]['x'][0] for neigh in neighbors]

        # Get 2-hop neighbors (excluding the original node)
        hop2_nodes = set()
        for neigh in neighbors:
            hop2_nodes.update(G.neighbors(neigh))
        hop2_nodes.discard(node)
        hop2_nodes -= set(neighbors)
        hop2_feats = [G.nodes[n2]['x'][0] for n2 in hop2_nodes]

        # Calculate metric using both neighborhoods
        if hop1_feats and hop2_feats:
            avg_hop1 = np.mean(hop1_feats)
            avg_hop2 = np.mean(hop2_feats)
            # Weighted formula: |node_feat - 0.7*avg_hop1 - 0.3*avg_hop2|
            value = abs(node_feat - 0.7 *avg_hop1 - 0.3 *avg_hop2)
            result += value
            count += 1

    if count == 0:
        return 0.0

    # Normalize to [0,1] range - assuming values typically fall within [0,5]
    normalized = result / count / 5.0
    return min(max(normalized, 0.0), 1.0)
