import numpy as np
import networkx as nx


def compute_local_easy1(G, edge_weight=0.3):
    """
    Homophily
    Local metric 1: Average difference between node feature and neighbor features,
    weighted by edge features.
    Returns a value in [0,1].

    Parameters:
        G: networkx graph
        edge_weight: weight given to edge features (0-1)
    """
    total = 0.0
    count = 0

    for node in G.nodes():
        if G.degree(node) == 0:
            continue

        node_feat = G.nodes[node]['x']
        neighbors = list(G.neighbors(node))

        # Calculate feature-wise differences for each neighbor
        diffs = []
        for neigh in neighbors:
            neigh_feat = G.nodes[neigh]['x']
            # Feature-wise absolute difference
            feat_diff = np.abs(node_feat - neigh_feat)

            # Get edge feature and incorporate it
            edge_attr = G[node][neigh]['edge_attr']
            # Intra-cluster edges (with [1,0]) should indicate lower difference
            # Inter-cluster edges (with [0,1]) should indicate higher difference
            edge_factor = 1.0 + edge_weight * (edge_attr[1] - edge_attr[0])

            # Apply edge factor to feature difference
            weighted_diff = np.mean(feat_diff) * edge_factor
            diffs.append(weighted_diff)

        # Average difference across all neighbors
        if diffs:
            avg_diff = np.mean(diffs)
            total += avg_diff
            count += 1

    if count == 0:
        return 0.0

    # Normalize to [0,1] range
    normalized = total / count / 5.0
    return min(max(normalized, 0.0), 1.0)


def compute_local_easy2(G, edge_weight=0.3):
    """
    Homophily squared
    Local metric 2: Quadratic relationship between node feature and neighbor features,
    weighted by edge features.
    Returns a value in [0,1].
    """
    total = 0.0
    count = 0

    for node in G.nodes():
        if G.degree(node) == 0:
            continue

        node_feat = G.nodes[node]['x']
        neighbors = list(G.neighbors(node))

        # Calculate weighted average neighbor features
        weighted_neigh_feats = []
        total_weight = 0

        for neigh in neighbors:
            neigh_feat = G.nodes[neigh]['x']
            edge_attr = G[node][neigh]['edge_attr']

            # Intra-cluster edges (with [1,0]) get higher weight
            # Inter-cluster edges (with [0,1]) get lower weight
            weight = 1.0 + edge_weight * (edge_attr[0] - edge_attr[1])

            weighted_neigh_feats.append(neigh_feat * weight)
            total_weight += weight

        # Compute weighted average
        if total_weight > 0:
            avg_neigh_feat = np.sum(weighted_neigh_feats, axis=0) / total_weight
        else:
            avg_neigh_feat = node_feat

        # Quadratic function: normalized squared difference
        quad_diff = np.mean((node_feat - avg_neigh_feat) ** 2)
        total += quad_diff
        count += 1

    if count == 0:
        return 0.0

    # Normalize to [0,1] range
    normalized = total / count / 25.0
    return min(max(normalized, 0.0), 1.0)


def compute_local_easy3(G, edge_weight=0.3):
    """
    Local metric 3: Weighted sum of 1-hop and 2-hop neighborhood features,
    with edge feature weighting.
    Returns a value in [0,1].
    """
    result = 0.0
    count = 0

    for node in G.nodes():
        if G.degree(node) == 0:
            continue

        node_feat = G.nodes[node]['x']

        # Get 1-hop neighbors with edge weights
        neighbors = list(G.neighbors(node))
        hop1_weighted_feats = []
        hop1_weights = []

        for neigh in neighbors:
            neigh_feat = G.nodes[neigh]['x']
            edge_attr = G[node][neigh]['edge_attr']

            # Intra-cluster edges get higher weight
            weight = 1.0 + edge_weight * (edge_attr[0] - edge_attr[1])
            hop1_weighted_feats.append(neigh_feat * weight)
            hop1_weights.append(weight)

        # Get 2-hop neighbors with derived weights
        hop2_nodes = set()
        hop2_weights = {}

        for i, neigh in enumerate(neighbors):
            for hop2 in G.neighbors(neigh):
                if hop2 != node and hop2 not in neighbors:
                    edge_attr1 = G[node][neigh]['edge_attr']
                    edge_attr2 = G[neigh][hop2]['edge_attr']

                    # Propagate edge weights (product of edge features along path)
                    if hop2 not in hop2_weights:
                        hop2_weights[hop2] = 0

                    # Higher weight for intra-cluster paths
                    path_weight = hop1_weights[i] * (1.0 + edge_weight * (edge_attr2[0] - edge_attr2[1]))
                    hop2_weights[hop2] += path_weight
                    hop2_nodes.add(hop2)

        # If we have both 1-hop and 2-hop neighbors, calculate the metric
        if len(neighbors) > 0 and len(hop2_nodes) > 0:
            # Calculate weighted average for 1-hop
            if sum(hop1_weights) > 0:
                avg_hop1 = np.sum(hop1_weighted_feats, axis=0) / sum(hop1_weights)
            else:
                avg_hop1 = np.zeros_like(node_feat)

            # Calculate weighted average for 2-hop
            hop2_weighted_feats = []
            for hop2 in hop2_nodes:
                hop2_weighted_feats.append(G.nodes[hop2]['x'] * hop2_weights[hop2])

            if sum(hop2_weights.values()) > 0:
                avg_hop2 = np.sum(hop2_weighted_feats, axis=0) / sum(hop2_weights.values())
            else:
                avg_hop2 = np.zeros_like(node_feat)

            # Weighted formula with edge-aware weighting
            weighted_combo = 0.7 * avg_hop1 + 0.3 * avg_hop2
            diff = np.mean(np.abs(node_feat - weighted_combo))

            result += diff
            count += 1

    if count == 0:
        return 0.0

    # Normalize to [0,1] range
    normalized = result / count / 5.0
    return min(max(normalized, 0.0), 1.0)
