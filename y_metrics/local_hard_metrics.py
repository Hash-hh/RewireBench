import numpy as np
import random


def compute_local_hard1(G, edge_weight=0.3):
    """
    Multi-Hop Feature Gradient: Measures how node features change as we move away from the node.
    Compares feature values between 1-hop and 3-hop neighborhoods.
    Returns a value in [0,1].
    """
    result = 0.0
    count = 0

    for node in G.nodes():
        # Get 1-hop, 2-hop, and 3-hop neighborhoods
        hop1 = set(G.neighbors(node))
        if not hop1:
            continue

        # Get edge-weighted hop1 features
        hop1_weighted_feats = []
        hop1_weights = []
        for n1 in hop1:
            edge_attr = G[node][n1]['edge_attr']
            # Intra-cluster edges get higher weight
            weight = 1.0 + edge_weight * (edge_attr[0] - edge_attr[1])
            hop1_weighted_feats.append(G.nodes[n1]['x'] * weight)
            hop1_weights.append(weight)

        # Get hop2 and hop3 neighborhoods
        hop2 = set()
        hop2_to_parent = {}  # Maps hop2 nodes to their hop1 parent
        for n1 in hop1:
            for n2 in G.neighbors(n1):
                if n2 != node and n2 not in hop1:
                    hop2.add(n2)
                    if n2 not in hop2_to_parent:
                        hop2_to_parent[n2] = []
                    hop2_to_parent[n2].append(n1)

        hop3 = set()
        hop3_weights = {}
        for n2 in hop2:
            for n3 in G.neighbors(n2):
                if n3 != node and n3 not in hop1 and n3 not in hop2:
                    hop3.add(n3)
                    # Propagate edge weights through the path
                    if n3 not in hop3_weights:
                        hop3_weights[n3] = 0

                    # Weight along the path: node -> n1 -> n2 -> n3
                    for n1 in hop2_to_parent[n2]:
                        # Get indices of n1 in hop1
                        idx = list(hop1).index(n1) if n1 in hop1 else -1
                        if idx >= 0:
                            path_weight = hop1_weights[idx]
                            # Edge n1 -> n2
                            path_weight *= (1.0 + edge_weight * (G[n1][n2]['edge_attr'][0] - G[n1][n2]['edge_attr'][1]))
                            # Edge n2 -> n3
                            path_weight *= (1.0 + edge_weight * (G[n2][n3]['edge_attr'][0] - G[n2][n3]['edge_attr'][1]))
                            hop3_weights[n3] += path_weight

        # Skip if any neighborhood is empty
        if not hop1 or not hop3:
            continue

        # Calculate weighted means
        if sum(hop1_weights) > 0:
            hop1_mean = np.sum(hop1_weighted_feats, axis=0) / sum(hop1_weights)
        else:
            hop1_mean = np.zeros_like(G.nodes[node]['x'])

        # Calculate hop3 weighted mean
        hop3_weighted_feats = []
        total_hop3_weight = sum(hop3_weights.values())
        if total_hop3_weight > 0:
            for n3 in hop3:
                hop3_weighted_feats.append(G.nodes[n3]['x'] * hop3_weights[n3])
            hop3_mean = np.sum(hop3_weighted_feats, axis=0) / total_hop3_weight
        else:
            # Use unweighted mean if no weight info
            hop3_mean = np.mean([G.nodes[n3]['x'] for n3 in hop3], axis=0)

        # Calculate multi-hop gradient (normalized absolute difference)
        gradient = np.mean(np.abs(hop3_mean - hop1_mean) / (np.abs(hop1_mean) + np.abs(hop3_mean) + 1e-6))
        result += gradient
        count += 1

    # Return normalized result
    return min(1.0, result / count) if count > 0 else 0.0


def compute_local_hard2(G, edge_weight=0.3):
    """
    Neighborhood Feature Consistency: Measures how much a node's feature aligns with
    a weighted combination of its 1-hop and 2-hop neighborhood features.
    Returns a value in [0,1].
    """
    result = 0.0
    count = 0

    for node in G.nodes():
        node_feat = G.nodes[node]['x']
        hop1 = list(G.neighbors(node))
        if not hop1:
            continue

        # Get 1-hop neighbors with edge weights
        hop1_weighted_feats = []
        hop1_weights = []

        for n1 in hop1:
            edge_attr = G[node][n1]['edge_attr']
            # Intra-cluster edges get higher weight
            weight = 1.0 + edge_weight * (edge_attr[0] - edge_attr[1])
            hop1_weighted_feats.append(G.nodes[n1]['x'] * weight)
            hop1_weights.append(weight)

        # Get 2-hop neighborhood with edge weights (excluding the original node and 1-hop neighbors)
        hop2 = set()
        hop2_weights = {}

        for i, n1 in enumerate(hop1):
            for n2 in G.neighbors(n1):
                if n2 != node and n2 not in hop1:
                    hop2.add(n2)

                    # Calculate weight based on path: node -> n1 -> n2
                    if n2 not in hop2_weights:
                        hop2_weights[n2] = 0

                    edge_attr1 = G[node][n1]['edge_attr']
                    edge_attr2 = G[n1][n2]['edge_attr']

                    # Higher weight for intra-cluster paths
                    path_weight = hop1_weights[i] * (1.0 + edge_weight * (edge_attr2[0] - edge_attr2[1]))
                    hop2_weights[n2] += path_weight

        if not hop2:
            continue

        # Calculate weighted mean features
        if sum(hop1_weights) > 0:
            hop1_mean = np.sum(hop1_weighted_feats, axis=0) / sum(hop1_weights)
        else:
            hop1_mean = np.zeros_like(node_feat)

        # Calculate hop2 weighted mean
        hop2_weighted_feats = []
        for n2 in hop2:
            hop2_weighted_feats.append(G.nodes[n2]['x'] * hop2_weights[n2])

        if sum(hop2_weights.values()) > 0:
            hop2_mean = np.sum(hop2_weighted_feats, axis=0) / sum(hop2_weights.values())
        else:
            hop2_mean = np.zeros_like(node_feat)

        # Complex weighting rule: use feature-dependent weights
        # Base weight depends on mean feature value
        node_feat_mean = np.mean(node_feat)
        w1 = 0.6 + 0.2 * np.sin(node_feat_mean)  # Makes weight depend on node feature
        w2 = 1 - w1
        pred_feat = w1 * hop1_mean + w2 * hop2_mean

        # How well does this rule predict the node's feature?
        consistency = 1.0 - np.mean(np.minimum(1.0, np.abs(node_feat - pred_feat) / (np.abs(node_feat) + 1e-6)))
        result += consistency
        count += 1

    return result / count if count > 0 else 0.0


def compute_local_hard3(G, edge_weight=0.3):
    """
    Feature Path Coherence: Samples random walks from each node and measures
    how smoothly/coherently features change along the walk paths.
    Returns a value in [0,1].
    """
    path_length = 4
    num_paths = 3
    result = 0.0
    count = 0

    for start_node in G.nodes():
        node_coherence = 0.0
        valid_paths = 0

        for _ in range(num_paths):
            path = [start_node]
            current = start_node
            total_edge_factor = 0

            # Generate random walk, biased by edge types
            for step in range(path_length - 1):
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break

                # Bias random walk to prefer intra-cluster edges
                edge_weights = []
                for neigh in neighbors:
                    edge_attr = G[current][neigh]['edge_attr']
                    # Higher weight for intra-cluster edges [1,0], lower for inter-cluster [0,1]
                    weight = 1.0 + edge_weight * (edge_attr[0] - edge_attr[1])
                    edge_weights.append(max(0.01, weight))  # Ensure positive weights

                # Normalize weights to probabilities
                total_weight = sum(edge_weights)
                probs = [w / total_weight for w in edge_weights] if total_weight > 0 else None

                # Select next node based on probabilities
                next_node = random.choices(neighbors, weights=probs, k=1)[0]

                # Track the edge factor for this step
                edge_attr = G[current][next_node]['edge_attr']
                edge_factor = 1.0 + edge_weight * (edge_attr[0] - edge_attr[1])
                total_edge_factor += edge_factor

                path.append(next_node)
                current = next_node

            if len(path) == path_length:
                # Get features along the path
                features = [G.nodes[n]['x'] for n in path]

                # Calculate coherence as inverse of feature differences along path
                diffs = [np.mean(np.abs(features[i + 1] - features[i])) for i in range(len(features) - 1)]

                # Apply non-linear transformation: high for smooth changes, low for abrupt changes
                # Adjust coherence based on edge types in path
                avg_edge_factor = total_edge_factor / (path_length - 1)
                coherence = np.exp(-2 * np.mean(diffs) / avg_edge_factor)

                node_coherence += coherence
                valid_paths += 1

        if valid_paths > 0:
            result += node_coherence / valid_paths
            count += 1

    return result / count if count > 0 else 0.0
