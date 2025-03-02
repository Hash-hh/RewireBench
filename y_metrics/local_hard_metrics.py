import numpy as np
import random


def compute_local_hard1(G):
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

        hop2 = set()
        for n1 in hop1:
            hop2.update(G.neighbors(n1))
        hop2 -= {node} | hop1

        hop3 = set()
        for n2 in hop2:
            hop3.update(G.neighbors(n2))
        hop3 -= {node} | hop1 | hop2

        # Skip if any neighborhood is empty
        if not hop1 or not hop3:
            continue

        # Get mean feature values for each hop
        hop1_mean = np.mean([G.nodes[n]['x'][0] for n in hop1])
        hop3_mean = np.mean([G.nodes[n]['x'][0] for n in hop3])

        # Calculate multi-hop gradient (normalized absolute difference)
        gradient = abs(hop3_mean - hop1_mean) / (abs(hop1_mean) + abs(hop3_mean) + 1e-6)
        result += gradient
        count += 1

    # Return normalized result
    return min(1.0, result / count) if count > 0 else 0.0

def compute_local_hard2(G):
    """
    Neighborhood Feature Consistency: Measures how much a node's feature aligns with
    a weighted combination of its 1-hop and 2-hop neighborhood features.
    Returns a value in [0,1].
    """
    result = 0.0
    count = 0

    for node in G.nodes():
        node_feat = G.nodes[node]['x'][0]
        hop1 = list(G.neighbors(node))
        if not hop1:
            continue

        # Get 2-hop neighborhood (excluding the original node and 1-hop neighbors)
        hop2 = set()
        for n1 in hop1:
            hop2.update(G.neighbors(n1))
        hop2 -= {node} | set(hop1)

        if not hop2:
            continue

        # Calculate weighted prediction based on 1-hop and 2-hop features
        hop1_mean = np.mean([G.nodes[n]['x'][0] for n in hop1])
        hop2_mean = np.mean([G.nodes[n]['x'][0] for n in hop2])

        # Complex weighting rule: use inverse distance-based weighting
        w1 = 0.6 + 0.2 * np.sin(node_feat)  # Makes weight depend on node feature
        w2 = 1 - w1
        pred_feat = w1 * hop1_mean + w2 * hop2_mean

        # How well does this rule predict the node's feature?
        consistency = 1.0 - min(1.0, abs(node_feat - pred_feat) / (abs(node_feat) + 1e-6))
        result += consistency
        count += 1

    return result / count if count > 0 else 0.0

def compute_local_hard3(G):
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

            # Generate random walk
            for _ in range(path_length - 1):
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break
                current = random.choice(neighbors)
                path.append(current)

            if len(path) == path_length:
                # Get features along the path
                features = [G.nodes[n]['x'][0] for n in path]

                # Calculate coherence as inverse of feature differences along path
                diffs = [abs(features[ i +1] - features[i]) for i in range(len(features ) -1)]
                # Apply non-linear transformation: high for smooth changes, low for abrupt changes
                coherence = np.exp(-2 * np.mean(diffs))
                node_coherence += coherence
                valid_paths += 1

        if valid_paths > 0:
            result += node_coherence / valid_paths
            count += 1

    return result / count if count > 0 else 0.0
