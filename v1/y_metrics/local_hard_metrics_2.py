import numpy as np
import random


def compute_local_hard4(G, edge_weight=0.3):
    """
    Structural Feature Alignment with improved numerical stability and performance.
    """
    result = 0.0
    count = 0

    for node in G.nodes():
        # Get direct neighbors (faster than building full subgraph)
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:  # Skip isolated nodes
            continue

        # Calculate structural properties efficiently
        degree = len(neighbors)

        # Count triangles and calculate clustering more efficiently
        triangles = 0
        actual_edges = 0
        for i, n1 in enumerate(neighbors):
            for j in range(i + 1, len(neighbors)):
                n2 = neighbors[j]
                if G.has_edge(n1, n2):
                    actual_edges += 1

                    # Calculate triangle weights with clipping for stability
                    edge1_attr = G[node][n1]['edge_attr']
                    edge2_attr = G[node][n2]['edge_attr']
                    edge3_attr = G[n1][n2]['edge_attr']

                    edge1_factor = 1.0 + edge_weight * (edge1_attr[0] - edge1_attr[1])
                    edge2_factor = 1.0 + edge_weight * (edge2_attr[0] - edge2_attr[1])
                    edge3_factor = 1.0 + edge_weight * (edge3_attr[0] - edge3_attr[1])

                    # Clip factors to prevent overflow
                    edge1_factor = max(-5, min(5, edge1_factor))
                    edge2_factor = max(-5, min(5, edge2_factor))
                    edge3_factor = max(-5, min(5, edge3_factor))

                    triangles += edge1_factor * edge2_factor * edge3_factor

        possible_edges = degree * (degree - 1) / 2
        clustering = actual_edges / possible_edges if possible_edges > 0 else 0

        # Get node feature and calculate weighted variance
        node_feat = G.nodes[node]['x']

        # Calculate neighbor feature variance more efficiently
        weighted_feats_sum = np.zeros_like(node_feat, dtype=float)
        weighted_feats_squared_sum = np.zeros_like(node_feat, dtype=float)
        total_weight = 0

        for n in neighbors:
            edge_attr = G[node][n]['edge_attr']
            weight = max(0.1, 1.0 + edge_weight * (edge_attr[0] - edge_attr[1]))  # Ensure positive

            feat = G.nodes[n]['x']
            weighted_feat = feat * weight

            weighted_feats_sum += weighted_feat
            weighted_feats_squared_sum += weighted_feat * feat
            total_weight += weight

        if total_weight > 0:
            weighted_mean = weighted_feats_sum / total_weight
            weighted_var = weighted_feats_squared_sum / total_weight - weighted_mean * weighted_mean
            weighted_var = np.clip(weighted_var, 0, 10)  # Clip to avoid negative variance
        else:
            weighted_var = np.zeros_like(node_feat)

        # Simplified alignment calculation
        struct_sig = np.array([degree, clustering, triangles])
        max_sig = np.max(struct_sig) if np.max(struct_sig) > 0 else 1
        struct_sig_norm = np.clip(struct_sig / max_sig, 0, 1)

        alignment = 0
        for i, var in enumerate(weighted_var):
            # Bounded and stable correlation
            signal = np.clip(node_feat[i], -10, 10)
            noise = np.clip(np.mean(struct_sig_norm) * var, 0, 10)
            alignment += np.exp(-min(5, abs(signal - noise))) / len(node_feat)

        result += alignment
        count += 1

    return min(1.0, result / count) if count > 0 else 0.0


def compute_local_hard5(G, edge_weight=0.3):
    """
    Feature-Weighted Motif Score with improved performance and numerical stability.
    """
    result = 0.0
    count = 0
    max_paths = 3  # Reduce number of sampled paths
    max_path_len = 3

    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            continue

        node_feat = G.nodes[node]['x']
        motif_score = 0

        # Triangle motifs - optimized calculation
        for i, n1 in enumerate(neighbors):
            n1_feat = G.nodes[n1]['x']
            edge1_attr = G[node][n1]['edge_attr']
            edge1_factor = np.clip(1.0 + (2 * edge_weight) * (edge1_attr[0] - edge1_attr[1]), -5, 5)

            for j in range(i + 1, len(neighbors)):
                n2 = neighbors[j]
                if not G.has_edge(n1, n2):
                    continue

                n2_feat = G.nodes[n2]['x']
                edge2_attr = G[node][n2]['edge_attr']
                edge3_attr = G[n1][n2]['edge_attr']

                edge2_factor = np.clip(1.0 + (2 * edge_weight) * (edge2_attr[0] - edge2_attr[1]), -5, 5)
                edge3_factor = np.clip(1.0 + (2 * edge_weight) * (edge3_attr[0] - edge3_attr[1]), -5, 5)

                # Calculate feature differences with clipping
                feat_diff1 = np.mean(np.clip(np.abs(node_feat - n1_feat), 0, 10))
                feat_diff2 = np.mean(np.clip(np.abs(node_feat - n2_feat), 0, 10))
                feat_diff3 = np.mean(np.clip(np.abs(n1_feat - n2_feat), 0, 10))

                avg_diff = (feat_diff1 + feat_diff2 + feat_diff3) / 3

                # More stable calculation
                triangle_weight = np.clip(edge1_factor * edge2_factor * edge3_factor, -10, 10)
                triangle_quality = np.exp(-min(5, avg_diff)) * min(10, triangle_weight)
                motif_score += triangle_quality

        # Simplified path sampling
        if len(neighbors) >= 2:
            path_score = 0
            for _ in range(max_paths):
                # Simple 2-hop path
                if len(neighbors) > 0:
                    n1 = random.choice(neighbors)
                    n1_neighbors = list(G.neighbors(n1))
                    filtered = [n for n in n1_neighbors if n != node and n not in neighbors]

                    if filtered:
                        n2 = random.choice(filtered)

                        edge1_attr = G[node][n1]['edge_attr']
                        edge2_attr = G[n1][n2]['edge_attr']

                        edge1_factor = np.clip(1.0 + (2 * edge_weight) * (edge1_attr[0] - edge1_attr[1]), -5, 5)
                        edge2_factor = np.clip(1.0 + (2 * edge_weight) * (edge2_attr[0] - edge2_attr[1]), -5, 5)

                        feat_diff1 = np.mean(np.clip(np.abs(G.nodes[node]['x'] - G.nodes[n1]['x']), 0, 10))
                        feat_diff2 = np.mean(np.clip(np.abs(G.nodes[n1]['x'] - G.nodes[n2]['x']), 0, 10))

                        path_consistency = np.exp(-min(5, abs(feat_diff1 - feat_diff2)))
                        path_weight = np.clip(edge1_factor * edge2_factor, -10, 10)

                        path_score += path_consistency * path_weight

            motif_score += path_score

        # Normalize score
        norm_factor = max(1, len(neighbors))
        node_score = motif_score / norm_factor
        result += min(1.0, node_score)
        count += 1

    return result / count if count > 0 else 0.0


def compute_local_hard6(G, edge_weight=0.3):
    """
    Feature-Structure Consistency with improved stability and performance.
    """
    result = 0.0
    count = 0
    num_iterations = 2  # Reduced iterations for speed

    for node in G.nodes():
        # Only use 1-hop neighborhood for speed
        local_nodes = {node} | set(G.neighbors(node))
        if len(local_nodes) <= 1:  # Skip isolated nodes
            continue

        # Store original features
        node_features = {}
        for n in local_nodes:
            node_features[n] = np.clip(G.nodes[n]['x'], -10, 10)  # Clip features

        # Initialize aggregated features
        aggregated_features = {n: node_features[n].copy() for n in node_features}

        # Simplified aggregation
        for _ in range(num_iterations):
            new_features = {}
            for n in local_nodes:
                neighbors = [neigh for neigh in G.neighbors(n) if neigh in local_nodes]
                if not neighbors:
                    new_features[n] = aggregated_features[n].copy()
                    continue

                # Simple structural properties
                n_deg = len(neighbors)

                # Start with original features
                agg_feat = aggregated_features[n].copy()
                total_weight = 1.0  # Include self weight

                # Aggregate neighbor features with safeguards
                for neigh in neighbors:
                    if neigh not in aggregated_features:
                        continue

                    edge_attr = G[n][neigh]['edge_attr']
                    edge_factor = np.clip(1.0 + (2 * edge_weight) * (edge_attr[0] - edge_attr[1]), -5, 5)

                    # Simple weighted average instead of complex transformation
                    weight = abs(edge_factor)
                    agg_feat += weight * aggregated_features[neigh]
                    total_weight += weight

                # Normalize
                if total_weight > 0:
                    new_features[n] = agg_feat / total_weight
                else:
                    new_features[n] = agg_feat

            aggregated_features = new_features

        # Measure consistency
        orig_feat = node_features[node]
        final_feat = np.clip(aggregated_features[node], -10, 10)

        # Safer consistency calculation
        feat_diff = np.clip(np.abs(final_feat - orig_feat), 0, 10)
        feat_consistency = np.mean(np.exp(-feat_diff))

        # Simpler structural consistency
        struct_consist_score = 0.5  # Default value
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 1:
            # Just calculate feature similarity between a few neighbor pairs
            sampled_neighbors = neighbors[:min(3, len(neighbors))]
            pairs = 0

            for i, n1 in enumerate(sampled_neighbors):
                for n2 in sampled_neighbors[i + 1:]:
                    if n1 not in aggregated_features or n2 not in aggregated_features:
                        continue

                    # Simple feature similarity
                    feat_sim = np.exp(-np.mean(np.clip(np.abs(
                        aggregated_features[n1] - aggregated_features[n2]), 0, 5)))
                    struct_consist_score += feat_sim
                    pairs += 1

            if pairs > 0:
                struct_consist_score /= pairs

        node_score = 0.5 * feat_consistency + 0.5 * struct_consist_score
        result += node_score
        count += 1

    return result / max(1, count)
