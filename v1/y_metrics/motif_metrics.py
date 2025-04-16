import numpy as np
import networkx as nx


def balanced_motif_feature_metric(G, max_hop=3):
    """
    Computes a graph-level metric that:
    1. Is learnable by standard GNNs on honest graphs.
    2. Becomes difficult to approximate on corrupted (rewired) graphs.

    This metric combines motif counts (triangles) with k-hop feature variations,
    making it both structure-aware and feature-sensitive.

    Changes:
    - Triangle component is now normalized by **edge count** instead of node count.
    - Feature component normalization is retained.
    - Added detailed comments for clarity.

    :param G: NetworkX graph with node features stored as G.nodes[n]['x'] (numpy arrays).
    :param max_hop: Maximum number of hops to consider for k-hop feature variation.
    :return: Graph-level score that is sensitive to structure and rewiring.
    """
    # Extract node features and triangle counts
    node_features = {n: G.nodes[n]['x'] for n in G.nodes()}
    triangle_counts = nx.triangles(G)  # Dictionary {node: num_triangles}

    # Initialize components
    triangle_component = 0.0  # Motif-based score
    feature_component = 0.0  # Feature variation score
    num_edges = max(1, G.number_of_edges())  # Avoid division by zero

    # Iterate through all nodes in the graph
    for node in G.nodes():
        if G.degree(node) == 0:
            continue  # Skip isolated nodes

        node_feat = node_features[node]  # Feature vector of the current node
        neighbors = list(G.neighbors(node))
        triangle_count = triangle_counts[node]  # Number of triangles node is part of

        # Track k-hop neighbors (for feature variation component)
        k_hop_nodes = set()
        current_front = {node}

        for h in range(max_hop):
            new_front = set()
            for n in current_front:
                for neigh in G.neighbors(n):
                    if neigh not in k_hop_nodes and neigh != node:
                        new_front.add(neigh)
            k_hop_nodes.update(new_front)
            current_front = new_front

        # Motif-based interaction (triangle-aware feature aggregation)
        if triangle_count > 0:
            triangle_neighbors = set()
            for n1 in neighbors:
                for n2 in neighbors:
                    if n1 != n2 and G.has_edge(n1, n2):
                        triangle_neighbors.add(n1)
                        triangle_neighbors.add(n2)

            if triangle_neighbors:
                tri_feats = np.array([node_features[n] for n in triangle_neighbors])
                tri_feat_mean = np.mean(tri_feats, axis=0)

                # Interaction term: captures how well the node aligns with its triangle context
                interaction = np.mean(node_feat * tri_feat_mean) / (1 + np.mean(np.abs(node_feat - tri_feat_mean)))

                # Scale by triangle count and apply tanh to prevent large values dominating
                triangle_component += triangle_count * np.tanh(interaction)

        # Feature gradient variation across hops
        if len(k_hop_nodes) > 1:
            hop_distances = {}
            for n in k_hop_nodes:
                try:
                    hop_distances[n] = nx.shortest_path_length(G, source=node, target=n)
                except nx.NetworkXNoPath:
                    continue

            hop_feats = {}
            for hop in range(1, max_hop + 1):
                hop_nodes = [n for n, h in hop_distances.items() if h == hop]
                if hop_nodes:
                    hop_feats[hop] = np.mean([node_features[n] for n in hop_nodes], axis=0)

            # Compute feature gradients (difference between successive hops)
            gradients = []
            hops = sorted(hop_feats.keys())
            for i in range(len(hops) - 1):
                h1, h2 = hops[i], hops[i + 1]
                grad = np.mean(np.abs(hop_feats[h1] - hop_feats[h2]))
                gradients.append(grad)

            if gradients:
                # Log-scaled factor to ensure larger triangles contribute more meaningfully
                feature_component += np.mean(gradients) * (1 + np.log(1 + triangle_count))

    # Normalize the triangle component by **number of edges**, NOT nodes
    triangle_component /= num_edges

    # Feature component remains normalized per node
    num_nodes = max(1, len(G.nodes()))  # Avoid division by zero
    feature_component /= num_nodes

    # Final combination: ensures both structure and feature sensitivity matter
    result = triangle_component * (1 + feature_component)

    return result
