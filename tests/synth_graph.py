import networkx as nx
import numpy as np
import random

def generate_synthetic_graph(num_nodes, num_clusters, H=0.8, D=6):
    """
    Generate a graph using a stochastic block model.

    Parameters:
      num_nodes: total number of nodes (e.g., between 40 and 80)
      num_clusters: number of clusters (e.g., 3 to 6)
      H: homophily parameter, where H=1 means all nodes in a cluster get the same feature.
         Values less than 1 allow for some randomization.
    """
    # Distribute nodes approximately equally across clusters.
    sizes = [num_nodes // num_clusters] * num_clusters
    remainder = num_nodes % num_clusters
    for i in range(remainder):
        sizes[i] += 1

    # Define a probability matrix for intra-cluster and inter-cluster connections.
    p_intra = 0.8  # High probability within clusters.
    p_inter = 0.5  # Low probability between clusters.
    probs = [[p_intra if i == j else p_inter for j in range(num_clusters)] for i in range(num_clusters)]

    # Generate the graph.
    seed = random.randint(0, 10000)
    G = nx.stochastic_block_model(sizes, probs, seed=seed)

    # --- Assign Node Features ---
    # We want a D-dim one-hot feature vector that is influenced by the node's cluster,
    # but not trivially equal to the cluster id.
    for node in G.nodes():
        # networkx's stochastic_block_model sets a 'block' attribute for the node.
        block = G.nodes[node]['block']
        # Choose a "preferred" feature index based on the block (wrap it into [0,5]).
        preferred_feature = block % D
        feat = np.zeros(D, dtype=np.float32)
        # With probability H, assign the preferred feature.
        if random.random() < H:
            feat[preferred_feature] = 1.0
        else:
            # Otherwise, assign a random feature.
            rand_feature = random.randint(0, D-1)
            feat[rand_feature] = 1.0
        # Save the feature into the node attribute.
        G.nodes[node]['x'] = feat

    # --- Assign Edge Features ---
    # We create a 4D one-hot vector for each edge. For example:
    #   [1, 0, 0, 0] → intra-cluster edge
    #   [0, 1, 0, 0] → inter-cluster edge
    for u, v in G.edges():
        block_u = G.nodes[u]['block']
        block_v = G.nodes[v]['block']
        edge_feat = np.zeros(4, dtype=np.float32)
        if block_u == block_v:
            edge_feat[0] = 1.0  # Intra-cluster
        else:
            edge_feat[1] = 1.0  # Inter-cluster
        G[u][v]['edge_attr'] = edge_feat

    return G
