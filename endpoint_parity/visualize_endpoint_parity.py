import os
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def visualize_samples(dataset, num_samples_per_class, out_dir, root):
    """
    Draws `num_samples_per_class` from each label and
    saves PNGs under root/out_dir with the label in the filename.
    Main path nodes are drawn horizontally, with branches alternating up/down.
    """
    save_dir = os.path.join(root, out_dir)
    os.makedirs(save_dir, exist_ok=True)

    L = dataset.L  # path length from the dataset object
    idx_pos = [i for i, d in enumerate(dataset) if int(d.y) == 1]
    idx_neg = [i for i, d in enumerate(dataset) if int(d.y) == 0]

    for label, idx_list in [("pos", idx_pos), ("neg", idx_neg)]:
        samples = random.sample(idx_list, num_samples_per_class)
        for idx in samples:
            data = dataset[idx]
            G = nx.Graph()
            G.add_edges_from(data.edge_index.t().tolist())

            # Build positions
            pos = {}
            # 1) main chain 0..L at y=0
            for v in range(L + 1):
                pos[v] = np.array([v, 0.0])

            # 2) Find branches using BFS from each leaf node
            branch_paths = []

            # Find all leaf nodes (degree 1) that aren't endpoints of the main chain
            leaf_nodes = [n for n in G.nodes() if G.degree(n) == 1 and n > L]

            for leaf in leaf_nodes:
                # Trace path back to main chain
                path = [leaf]
                current = leaf
                while True:
                    neighbors = list(G.neighbors(current))
                    next_node = [n for n in neighbors if n not in path][0]  # Get the unvisited neighbor
                    path.append(next_node)

                    # Check if we reached the main chain
                    if next_node <= L:
                        attach_point = next_node
                        branch_paths.append((attach_point, path))
                        break
                    current = next_node

            # 3) Position branches with proper spacing - alternating up/down
            branch_by_attach = {}
            for attach, path in branch_paths:
                if attach not in branch_by_attach:
                    branch_by_attach[attach] = []
                branch_by_attach[attach].append(path)

            for attach, branches in branch_by_attach.items():
                for i, path in enumerate(branches):
                    # Alternate up (-1) and down (1) directions
                    direction = 1 if i % 2 == 0 else -1
                    # Position each node in the branch path (excluding the attach point)
                    branch_nodes = path[:-1]  # exclude the main chain node
                    for depth, node in enumerate(branch_nodes):
                        # Increase y offset with depth
                        depth_factor = (depth + 1) / len(branch_nodes)
                        pos[node] = np.array([attach, depth_factor * direction])

            # Create node-to-feature mapping
            node_features = {}
            for i, feature in enumerate(data.x):
                node_features[i] = feature.tolist()

            # Get node labels by color for the title
            blue_nodes = [i for i, feature in enumerate(data.x) if feature.tolist() == [0, 1]]
            red_nodes = [i for i, feature in enumerate(data.x) if feature.tolist() == [1, 0]]

            # Create color map that exactly matches node ids
            node_colors = {}
            for node in G.nodes():
                if node in node_features:
                    feature = node_features[node]
                    node_colors[node] = 'blue' if feature == [0, 1] else 'red'
                else:
                    node_colors[node] = 'gray'  # Fallback

            # Draw
            plt.figure(figsize=(12, 4))
            nx.draw(G, pos,
                    node_color=[node_colors[node] for node in G.nodes()],
                    with_labels=True,
                    node_size=300,
                    font_size=8,
                    edge_color='black')

            title = (f"{data.name} | y={int(data.y)}\n"
                     f"Blue ({len(blue_nodes)}): {', '.join(map(str, blue_nodes))}\n"
                     f"Red ({len(red_nodes)}): {', '.join(map(str, red_nodes))}")

            plt.title(title)
            fname = f"{data.name}_y{int(data.y)}.png"
            plt.savefig(os.path.join(save_dir, fname), bbox_inches='tight')
            plt.close()

    print(f"Saved {2 * num_samples_per_class} visualizations to {save_dir}")
