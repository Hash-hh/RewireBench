import os
import torch
import numpy as np
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
import torch_geometric.utils as utils
import random
from tqdm import tqdm
import copy
import os.path as osp
import gc  # Garbage collector
import traceback


class SimpleZINCCorruptor:
    """
    A simple class to load the ZINC dataset and corrupt it by replacing edges
    while maintaining connectivity and overall graph structure.
    Uses PyTorch Geometric operations directly without NetworkX.
    """

    def __init__(self, root='./data', subset=True):
        """
        Initialize the SimpleZINCCorruptor.

        Args:
            root (str): Root directory where the dataset should be stored
            subset (bool): If True, use only a subset of the dataset
        """
        self.root = root
        self.subset = subset

        # Load ZINC dataset
        print("Loading ZINC dataset...")
        # pre_transform = get_pretransform(pretransforms=[GraphAddRemainSelfLoop(),
        #                                                       GraphToUndirected(),
        #                                                       GraphExpandDim(),
        #                                                       GraphAttrToOneHot(21,
        #                                                                         1)]
        #                                  )
        pre_transform = None
        self.dataset = ZINC(self.root, subset=self.subset, split='train', pre_transform=pre_transform)

        print(f"Loaded ZINC dataset with {len(self.dataset)} graphs")

        # Print some information about the first graph
        data = self.dataset[0]
        print(f"Sample graph: {len(data.x)} nodes, {data.edge_index.shape[1] // 2} edges")
        print(f"Node feature dimensions: {data.x.shape}")
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            print(f"Edge feature dimensions: {data.edge_attr.shape}")

    def corrupt_dataset(self, corruption_level=0.5, output_dir="corrupted_zinc", batch_size=100):
        if corruption_level < 0 or corruption_level > 1:
            raise ValueError("Corruption level must be between 0 and 1")

        print(f"Corrupting ZINC dataset with corruption level {corruption_level}...")

        # Create output directory
        save_path = osp.join(output_dir, f"zinc_{'subset' if self.subset else 'full'}_corruption{corruption_level:.2f}")
        os.makedirs(save_path, exist_ok=True)

        subset_dir = osp.join(save_path, 'subset' if self.subset else 'full')
        os.makedirs(subset_dir, exist_ok=True)
        processed_dir = osp.join(subset_dir, 'processed')
        os.makedirs(processed_dir, exist_ok=True)

        splits = ['train', 'val', 'test']

        for split in splits:
            print(f"Processing {split} split...")

            try:
                original_split = ZINC(self.root, subset=self.subset, split=split)
            except Exception as e:
                print(f"Error loading {split} split: {str(e)}. Skipping.")
                continue

            loader = DataLoader(original_split, batch_size=batch_size, shuffle=False)
            data_list = []

            for batch in tqdm(loader, desc=f"Processing {split} batch"):
                batch = batch.to('cpu')

                for data in batch.to_data_list():
                    data.org_edge_index = data.edge_index.clone()

                    # Remove edge attributes
                    data.edge_attr = None

                    if corruption_level == 0:
                        corrupted_data = data  # Just store without corruption
                    else:
                        corrupted_data = self._corrupt_graph_pyg(data, corruption_level)

                    data_list.append(corrupted_data)

                # Periodically collect garbage
                if len(data_list) % 1000 == 0:
                    gc.collect()

            # Save in batched format
            torch.save(original_split.collate(data_list), osp.join(processed_dir, f'{split}.pt'))

            print(f"Saved {len(data_list)} graphs for {split} split")

            # Clear memory after saving
            data_list = None
            gc.collect()

        # Copy metadata files
        original_processed_dir = osp.join(self.root, 'subset' if self.subset else 'full', 'processed')
        for fname in ['pre_transform.pt', 'pre_filter.pt']:
            src_file = osp.join(original_processed_dir, fname)
            if osp.exists(src_file):
                torch.save(torch.load(src_file), osp.join(processed_dir, fname))

        metadata = {'corruption_level': corruption_level}
        with open(osp.join(save_path, 'metadata.txt'), 'w') as f:
            for k, v in metadata.items():
                f.write(f"{k}: {v}\n")

        print(f"Corrupted dataset saved to {save_path}")
        return save_path


    def _corrupt_graph_pyg(self, data, corruption_level):
        """
        Creates corrupted graphs with well-balanced degree distribution.
        Args:
            data: PyG Data object
            corruption_level: Level of corruption from 0 to 1

        Returns:
            PyG Data object with corrupted edges
        """
        # Create a copy to modify
        corrupted_data = copy.deepcopy(data)

        # Get number of nodes and edges
        num_nodes = corrupted_data.x.size(0)
        edge_index = corrupted_data.edge_index
        num_edges = edge_index.size(1)

        # Verify we're preserving all node features (should be 21 for ZINC)
        # assert corrupted_data.x.size(1) == 21, f"Expected 21 node features, got {corrupted_data.x.size(1)}"

        # If only a few nodes, hard to balance
        if num_nodes <= 4:
            return corrupted_data

        # # Convert to undirected edge representation
        # edge_index_undirected = utils.to_undirected(edge_index, num_nodes=num_nodes)
        # edge_index_undirected = torch.unique(torch.sort(edge_index_undirected, dim=0)[0], dim=1)
        # num_edges = edge_index_undirected.size(1)

        # If corruption level is 0, return the original graph; we don't even come here
        if corruption_level == 0:
            return corrupted_data

        # Define target degrees based on chemical graph characteristics
        # Aim for most nodes to have degrees between 2-4
        target_avg_degree = min(4.0, 2.0 + 2.0 * corruption_level)
        target_edges = int(target_avg_degree * num_edges / 2)  # TODO: Try different values for target_edges

        # Initialize with a spanning tree where degrees are already balanced
        new_edges = set()
        visited = [False] * num_nodes
        new_degrees = {i: 0 for i in range(num_nodes)}

        # Start from a random node
        start_node = random.randint(0, num_nodes - 1)
        visited[start_node] = True
        frontier = [start_node]

        # Build a balanced spanning tree (limit maximum degree to 3 during construction)
        while sum(visited) < num_nodes:
            # Choose a node from the frontier with lowest degree
            frontier.sort(key=lambda x: new_degrees[x])
            current = frontier[0]

            # Find an unvisited neighbor
            unvisited = [n for n in range(num_nodes) if not visited[n]]
            if not unvisited:
                break

            # Select a random unvisited node
            next_node = random.choice(unvisited)

            # Add edge
            edge = (min(current, next_node), max(current, next_node))
            new_edges.add(edge)
            new_degrees[current] += 1
            new_degrees[next_node] += 1

            # Update visited and frontier
            visited[next_node] = True
            if new_degrees[current] < 3:  # Keep in frontier if degree < 3
                frontier.append(next_node)

            # If current node degree is now 3, remove it from frontier
            if new_degrees[current] >= 3:
                frontier.remove(current)

        # Calculate how many more edges we need
        remaining_edges = target_edges - len(new_edges)

        # Focus on increasing the degree of nodes with degree < 3
        for _ in range(remaining_edges):
            # Get nodes sorted by degree (lowest first)
            low_degree_nodes = sorted(
                [n for n in range(num_nodes) if new_degrees[n] < 3],
                key=lambda x: new_degrees[x]
            )

            if not low_degree_nodes:
                # If no low degree nodes left, break
                break

            source = low_degree_nodes[0]

            # Find nodes that aren't already connected to this node
            candidates = []
            for target in range(num_nodes):
                if target != source and (min(source, target), max(source, target)) not in new_edges:
                    # Prefer nodes with degree < 4 (not too high)
                    if new_degrees[target] < 4:
                        score = 1.0 / (new_degrees[target] + 1)
                        candidates.append((target, score))

            if candidates:
                # Sort candidates by score (highest first)
                candidates.sort(key=lambda x: x[1], reverse=True)
                # Randomize a bit among the top candidates
                top_candidates = candidates[:min(3, len(candidates))]
                target, _ = random.choice(top_candidates)

                # Add the edge
                edge = (min(source, target), max(source, target))
                new_edges.add(edge)
                new_degrees[source] += 1
                new_degrees[target] += 1
            else:
                # If no good candidates, try any node
                for target in range(num_nodes):
                    if target != source and (min(source, target), max(source, target)) not in new_edges:
                        edge = (min(source, target), max(source, target))
                        new_edges.add(edge)
                        new_degrees[source] += 1
                        new_degrees[target] += 1
                        break

        # Special case: ensure no degree-1 nodes remain
        degree_1_nodes = [node for node, degree in new_degrees.items() if degree == 1]
        for node in degree_1_nodes:
            # Find candidates that aren't already connected to this node
            candidates = []
            for other in range(num_nodes):
                if other != node and (min(node, other), max(node, other)) not in new_edges:
                    # Prefer nodes with lower degrees
                    candidates.append((other, new_degrees[other]))

            if candidates:
                candidates.sort(key=lambda x: x[1])  # Sort by degree (lowest first)
                other = candidates[0][0]

                edge = (min(node, other), max(node, other))
                new_edges.add(edge)
                new_degrees[node] += 1
                new_degrees[other] += 1

        # Convert to PyG format (adding both directions for undirected edges)
        final_edges = []
        for u, v in new_edges:
            final_edges.extend([(u, v), (v, u)])

        # Create the new edge index
        new_edge_index = torch.tensor(final_edges, dtype=torch.long).t()

        # Verify connectivity
        def is_connected(edge_idx):
            row, col = edge_idx
            visited = torch.zeros(num_nodes, dtype=torch.bool)
            stack = [0]
            visited[0] = True

            while stack:
                node = stack.pop()
                neighbors = col[row == node]
                for n in neighbors:
                    n = n.item()
                    if not visited[n]:
                        visited[n] = True
                        stack.append(n)

            return torch.all(visited).item()

        # Final connectivity check
        if not is_connected(new_edge_index):
            print("Warning: Connectivity check failed, reverting to original edges")
            return corrupted_data

        # Ensure we're preserving all node features in the corrupted graph
        corrupted_data.edge_index = new_edge_index

        # # Debug: Check node feature dimensions before returning
        # if corrupted_data.x.size(1) != 21:
        #     print(f"Warning: Node feature dimension mismatch. Expected 21, got {corrupted_data.x.size(1)}")

        return corrupted_data


class CorruptedZINC(ZINC):
    """
    Load a corrupted ZINC dataset that was previously saved.
    Follows the same interface as the original ZINC dataset.
    """

    def __init__(self, root, subset=True, split='train', transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.subset = subset
        self.split = split

        super().__init__(root, subset, split, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        # Return only the current split file
        return [f'{self.split}.pt']

    def download(self):
        # No download needed
        pass

    def process(self):
        # No processing needed, data is already processed
        pass

