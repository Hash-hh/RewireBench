import os
import torch
from torch_geometric.datasets import ZINC
import os.path as osp
from tqdm import tqdm


def apply_transforms_to_dataset(dataset_path, batch_size=100):
    """
    Apply transforms to a dataset in batches to avoid memory issues.

    Args:
        dataset_path: Path to the corrupted dataset
        batch_size: Number of graphs to process at once
    """
    print(f"Processing dataset: {dataset_path}")

    # Process each split separately
    for split in ['train', 'val', 'test']:
        split_file = osp.join(dataset_path, 'subset', 'processed', f'{split}.pt')

        if not osp.exists(split_file):
            print(f"  Skipping {split} split: file not found")
            continue

        print(f"  Processing {split} split...")

        # Load data
        data, slices = torch.load(split_file)

        # Create backup
        backup_file = f"{split_file}.bak"
        if not osp.exists(backup_file):
            torch.save((data, slices), backup_file)

        # Process in batches
        processed_graphs = []

        # Calculate total number of graphs
        num_graphs = len(slices['x']) - 1

        # Process in batches
        for batch_start in tqdm(range(0, num_graphs, batch_size), desc=f"Processing {split}"):
            batch_end = min(batch_start + batch_size, num_graphs)

            # Extract batch of graphs
            batch_graphs = []
            for i in range(batch_start, batch_end):
                graph_data = {}
                for key in data.keys():
                    if key == 'batch':
                        continue
                    graph_data[key] = data[key][slices[key][i]:slices[key][i + 1]]

                # Convert to Data object
                from torch_geometric.data import Data
                graph = Data()
                for k, v in graph_data.items():
                    graph[k] = v

                batch_graphs.append(graph)

            # Apply transforms to batch
            for graph in batch_graphs:
                # Step 1: Validate and fix edge_index if needed
                if graph.edge_index.size(0) != 2:
                    # Reshape if possible, or create self-loops
                    if graph.edge_index.numel() > 0 and graph.edge_index.numel() % 2 == 0:
                        num_edges = graph.edge_index.numel() // 2
                        graph.edge_index = graph.edge_index.view(2, num_edges)
                    else:
                        # Create self-loops
                        num_nodes = graph.num_nodes
                        idx = torch.arange(num_nodes, device=graph.x.device)
                        graph.edge_index = torch.stack([idx, idx], dim=0)

                # Step 2: Add self-loops
                from torch_geometric.utils import add_remaining_self_loops
                graph.edge_index = add_remaining_self_loops(graph.edge_index, num_nodes=graph.num_nodes)[0]

                # Step 3: Convert to undirected
                from torch_geometric.utils import to_undirected
                graph.edge_index = to_undirected(graph.edge_index, num_nodes=graph.num_nodes)

                # Step 4: Expand dimensions
                if graph.y.ndim == 1:
                    graph.y = graph.y.unsqueeze(0)

                # Step 5: One-hot encode node features
                graph.x = torch.nn.functional.one_hot(graph.x.squeeze(), 21).float()

                processed_graphs.append(graph)

        # Use PyG's collate method to combine graphs
        dummy_dataset = ZINC('./data', subset=True, split=split)
        collated_data, collated_slices = dummy_dataset.collate(processed_graphs)

        # Save processed data
        torch.save((collated_data, collated_slices), split_file)
        print(f"  Saved {len(processed_graphs)} processed graphs for {split} split")


if __name__ == "__main__":
    # Process all corrupted datasets
    corrupted_dirs = [d for d in os.listdir("corrupted_zinc") if os.path.isdir(os.path.join("corrupted_zinc", d))]

    for dir_name in corrupted_dirs:
        dataset_path = os.path.join("corrupted_zinc", dir_name)
        apply_transforms_to_dataset(dataset_path)
