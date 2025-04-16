from zinc_corruptor import SimpleZINCCorruptor, CorruptedZINC
import torch
import os
import traceback
from torch_geometric.loader import DataLoader


def corrupt_zinc_dataset(corruption_levels=None):
    """
    Corrupt the ZINC dataset at multiple corruption levels.

    Args:
        corruption_levels: List of corruption levels from 0 to 1

    Returns:
        dict: Dictionary mapping corruption levels to dataset paths
    """
    if corruption_levels is None:
        corruption_levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    corruptor = SimpleZINCCorruptor(root='./data', subset=True)
    dataset_paths = {}

    for level in corruption_levels:
        try:
            # Use a smaller batch size to avoid memory issues
            path = corruptor.corrupt_dataset(corruption_level=level, output_dir="corrupted_zinc", batch_size=100)
            if path:
                dataset_paths[level] = path
                print(f"Successfully created dataset with corruption level {level}")
        except Exception as e:
            print(f"Error creating dataset with corruption level {level}: {str(e)}")
            traceback.print_exc()

    return dataset_paths


def load_and_test_datasets(dataset_paths):
    """
    Load and test the corrupted datasets.

    Args:
        dataset_paths: Dictionary mapping corruption levels to dataset paths
    """
    print("\nTesting corrupted datasets:")

    for level, path in dataset_paths.items():
        try:
            print(f"\nLoading dataset with corruption level {level}...")

            # Load the corrupted dataset
            dataset = CorruptedZINC(root=path, subset=True)

            print(f"Dataset size: {len(dataset)} graphs")

            # Test getting a single graph
            sample_data = dataset[0]
            print(f"Sample graph: {len(sample_data.x)} nodes, {sample_data.edge_index.shape[1] // 2} edges")

            # Check if original edge index was preserved
            if hasattr(sample_data, 'org_edge_index') and sample_data.org_edge_index is not None:
                print(f"Original edges: {sample_data.org_edge_index.shape[1] // 2}")

                # Calculate percentage of edges that were changed
                orig_edges = set([(u.item(), v.item()) for u, v in
                                  zip(sample_data.org_edge_index[0], sample_data.org_edge_index[1])])
                new_edges = set(
                    [(u.item(), v.item()) for u, v in zip(sample_data.edge_index[0], sample_data.edge_index[1])])

                # Count unique edges (since edge_index contains both directions)
                orig_edges_unique = set()
                new_edges_unique = set()

                for u, v in orig_edges:
                    orig_edges_unique.add((min(u, v), max(u, v)))

                for u, v in new_edges:
                    new_edges_unique.add((min(u, v), max(u, v)))

                # Count common edges (in both sets)
                common_edges = orig_edges_unique.intersection(new_edges_unique)

                # Calculate percentage of original edges that remain
                if orig_edges_unique:
                    pct_original = len(common_edges) / len(orig_edges_unique) * 100
                    pct_changed = 100 - pct_original
                    print(f"Percentage of original edges remaining: {pct_original:.1f}%")
                    print(f"Corruption rate (expected {level * 100:.1f}%): {pct_changed:.1f}%")

            # Create a dataloader with a small batch size
            loader = DataLoader(dataset, batch_size=16, shuffle=True)

            # Get a batch
            batch = next(iter(loader))
            print(f"Batch size: {batch.num_graphs} graphs")
            print(f"Batch node features shape: {batch.x.shape}")
            print(f"Batch edge index shape: {batch.edge_index.shape}")

            # Check if GNN operations work on the batch
            print("Verifying batch can be used with GNN operations... ", end="")
            x = batch.x.float()
            edge_index = batch.edge_index
            node_dim = x.size(-1)

            # Simple mock GNN operation (just matrix multiplication)
            W = torch.nn.Parameter(torch.randn(node_dim, 8))
            output = torch.matmul(x, W)
            print(f"Success! Output shape: {output.shape}")

        except Exception as e:
            print(f"Error testing dataset with corruption level {level}: {e}")
            traceback.print_exc()


def main():
    try:
        # Start with fewer corruption levels for testing
        print("Creating corrupted ZINC datasets")
        dataset_paths = corrupt_zinc_dataset([0.0])

        # Test loading and using the datasets
        load_and_test_datasets(dataset_paths)

        print("\nDone! You can now use these corrupted datasets for GNN training.")
        print("\nAvailable datasets:")
        for level, path in dataset_paths.items():
            print(f"  - Corruption level {level:.1f}: {path}")

        print("\nUsage example:")
        print("from zinc_corruptor import CorruptedZINC")
        print("dataset = CorruptedZINC(root='<dataset_path>', subset=True)")
        print("# Then use as you would use the regular ZINC dataset")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
