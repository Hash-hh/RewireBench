import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from chain_blocks_dataset import ChainBlocksDataset
from train import run_experiment
import shutil
from datetime import datetime


def visualize_graph(data, title=None, save_path=None):
    """Visualize a single graph from the dataset"""
    try:
        import networkx as nx
        from torch_geometric.utils import to_networkx

        # Convert to networkx graph
        G = to_networkx(data, to_undirected=True)

        # Get node positions using spring layout
        pos = nx.spring_layout(G, seed=42)

        plt.figure(figsize=(8, 8))

        # Get special nodes
        special_nodes = data.special_nodes.tolist()

        # Calculate node colors based on feature magnitude
        node_colors = []
        for i in range(data.x.size(0)):
            if i in special_nodes:
                node_colors.append('red')  # Special nodes in red
            else:
                node_colors.append('skyblue')  # Regular nodes in blue

        # Draw the graph
        nx.draw(G, pos, node_color=node_colors, with_labels=True,
                node_size=500, font_size=10, font_weight='bold')

        # Add title if provided
        if title:
            plt.title(title)

        plt.tight_layout()

        # Save if path is provided
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return None

        return plt
    except ImportError:
        print("Networkx or matplotlib not available for visualization")
        return None


def main():
    # Set paths
    root_dir = './data'
    config_path = 'config/config_polynomial.yaml'
    train_config_path = 'config/train_config.yaml'

    # Create the dataset if it doesn't exist
    dataset = ChainBlocksDataset(root=root_dir, config_path=config_path)

    # Print some information
    print(f"Dataset contains {len(dataset)} graphs")

    # Get the first graph
    data = dataset[0]

    print(f"\nGraph: {data.graph_name}")
    print(f"  Number of nodes: {data.x.size(0)}")
    print(f"  Number of edges (normal): {data.edge_index.size(1) // 2}")
    print(f"  Number of edges (oracle): {data.oracle_edge_index.size(1) // 2}")
    print(f"  Special nodes: {data.special_nodes.tolist()}")
    print(f"  Graph label: {data.y.item()}")

    # Ensure experiments directory exists
    os.makedirs("experiments", exist_ok=True)

    # Create a temp directory for initial sample visualizations
    temp_sample_dir = os.path.join("experiments", "temp_sample_viz")
    os.makedirs(temp_sample_dir, exist_ok=True)

    # Visualize a sample graph
    print("\nVisualizing sample graph...")
    normal_graph_path = os.path.join(temp_sample_dir, "normal_graph.png")
    visualize_graph(data,
                    title=f"Graph with normal edges\nSpecial nodes: {data.special_nodes.tolist()}",
                    save_path=normal_graph_path)

    # Create a version with oracle edges for visualization
    data_oracle = data.clone()
    data_oracle.edge_index = data.oracle_edge_index

    oracle_graph_path = os.path.join(temp_sample_dir, "oracle_graph.png")
    visualize_graph(data_oracle,
                    title=f"Graph with oracle edges (shortcuts)\nSpecial nodes: {data_oracle.special_nodes.tolist()}",
                    save_path=oracle_graph_path)

    # Ask user if they want to run training
    user_input = 'y'

    if user_input == 'y':
        print("\nStarting GNN training and evaluation...")
        results, exp_dir = run_experiment(train_config_path)

        # Copy the sample visualizations to the experiment directory
        sample_viz_dir = os.path.join(exp_dir, "sample_visualizations")
        os.makedirs(sample_viz_dir, exist_ok=True)

        shutil.copy(normal_graph_path, os.path.join(sample_viz_dir, "normal_graph.png"))
        shutil.copy(oracle_graph_path, os.path.join(sample_viz_dir, "oracle_graph.png"))

        print(f"Sample graph visualizations copied to: {sample_viz_dir}")
        print(f"\nExperiment completed! All results saved to: {exp_dir}")

        # Clean up temporary directory
        shutil.rmtree(temp_sample_dir)
    else:
        print("Skipping GNN training")

        # Keep the sample visualizations in a permanent folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_dir = os.path.join("experiments", f"{timestamp}_sample_visualizations")
        os.makedirs(sample_dir, exist_ok=True)

        shutil.copy(normal_graph_path, os.path.join(sample_dir, "normal_graph.png"))
        shutil.copy(oracle_graph_path, os.path.join(sample_dir, "oracle_graph.png"))

        print(f"Sample visualizations saved to: {sample_dir}")

        # Clean up temporary directory
        shutil.rmtree(temp_sample_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
