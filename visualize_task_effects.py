import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from chain_blocks_dataset import ChainBlocksDataset
import yaml
import argparse
import networkx as nx
from torch_geometric.utils import to_networkx

def visualize_graph_with_values(data, title=None, save_path=None):
    """Visualize a graph with special nodes highlighted and their values shown"""
    try:
        # Convert to networkx graph
        G = to_networkx(data, to_undirected=True)
        
        # Get node positions using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        plt.figure(figsize=(10, 8))
        
        # Get special nodes
        special_nodes = data.special_nodes.tolist()
        special_values = data.special_values.tolist()
        
        # Calculate node colors based on feature magnitude
        node_colors = []
        node_sizes = []
        for i in range(data.x.size(0)):
            if i in special_nodes:
                node_colors.append('red')  # Special nodes in red
                node_sizes.append(800)
            else:
                node_colors.append('skyblue')  # Regular nodes in blue
                node_sizes.append(300)
        
        # Draw the graph
        nx.draw(G, pos, node_color=node_colors, with_labels=True, 
                node_size=node_sizes, font_size=8, font_weight='bold')
        
        # Add value labels for special nodes
        special_labels = {}
        for i, node_idx in enumerate(special_nodes):
            val = special_values[i]
            special_labels[node_idx] = f"{node_idx}\n({val:.2f})"
        
        nx.draw_networkx_labels(G, pos, labels=special_labels, font_size=10, font_weight='bold')
        
        # Add target value to the title
        if title:
            plt.title(f"{title}\nTarget value: {data.y.item():.4f}")
        else:
            plt.title(f"Target value: {data.y.item():.4f}")
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
            return None
        
        return plt
    except ImportError:
        print("Networkx or matplotlib not available for visualization")
        return None

def main():
    parser = argparse.ArgumentParser(description="Visualize graphs from different task types")
    parser.add_argument('--config-dir', default='./config', help='Directory containing config files')
    parser.add_argument('--output-dir', default='./task_visualizations', help='Directory to save visualizations')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all config files
    config_files = [f for f in os.listdir(args.config_dir) if f.startswith('config_') and f.endswith('.yaml')]

    for config_file in config_files:
        config_path = os.path.join(args.config_dir, config_file)
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        task_type = config.get('task', {}).get('type', 'default')
        print(f"Processing task type: {task_type}")
        
        # Set fixed number of blocks for visualization
        config['graph']['num_blocks'] = 5
        
        # Create a temporary config file with fixed number of blocks
        temp_config_path = os.path.join(args.output_dir, f"temp_{config_file}")
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Create the dataset
        dataset = ChainBlocksDataset(root=os.path.join(args.output_dir, f"data_{task_type}"), 
                                    config_path=temp_config_path)
        
        # Visualize a few graphs
        task_output_dir = os.path.join(args.output_dir, task_type)
        os.makedirs(task_output_dir, exist_ok=True)
        
        for i in range(min(5, len(dataset))):
            data = dataset[i]
            
            # Visualize normal edges
            normal_path = os.path.join(task_output_dir, f"graph_{i}_normal.png")
            visualize_graph_with_values(data, 
                                      title=f"Task: {task_type} - Graph {i} (Normal Edges)",
                                      save_path=normal_path)
            
            # Create a version with oracle edges for visualization
            data_oracle = data.clone()
            data_oracle.edge_index = data.oracle_edge_index
            
            oracle_path = os.path.join(task_output_dir, f"graph_{i}_oracle.png")
            visualize_graph_with_values(data_oracle,
                                      title=f"Task: {task_type} - Graph {i} (Oracle Edges)",
                                      save_path=oracle_path)
            
        print(f"Saved visualizations for {task_type} to {task_output_dir}")
        
        # Clean up temporary config
        os.remove(temp_config_path)

if __name__ == "__main__":
    main()