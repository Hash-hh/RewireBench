import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
import os
import yaml
from tqdm import tqdm
import math


class ChainBlocksDataset(InMemoryDataset):
    """
    Dataset of chain blocks with different task mechanisms based on configuration.
    """

    def __init__(self, root, config_path, transform=None, pre_transform=None, pre_filter=None):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # Use the task name in the filename to differentiate datasets
        task_name = self.config.get('task', {}).get('type', 'default')
        return [f'chain_blocks_{task_name}.pt']

    @property
    def processed_file_names(self):
        # Use the task name in the filename to differentiate datasets
        task_name = self.config.get('task', {}).get('type', 'default')
        return [f'chain_blocks_{task_name}_processed.pt']

    def _create_base_graph(self, num_nodes_per_block, num_blocks, distance_threshold=0.5):
        """
        Creates the basic graph structure: nodes, blocks, and connections.
        All task-specific mechanisms will build on this base structure.
        """
        feature_dim = self.config['graph'].get('feature_dim', 16)

        # Node features and graph structure
        node_features = []
        edge_index = []
        block_labels = []  # Which block each node belongs to
        special_nodes = []

        # Generate blocks
        for block_idx in range(num_blocks):
            # Create nodes for this block
            block_size = num_nodes_per_block
            block_offset = block_idx * block_size

            # Generate random positions for nodes in this block
            positions = np.random.rand(block_size, 2)  # 2D positions

            # Generate random node features
            for i in range(block_size):
                node_idx = block_offset + i
                feature_vec = np.random.normal(0, 1, feature_dim)
                node_features.append(feature_vec)
                block_labels.append(block_idx)

            # Connect nodes within the block
            for i in range(block_size):
                for j in range(i + 1, block_size):
                    i_pos = positions[i]
                    j_pos = positions[j]
                    dist = np.sqrt(np.sum((i_pos - j_pos) ** 2))

                    if dist < distance_threshold:
                        edge_index.append([block_offset + i, block_offset + j])
                        edge_index.append([block_offset + j, block_offset + i])  # Undirected

            # Select special nodes according to the configured strategy
            special_node_selection = self.config.get('task', {}).get('special_node_selection', 'endpoints_only')

            if special_node_selection == 'endpoints_only' and (block_idx == 0 or block_idx == num_blocks - 1):
                # Original behavior: only first and last blocks have special nodes
                special_node = block_offset + np.random.randint(0, block_size)
                special_nodes.append(special_node)
            elif special_node_selection == 'one_per_block':
                # One special node per block
                special_node = block_offset + np.random.randint(0, block_size)
                special_nodes.append(special_node)
            elif special_node_selection == 'custom':
                # Custom selection based on block indices in config
                if block_idx in self.config['task'].get('special_block_indices', [0, num_blocks - 1]):
                    special_node = block_offset + np.random.randint(0, block_size)
                    special_nodes.append(special_node)

        # Connect adjacent blocks
        for block_idx in range(num_blocks - 1):
            block_a_offset = block_idx * num_nodes_per_block
            block_b_offset = (block_idx + 1) * num_nodes_per_block

            # Connect random nodes between adjacent blocks
            num_connections = self.config['graph'].get('inter_block_connections', 2)
            for _ in range(num_connections):
                node_a = block_a_offset + np.random.randint(0, num_nodes_per_block)
                node_b = block_b_offset + np.random.randint(0, num_nodes_per_block)

                edge_index.append([node_a, node_b])
                edge_index.append([node_b, node_a])  # Undirected

        # Create oracle edges based on the configured connection type
        oracle_connection_type = self.config.get('task', {}).get('oracle_connection_type', 'all_to_all')
        oracle_edge_index = edge_index.copy()

        if oracle_connection_type == 'all_to_all':
            # Connect all special nodes to each other (original behavior)
            for i in range(len(special_nodes)):
                for j in range(i + 1, len(special_nodes)):
                    oracle_edge_index.append([special_nodes[i], special_nodes[j]])
                    oracle_edge_index.append([special_nodes[j], special_nodes[i]])
        elif oracle_connection_type == 'sequential':
            # Connect each special node only to the next one
            for i in range(len(special_nodes) - 1):
                oracle_edge_index.append([special_nodes[i], special_nodes[i + 1]])
                oracle_edge_index.append([special_nodes[i + 1], special_nodes[i]])
        elif oracle_connection_type == 'boosted_connections':
            # Add more inter-block connections
            boost_factor = self.config['task'].get('boost_factor', 3)
            additional_connections = num_connections * boost_factor

            for block_idx in range(num_blocks - 1):
                block_a_offset = block_idx * num_nodes_per_block
                block_b_offset = (block_idx + 1) * num_nodes_per_block

                for _ in range(additional_connections):
                    node_a = block_a_offset + np.random.randint(0, num_nodes_per_block)
                    node_b = block_b_offset + np.random.randint(0, num_nodes_per_block)

                    oracle_edge_index.append([node_a, node_b])
                    oracle_edge_index.append([node_b, node_a])  # Undirected

        # Convert to tensors
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long) if edge_index else torch.zeros((2, 0),
                                                                                                           dtype=torch.long)
        oracle_edge_index = torch.tensor(np.array(oracle_edge_index).T,
                                         dtype=torch.long) if oracle_edge_index else torch.zeros((2, 0),
                                                                                                 dtype=torch.long)
        special_nodes = torch.tensor(special_nodes, dtype=torch.long)
        block_labels = torch.tensor(block_labels, dtype=torch.long)

        return x, edge_index, oracle_edge_index, special_nodes, block_labels

    def _apply_task_mechanism(self, x, special_nodes, block_labels):
        """
        Applies the selected task mechanism to calculate the target value.
        """
        task_type = self.config.get('task', {}).get('type', 'default')

        # Assign special values to special nodes
        special_values = torch.rand(len(special_nodes))

        # Mark special nodes in the feature vectors
        for i, node_idx in enumerate(special_nodes):
            # Use the last feature dimension to store the special value
            x[node_idx, -1] = special_values[i]
            # Use second-to-last feature to mark as special node
            x[node_idx, -2] = 1.0

        # Calculate target based on the task type
        if task_type == 'default':
            # Original behavior: simple sum of special values
            y = torch.sum(special_values).reshape(1)

        elif task_type == 'weighted_sequence':
            # Weight values based on their position in the sequence
            # Later nodes have exponentially higher weights
            num_blocks = len(torch.unique(block_labels))
            weights = torch.tensor([math.exp(i) for i in range(len(special_values))])
            y = torch.sum(special_values * weights).reshape(1)

        elif task_type == 'polynomial':
            # Apply a polynomial function to the sequence of values
            degree = self.config['task'].get('polynomial_degree', 3)
            coefficients = torch.tensor([1.0 / (i + 1) for i in range(degree + 1)])

            # Calculate polynomial: c0 + c1*x + c2*x^2 + ... + cn*x^n
            y = torch.tensor(0.0)
            for i, val in enumerate(special_values):
                for power, coef in enumerate(coefficients):
                    y += coef * (val ** power)
            y = y.reshape(1)

        elif task_type == 'parity':
            # Binarize special values and calculate parity (sum % 2)
            binary_values = (special_values > 0.5).float()
            y = (torch.sum(binary_values) % 2).reshape(1)

        elif task_type == 'max_min_ratio':
            # Calculate ratio between max and min values
            y = (torch.max(special_values) / (torch.min(special_values) + 1e-6)).reshape(1)

        else:
            # Default to sum if unknown task type
            y = torch.sum(special_values).reshape(1)

        # Apply final non-linearity if specified
        nonlinearity = self.config.get('task', {}).get('nonlinearity', None)
        if nonlinearity == 'tanh':
            y = torch.tanh(y)
        elif nonlinearity == 'sigmoid':
            y = torch.sigmoid(y)

        return y, special_values

    def generate_chain_block_graph(self, num_nodes_per_block, num_blocks):
        """
        Main function to generate a chain block graph with the configured task mechanism.
        """
        # Create base graph structure
        x, edge_index, oracle_edge_index, special_nodes, block_labels = self._create_base_graph(
            num_nodes_per_block, num_blocks,
            distance_threshold=self.config['graph'].get('distance_threshold', 0.5)
        )

        # Apply the selected task mechanism
        y, special_values = self._apply_task_mechanism(x, special_nodes, block_labels)

        return x, edge_index, oracle_edge_index, special_nodes, y, special_values, block_labels

    def process(self):
        # Generate data
        num_graphs = self.config['dataset'].get('num_graphs', 1000)
        num_nodes_per_block = self.config['graph'].get('nodes_per_block', 10)
        num_blocks = self.config['graph'].get('num_blocks', 5)

        data_list = []

        # Generate multiple random graphs
        for graph_idx in tqdm(range(num_graphs), desc="Generating graphs"):
            x, edge_index, oracle_edge_index, special_nodes, y, special_values, block_labels = self.generate_chain_block_graph(
                num_nodes_per_block, num_blocks
            )

            # Create a data object
            data = Data(
                x=x,
                edge_index=edge_index,
                oracle_edge_index=oracle_edge_index,
                y=y,
                special_nodes=special_nodes,
                special_values=special_values,
                block_labels=block_labels,
                graph_name=f"chain_block_{graph_idx}"
            )

            data_list.append(data)

        # Process the entire dataset
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
