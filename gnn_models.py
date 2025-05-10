import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, GraphConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool


class GNN(torch.nn.Module):
    """
    A flexible GNN implementation that supports different layer types and configurations.
    """

    def __init__(self, config):
        super(GNN, self).__init__()

        # Extract model configuration
        self.num_features = config['model']['num_features']
        self.hidden_dim = config['model']['hidden_dim']
        self.num_layers = config['model']['num_layers']
        self.dropout = config['model']['dropout']
        self.gnn_type = config['model']['gnn_type']
        self.readout_type = config['model']['readout']
        self.batch_norm = config['model']['batch_norm']
        self.residual = config['model']['residual']

        # Initialize layers
        self.convs = nn.ModuleList()

        # Create first layer (input features to hidden)
        self.convs.append(self._create_conv_layer(self.num_features, self.hidden_dim))

        # Create intermediate layers (hidden to hidden)
        for _ in range(self.num_layers - 2):
            self.convs.append(self._create_conv_layer(self.hidden_dim, self.hidden_dim))

        # Create last layer if there's more than one layer
        if self.num_layers > 1:
            self.convs.append(self._create_conv_layer(self.hidden_dim, self.hidden_dim))

        # Batch normalization layers if enabled
        if self.batch_norm:
            self.batch_norms = nn.ModuleList()
            for _ in range(self.num_layers):
                self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))

        # Output layer for graph-level prediction
        # self.output = nn.Sequential(
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(p=self.dropout),
        #     nn.Linear(self.hidden_dim, 1)  # Binary or regression output
        # )

        self.output = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, 1) # Binary or regression output
        )

    def _create_conv_layer(self, in_dim, out_dim):
        """Create a convolutional layer based on the specified GNN type"""
        if self.gnn_type == 'GCN':
            return GCNConv(in_dim, out_dim)
        elif self.gnn_type == 'SAGE':
            return SAGEConv(in_dim, out_dim)
        elif self.gnn_type == 'GAT':
            return GATConv(in_dim, out_dim)
        elif self.gnn_type == 'GIN':
            nn_layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            return GINConv(nn_layer)
        elif self.gnn_type == 'GraphConv':
            return GraphConv(in_dim, out_dim)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")

    def forward(self, x, edge_index, batch):
        h = x  # Initial node features
        prev_h = None  # For residual connections

        for i, conv in enumerate(self.convs):
            # Store previous representation if using residual connections
            if self.residual and i > 0:
                prev_h = h

            # Apply convolution
            h = conv(h, edge_index)

            # Apply batch norm if enabled
            if self.batch_norm:
                h = self.batch_norms[i](h)

            # Apply activation
            h = F.relu(h)

            # Apply residual connection if enabled and dimensions match
            if self.residual and i > 0 and h.shape == prev_h.shape:
                h = h + prev_h

            # Apply dropout (except last layer)
            if i < len(self.convs) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Apply readout to get graph-level representation
        if self.readout_type == 'mean':
            h_graph = global_mean_pool(h, batch)
        elif self.readout_type == 'sum':
            h_graph = global_add_pool(h, batch)
        elif self.readout_type == 'max':
            h_graph = global_max_pool(h, batch)
        else:
            raise ValueError(f"Unsupported readout type: {self.readout_type}")

        # Apply output layers to get final prediction
        out = self.output(h_graph)

        return out.squeeze()  # Remove last dimension for loss calculation
