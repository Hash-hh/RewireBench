import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class SimpleGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        """
        A simple GCN-based graph regressor.
          - in_channels: size of node features (here, 6).
          - hidden_channels: number of hidden units.
          - num_layers: total layers (including input and output layers).
        """
        super(SimpleGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, 1))  # Output single value per node.

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)
        # Global pooling: average over nodes for graph-level representation.
        x = global_mean_pool(x, batch)
        # Normalize output to [0, 1]
        return torch.sigmoid(x).squeeze(-1)
