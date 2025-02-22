import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool


class SimpleGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, edge_dim):
        super(SimpleGNN, self).__init__()
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(GINEConv(
            nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU()
            ),
            edge_dim=edge_dim
        ))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.ReLU()
                ),
                edge_dim=edge_dim
            ))

        # Output layer
        self.convs.append(GINEConv(
            nn.Sequential(
                nn.Linear(hidden_channels, 1)
            ),
            edge_dim=edge_dim
        ))

        self.norm = nn.BatchNorm1d(1)  # Normalizing the final scalar per graph


    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.org_edge_index, data.org_edge_attr, data.batch
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index, edge_attr))
        x = self.convs[-1](x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        x = self.norm(x)
        return torch.sigmoid(x)
