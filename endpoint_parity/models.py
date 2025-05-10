import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool


class GNN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, num_layers, conv_type, readout, dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.conv_type = conv_type

        Conv = {"GCN": GCNConv, "SAGE": SAGEConv, "GIN": GINConv}[conv_type]

        for i in range(num_layers):
            in_c = in_dim if i == 0 else hid_dim
            out_c = hid_dim
            if conv_type == "GIN":
                nn = torch.nn.Sequential(
                    torch.nn.Linear(in_c, hid_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hid_dim, out_c)
                )
                self.convs.append(GINConv(nn))
            else:
                self.convs.append(Conv(in_c, out_c))

        self.readout = readout
        self.dropout = dropout

        # Enhanced classifier with non-linearities
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, hid_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hid_dim, 2)
        )

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.readout == "mean":
            x = global_mean_pool(x, batch)
        elif self.readout == "max":
            x = global_max_pool(x, batch)
        elif self.readout == "add":
            x = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown readout {self.readout}")

        return self.classifier(x)
