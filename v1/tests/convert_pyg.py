import random
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset, Data
from synth_graph import generate_synthetic_graph

def convert_to_pyg(G):
    """
    Convert a networkx graph to a PyTorch Geometric Data object.
    The function 'from_networkx' can extract node attributes ('x') and
    edge attributes ('edge_attr') if we specify them.
    """
    data = from_networkx(G, group_node_attrs=['x'], group_edge_attrs=['edge_attr'])
    return data


# --- Creating a Dataset ---
class SyntheticGraphDataset(InMemoryDataset):
    def __init__(self, num_graphs, transform=None, pre_transform=None):
        self.num_graphs = num_graphs
        super(SyntheticGraphDataset, self).__init__('.', transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # No raw files are used since we generate graphs on the fly.
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Not needed as we are generating data.
        pass

    def process(self):
        data_list = []
        for _ in range(self.num_graphs):
            num_nodes = random.randint(40, 80)
            num_clusters = random.randint(3, 6)
            G = generate_synthetic_graph(num_nodes, num_clusters, H=H)
            data = convert_to_pyg(G)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
