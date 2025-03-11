import random
import torch
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data
from graph_rewiring import modify_graph
from graph_generation import generate_synthetic_graph
from utils import convert_to_pyg
from y_metrics.compute_all_metrics import compute_all_metrics


class SyntheticRewiringDataset(InMemoryDataset):
    def __init__(self, root, num_graphs=100, min_nodes=40, max_nodes=80,
                 min_clusters=3, max_clusters=6, num_node_features=1, H=0.8,
                 p_intra=0.8, p_inter=0.1,
                 p_inter_remove=0.9, p_intra_remove=0.05,
                 p_inter_add=0.2, p_intra_add=0.2,
                 transform=None, pre_transform=None,
                 metrics_list=None):
        self.num_graphs = num_graphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.node_features = num_node_features
        self.H = H
        self.p_intra = p_intra
        self.p_inter = p_inter
        self.p_inter_remove = p_inter_remove
        self.p_intra_remove = p_intra_remove
        self.p_inter_add = p_inter_add
        self.p_intra_add = p_intra_add
        self.metrics_list = metrics_list
        super(SyntheticRewiringDataset, self).__init__(root, transform, pre_transform)
        self.data_list = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # No raw files.

    @property
    def processed_file_names(self):
        return ['synthetic_rewiring_dataset.pt']

    def download(self):
        pass  # No download needed.

    def process(self):
        data_list = []

        for i in tqdm(range(self.num_graphs), desc='Generating graphs', unit='graph'):
        # Randomly select number of nodes and clusters.
            num_nodes = random.randint(self.min_nodes, self.max_nodes)
            num_clusters = random.randint(self.min_clusters, self.max_clusters)

            # --- Generate the Original Graph ---
            G = generate_synthetic_graph(num_nodes, num_clusters, num_features=self.node_features, H=self.H,
                                         p_intra=self.p_intra, p_inter=self.p_inter)
            data_org = convert_to_pyg(G)
            # Save original connectivity.
            org_edge_index = data_org.edge_index.clone()
            org_edge_attr = data_org.edge_attr.clone() if hasattr(data_org, 'edge_attr') else None

            # Compute all metrics on the original graph.
            metrics_org = compute_all_metrics(G, metrics=self.metrics_list)

            # --- Apply Modifications (Rewiring) ---
            G_mod = modify_graph(G, p_inter_remove=self.p_inter_remove,
                                 p_intra_remove=self.p_intra_remove,
                                 p_inter_add=self.p_inter_add,
                                 p_intra_add=self.p_intra_add)
            data_mod = convert_to_pyg(G_mod)

            # Compute metrics on the rewired graph.
            metrics_rewire = compute_all_metrics(G_mod, metrics=self.metrics_list)

            # Create final Data object.
            data = Data(x=data_mod.x,
                        edge_index=data_mod.edge_index,
                        edge_attr=data_mod.edge_attr)
            # Store original connectivity.
            data.org_edge_index = org_edge_index
            data.org_edge_attr = org_edge_attr
            # Additional metadata.
            data.num_nodes = num_nodes
            data.num_clusters = num_clusters
            data.name = f'graph_{i}'
            # Store computed labels (each as a tensor of shape [4]).
            data.y = torch.tensor(metrics_org, dtype=torch.float32)
            data.y_rewire = torch.tensor(metrics_rewire, dtype=torch.float32)

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(data_list, self.processed_paths[0])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
