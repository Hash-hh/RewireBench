import random
import torch
from torch_geometric.data import InMemoryDataset, Data
from graph_rewiring import modify_graph
from graph_generation import generate_synthetic_graph
from utils import convert_to_pyg


class SyntheticRewiringDataset(InMemoryDataset):
    def __init__(self, root, num_graphs=100, min_nodes=40, max_nodes=80,
                 min_clusters=3, max_clusters=6, H=0.8,
                 p_intra=0.8, p_inter=0.1,
                 p_inter_remove=0.9, p_intra_remove=0.05, p_add=0.1,
                 transform=None, pre_transform=None):
        """
        Parameters:
          root: directory where the dataset should be saved.
          num_graphs: number of graphs in the dataset.
          min_nodes, max_nodes: range for the number of nodes per graph.
          min_clusters, max_clusters: range for number of clusters per graph.
          H: homophily parameter for node features.
          p_intra, p_inter: probabilities for intra- and inter-cluster edges in the original graph.
          p_inter_remove, p_intra_remove, p_add: parameters for the rewiring process.
        """
        self.num_graphs = num_graphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.H = H
        self.p_intra = p_intra
        self.p_inter = p_inter
        self.p_inter_remove = p_inter_remove
        self.p_intra_remove = p_intra_remove
        self.p_add = p_add
        super(SyntheticRewiringDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # No raw files since we generate the dataset.
        return []

    @property
    def processed_file_names(self):
        return ['synthetic_rewiring_dataset.pt']

    def download(self):
        # No download needed.
        pass

    def process(self):
        data_list = []
        for i in range(self.num_graphs):
            # Randomly select number of nodes and clusters.
            num_nodes = random.randint(self.min_nodes, self.max_nodes)
            num_clusters = random.randint(self.min_clusters, self.max_clusters)

            # --- Generate the Original Graph ---
            G = generate_synthetic_graph(num_nodes, num_clusters, H=self.H,
                                         p_intra=self.p_intra, p_inter=self.p_inter)
            data_org = convert_to_pyg(G)

            # Save original edge index and attributes.
            # (data_org.edge_index and data_org.edge_attr are torch tensors.)
            org_edge_index = data_org.edge_index.clone()
            org_edge_attr = data_org.edge_attr.clone() if hasattr(data_org, 'edge_attr') else None

            # --- Apply Modifications (Rewiring) ---
            G_mod = modify_graph(G, p_inter_remove=self.p_inter_remove,
                                 p_intra_remove=self.p_intra_remove,
                                 p_add=self.p_add)
            data_mod = convert_to_pyg(G_mod)

            # Create final Data object.
            data = Data(x=data_mod.x,
                        edge_index=data_mod.edge_index,
                        edge_attr=data_mod.edge_attr)
            # Store the original connectivity for comparison.
            data.org_edge_index = org_edge_index
            data.org_edge_attr = org_edge_attr
            # Optionally store additional metadata.
            data.num_nodes = num_nodes
            data.num_clusters = num_clusters
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
