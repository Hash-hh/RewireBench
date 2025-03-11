from torch_geometric.utils import from_networkx
import torch


def convert_to_pyg(G):
    """
    Convert a networkx graph to a PyTorch Geometric Data object.
    The function gathers node attribute 'x', edge attribute 'edge_attr',
    and preserves node community/block IDs.
    """
    # First convert with the basic attributes
    data = from_networkx(G, group_node_attrs=['x'], group_edge_attrs=['edge_attr'])

    # Extract block/cluster IDs and add them as a separate attribute
    block_ids = [G.nodes[i]['block'] for i in range(len(G.nodes))]
    data.block = torch.tensor(block_ids, dtype=torch.long)

    return data
