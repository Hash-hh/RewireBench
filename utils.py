from torch_geometric.utils import from_networkx


def convert_to_pyg(G):
    """
    Convert a networkx graph to a PyTorch Geometric Data object.
    The function gathers node attribute 'x' and edge attribute 'edge_attr'.
    """
    data = from_networkx(G, group_node_attrs=['x'], group_edge_attrs=['edge_attr'])
    return data
