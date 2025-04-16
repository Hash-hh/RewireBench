import torch
from torch_geometric.utils import add_remaining_self_loops, is_undirected, to_undirected, coalesce
from torch_geometric.transforms import Compose, AddRandomWalkPE, AddLaplacianEigenvectorPE
from torch_geometric.data import Data


def get_pretransform(pretransforms):
    if pretransforms is None:
        pretransforms = []

    if pretransforms:
        pretransforms = sorted(pretransforms, key=lambda p: PRETRANSFORM_PRIORITY[type(p)], reverse=True)
        return Compose(pretransforms)
    else:
        return None


class GraphAddRemainSelfLoop:
    """Adds remaining self loops to the graph."""

    def __call__(self, graph):
        # Check if edge_index exists and is not empty
        if graph.edge_index.numel() == 0:
            # Create self loops for all nodes
            num_nodes = graph.num_nodes
            idx = torch.arange(num_nodes, device=graph.edge_index.device)
            graph.edge_index = torch.stack([idx, idx], dim=0)
            return graph

        # Handle the case where edge_attr is None
        if graph.edge_attr is None:
            edge_index = add_remaining_self_loops(
                graph.edge_index,
                num_nodes=graph.num_nodes
            )[0]  # Only take first return value
            graph.edge_index = edge_index
        else:
            edge_index, edge_attr = add_remaining_self_loops(
                graph.edge_index,
                graph.edge_attr,
                num_nodes=graph.num_nodes
            )
            graph.edge_index = edge_index
            graph.edge_attr = edge_attr

        return graph


class GraphToUndirected:
    """Converts the graph to an undirected graph."""
    def __call__(self, graph):
        if not is_undirected(graph.edge_index, graph.edge_attr, num_nodes=graph.num_nodes):  # directed
            if graph.edge_attr is not None:  # have edge features
                edge_index, edge_attr = to_undirected(graph.edge_index, graph.edge_attr, num_nodes=graph.num_nodes)
            else:  # no edge features
                edge_index = to_undirected(graph.edge_index, num_nodes=graph.num_nodes)
                edge_attr = None

        else: # already undirected (do coalesce)
            if graph.edge_attr is not None:  # have edge features
                edge_index, edge_attr = coalesce(graph.edge_index, graph.edge_attr, num_nodes=graph.num_nodes)
            else:  # no edge features
                edge_index = coalesce(graph.edge_index, num_nodes=graph.num_nodes)
                edge_attr = None

        new_graph = Data(x=graph.x, edge_index=edge_index, edge_attr=edge_attr, y=graph.y, num_nodes=graph.num_nodes)

        # Copy over other attributes
        for key, item in graph:
            if key not in ['x', 'edge_index', 'edge_attr', 'y', 'num_nodes', 'batch']:
                new_graph[key] = item

        return new_graph


class GraphExpandDim:
    def __call__(self, graph):
        if graph.y.ndim == 1:
            graph.y = graph.y[None]
        if graph.edge_attr is not None and graph.edge_attr.ndim == 1:
            graph.edge_attr = graph.edge_attr[:, None]
        return graph



class GraphAttrToOneHot:
    def __init__(self, num_node_classes, num_edge_classes):
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes

    def __call__(self, graph):
        assert graph.x.dtype == torch.long
        graph.x = torch.nn.functional.one_hot(graph.x.squeeze(), self.num_node_classes).to(torch.float)

        # if graph.edge_attr is not None:
        #     assert graph.edge_attr.dtype == torch.long
        #     graph.edge_attr = torch.nn.functional.one_hot(graph.edge_attr.squeeze(), self.num_edge_classes).to(torch.float)

        return graph


# Priority of pretransforms
PRETRANSFORM_PRIORITY = {
    GraphExpandDim: 0,  # low
    GraphAddRemainSelfLoop: 100,  # highest
    GraphToUndirected: 99,  # high
    GraphAttrToOneHot: 0,  # low
}