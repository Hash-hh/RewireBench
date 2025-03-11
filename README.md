# RewireBench

RewireBench is a benchmark dataset for testing graph rewiring algorithms. It generates synthetic graphs with controlled properties, applies rewiring, and measures the impact through various metrics.

## Creating the Dataset

Run `main.py` to generate the benchmark dataset:

```bash
python main.py
```

The dataset will be created in the `rewire_bench/processed` directory.

## Parameters

The benchmark can be customized with the following parameters:

### Graph Generation
- `num_graphs`: Number of graphs to generate (default: 12000)
- `min_nodes`/`max_nodes`: Range of nodes per graph (default: 30-60)
- `min_clusters`/`max_clusters`: Range of clusters per graph (default: 2-6)
- `num_features`: Node feature dimensions (default: 1)
- `H`: Homophily parameter (default: 1)
- `p_intra`/`p_inter`: Intra/inter-cluster connection probabilities (default: 0.8/0.1)

### Graph Rewiring
- `p_inter_remove`: Probability to remove inter-cluster edges (default: 0.9)
- `p_intra_remove`: Probability to remove intra-cluster edges (default: 0.05)
- `p_inter_add`/`p_intra_add`: Probabilities to add inter/intra-cluster edges (default: 0.2)

## Metrics

The benchmark includes several metrics to evaluate graph properties:

### Local Easy Metrics
Simple metrics based on local neighborhood properties:
- `local_easy1`: Average difference between a node's feature and its neighbors' features
- `local_easy2`: Quadratic relationship (squared difference) between node and neighbor features
- `local_easy3`: Weighted combination of 1-hop and 2-hop neighborhood features

### Local Hard Metrics
More complex metrics involving larger neighborhoods:
- `local_hard1`: Multi-hop feature gradient between 1-hop and 3-hop neighborhoods
- `local_hard2`: Feature consistency between nodes and weighted combinations of their neighborhoods
- `local_hard3`: Feature coherence along random walk paths from each node

### Global Metrics
Structural metrics evaluating overall graph properties:
- `modularity`: Quality of community structure division
- `spectral_gap`: Second eigenvalue of the graph Laplacian (connectivity measure)
- `random_walk_stability`: How often random walks stay within the same community
- `conductance`: Measures inter-community vs. intra-community connectivity

## Using the Dataset

To use the dataset in your project:

1. Create a `raw` directory in your project
2. Copy the generated dataset file from `rewire_bench/processed/synthetic_rewiring_dataset.pt` to your `raw` directory
3. Use the dataset like any PyTorch Geometric dataset:

```python
from torch_geometric.data import InMemoryDataset
import torch

class SyntheticRewiringDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SyntheticRewiringDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['synthetic_rewiring_dataset.pt']

    @property
    def processed_file_names(self):
        return ['synthetic_rewiring_dataset_processed.pt']

    def download(self):
        pass

    def process(self):
        data_list = torch.load(self.raw_paths[0])
        if self.pre_transform:
            data_list = [self.pre_transform(d) for d in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
```
