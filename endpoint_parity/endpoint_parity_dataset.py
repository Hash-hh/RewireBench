"""
Synthetic Endpoint-Parity graph classification dataset for PyTorch Geometric.

Each graph has:
  - A simple path of length L = 16.
  - Exactly M = 5 length‑2 branches attached at random interior nodes.
  - One‑hot node features: [1,0] for “red”, [0,1] for “blue”.
  - Label y = 1 if both path endpoints are “blue”, else y = 0.
  - Total nodes N = (L+1) + 2*M = 22.
"""

import os
import shutil
import random
from tqdm import tqdm

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
import networkx as nx


class EndpointParityDataset(InMemoryDataset):
    """
    Configurable Endpoint-Parity dataset:
      - Path length = L
      - # of length‑3 branches = M
      - Two key blues per graph + n_distractors additional blues
        * y=1 positives: both path‑endpoints blue
        * y=0 negatives: two distinct middle-branch nodes blue
      - Force regeneration with force_regen=True
    """

    def __init__(self,
                 root: str,
                 L: int,
                 M: int,
                 num_graphs: int,
                 n_distractors: int or list = 0,  # Can be single int or range [min, max]
                 force_regen: bool = True,
                 transform=None,
                 pre_transform=None):
        self.L = L
        self.M = M
        self.num_graphs = num_graphs

        # Handle n_distractors as either int or range
        if isinstance(n_distractors, list):
            if len(n_distractors) == 1:
                self.n_distractors_min = n_distractors[0]
                self.n_distractors_max = n_distractors[0]
            else:
                self.n_distractors_min = min(n_distractors)
                self.n_distractors_max = max(n_distractors)
        else:
            self.n_distractors_min = n_distractors
            self.n_distractors_max = n_distractors

        # if you want to wipe out stale files:
        if force_regen:
            shutil.rmtree(os.path.join(root, 'processed'), ignore_errors=True)

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_file_names(self):
        # Include distractor range in filename
        return [f"EP_L{self.L}_M{self.M}_D{self.n_distractors_min}-{self.n_distractors_max}_{self.num_graphs}.pt"]

    def process(self):
        # build a 50/50, interleaved label list
        n_pos = self.num_graphs // 2
        labels = [1] * n_pos + [0] * (self.num_graphs - n_pos)
        random.shuffle(labels)

        data_list = []
        for label in tqdm(labels, desc="Generating graphs"):
            G = self._make_base_graph()

            # start everyone red
            color = {v: 'red' for v in G.nodes()}

            if label == 1:
                # positive: both endpoints blue
                color[0] = 'blue'
                color[self.L] = 'blue'
            else:
                # negative: choose two distinct middle nodes from branches
                middle_nodes = []
                for node in G.nodes():
                    if G.degree(node) == 2 and node > self.L:
                        neighbors = list(G.neighbors(node))
                        if all(G.degree(n) != 1 for n in neighbors):
                            middle_nodes.append(node)

                a, b = random.sample(middle_nodes, 2)
                color[a] = 'blue'
                color[b] = 'blue'

            # Sample number of distractors for this specific graph
            this_graph_distractors = random.randint(self.n_distractors_min, self.n_distractors_max)

            # Add distractor blue nodes
            if this_graph_distractors > 0:
                # For negative examples, explicitly exclude endpoints from candidates
                if label == 0:
                    candidates = [n for n, c in color.items() if c == 'red' and n != 0 and n != self.L]
                else:
                    candidates = [n for n, c in color.items() if c == 'red']

                num_distractors = min(this_graph_distractors, len(candidates))
                if num_distractors > 0:  # Make sure we have candidates
                    distractors = random.sample(candidates, num_distractors)
                    for node in distractors:
                        color[node] = 'blue'

            # sanity checks
            blues = [n for n, c in color.items() if c == 'blue']
            assert len(blues) == (
                        2 + this_graph_distractors), f"Expected {2 + this_graph_distractors} blue nodes, got {len(blues)}"
            if label == 1:
                assert 0 in blues and self.L in blues, f"Positive graph without endpoints blue"
            else:
                assert 0 not in blues and self.L not in blues, f"Negative graph with endpoint blue"

            # assign and convert
            nx.set_node_attributes(G, color, 'color')
            data = from_networkx(G)
            mapping = {'red': [1, 0], 'blue': [0, 1]}
            data.x = torch.tensor([mapping[c] for c in data.color], dtype=torch.float)
            data.y = torch.tensor([label], dtype=torch.long)
            uid = f"{random.getrandbits(32):08x}"
            data.name = f"EP_L{self.L}_M{self.M}_D{this_graph_distractors}_{uid}"

            data_list.append(data)

        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])


    def _make_base_graph(self):
        # 1) main path 0–1–…–L
        G = nx.path_graph(self.L + 1)
        next_node = self.L + 1

        # 2) attach M length‑3 branches at interior nodes 2..L-2
        sites = list(range(2, self.L - 1))
        for _ in range(self.M):
            v = random.choice(sites)
            G.add_edge(v, next_node)  # first branch node
            G.add_edge(next_node, next_node + 1)  # middle branch node
            G.add_edge(next_node + 1, next_node + 2)  # end branch node
            next_node += 3

        return G
