import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm


class SyntheticRewiringDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SyntheticRewiringDataset, self).__init__(root, transform, pre_transform)

        # Load processed data if it exists
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """Returns the raw dataset file name."""
        return ['synthetic_rewiring_dataset.pt']  # Treat as raw data

    @property
    def processed_file_names(self):
        """Defines where the processed dataset will be stored after pre_transform."""
        return ['synthetic_rewiring_dataset_processed.pt']

    def download(self):
        """No download step since data is manually provided in the 'raw' directory."""
        pass

    def process(self):
        """Process raw data into processed form."""
        os.makedirs(self.processed_dir, exist_ok=True)

        # Load the raw data list
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data_list = torch.load(raw_path)

        # Apply pre_transform if defined
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        # Collate the data list
        data, slices = self.collate(data_list)

        # Save processed dataset
        torch.save((data, slices), self.processed_paths[0])
