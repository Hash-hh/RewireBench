import torch
from torch_geometric.loader import DataLoader
from models.simple_gnn import SimpleGNN
from torch_geometric.data import Data


def compute_y_values(dataset, num_models=3, batch_size=16, in_channels=6, hidden_channels=16, num_layers=3):
    """
    For each graph in the dataset, run n randomly initialized GNN models (on GPU if available)
    to get a regression value (normalized to [0, 1]). The final y attribute for each graph will be a tensor
    of shape [num_models], e.g., y = [0.2, 0.6, 0.9, ...].
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create n random models.
    models = [SimpleGNN(in_channels, hidden_channels, num_layers).to(device) for _ in range(num_models)]

    # Prepare a DataLoader for the dataset.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize a list to store predictions for each graph.
    # We'll store predictions in a list of lists: one sublist per graph.
    y_preds = [[] for _ in range(len(dataset))]

    # For each model, run inference over the entire dataset.
    for model in models:
        model.eval()
        all_preds = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                # Make sure to have a 'batch' attribute for global pooling.
                if not hasattr(data, 'batch'):
                    # Create a batch vector if missing.
                    batch = torch.arange(data.num_graphs, device=data.x.device).repeat_interleave(
                        data.ptr[1:] - data.ptr[:-1])
                    data.batch = batch
                preds = model(data)  # shape: [batch_size]
                all_preds.append(preds.cpu())
        all_preds = torch.cat(all_preds, dim=0)  # Shape: (num_graphs,)
        # Append predictions for each graph.
        for i, pred in enumerate(all_preds):
            y_preds[i].append(pred.item())

    # Create a new list of Data objects with updated y values
    new_data_list = []
    for i, data in enumerate(dataset):
        # Clone the existing data object
        new_data = Data(**{k: v for k, v in data})
        # Add the predictions
        new_data.y = torch.tensor(y_preds[i], dtype=torch.float32)
        new_data_list.append(new_data)

    # Create new dataset with updated values
    dataset_type = type(dataset)
    new_dataset = dataset_type(root=dataset.root)
    new_dataset.data, new_dataset.slices = dataset_type.collate(new_data_list)

    return new_dataset
