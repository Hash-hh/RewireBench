from usage.get_rewire_bench import SyntheticRewiringDataset

my_pre_transform = None  # Define your pre-transform function here if needed
dataset = SyntheticRewiringDataset(root="rewire_bench", pre_transform=my_pre_transform)
print("Dataset has", len(dataset), "graphs.")
