from endpoint_parity_dataset import EndpointParityDataset

ds = EndpointParityDataset(root='./data/EndpointParity', num_graphs=20000, L=16, M=5)
ys = [int(data.y.item()) for data in ds]
print("Count y=1:", ys.count(1))
print("Count y=0:", ys.count(0))
# Should both be 10000
