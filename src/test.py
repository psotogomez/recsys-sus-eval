from torch_geometric.data import HeteroData, download_url, extract_zip
from torch_geometric.transforms import RandomLinkSplit, To
from torch_geometric.nn import LightGCN
import torch
#load model

model = LightGCN(num_nodes=1642641, embedding_dim=32, num_layers=2)

model.load_state_dict(torch.load('models/lightgcnrr.pth'))

#code a function to predict the link between a user and a product

