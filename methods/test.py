from torch_geometric.data import HeteroData, download_url, extract_zip
from torch_geometric.transforms import RandomLinkSplit, ToUndirected

from torch_geometric.nn import LightGCN
from torch_geometric.utils import degree
import torch
import tqdm
#load model

##YOUCHOOSE

data = torch.load('data/YOUCHOOSE/data.pt')
print('Data loaded from file')
print(data)

model = LightGCN(num_nodes=1660449, embedding_dim=64, num_layers=2)

model.load_state_dict(torch.load('models/lightgcn.pth'))


num_users, num_books = data['user'].num_nodes, data['item'].num_nodes
train_edge_label_index = data.edge_label_index
def test(k=10, batch_size=2000):
    emb = model.get_embedding(data.edge_index)
    user_emb, book_emb = emb[:num_users], emb[num_users:]

    precision = recall = total_examples = 0
    for start in tqdm(range(0, num_users, batch_size)):
        end = start + batch_size
        logits = user_emb[start:end] @ book_emb.t()

        # Exclude training edges:
        mask = ((train_edge_label_index[0] >= start) &
                (train_edge_label_index[0] < end))
        logits[train_edge_label_index[0, mask] - start,
               train_edge_label_index[1, mask] - num_users] = float('-inf')

        # Computing precision and recall:
        ground_truth = torch.zeros_like(logits, dtype=torch.bool)
        mask = ((data.edge_label_index[0] >= start) &
                (data.edge_label_index[0] < end))
        ground_truth[data.edge_label_index[0, mask] - start,
                     data.edge_label_index[1, mask] - num_users] = True
        node_count = degree(data.edge_label_index[0, mask] - start,
                            num_nodes=logits.size(0))

        topk_index = logits.topk(k, dim=-1).indices
        isin_mat = ground_truth.gather(1, topk_index)

        precision += float((isin_mat.sum(dim=-1) / k).sum())
        recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
        total_examples += int((node_count > 0).sum())

    return precision / total_examples, recall / total_examples


test(10)