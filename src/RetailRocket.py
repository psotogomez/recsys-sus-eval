import os.path as osp

import pandas as pd
import torch
#from sentence_transformers import SentenceTransformer

from torch_geometric.data import HeteroData, download_url, extract_zip
from torch_geometric.transforms import RandomLinkSplit, ToUndirected

# url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
# root = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/MovieLens')
# extract_zip(download_url(url, root), root)
# movie_path = osp.join(root, 'ml-latest-small', 'movies.csv')
# rating_path = osp.join(root, 'ml-latest-small', 'ratings.csv')


def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


class EventEncoder:
    def __init__(self):
        pass

    def __call__(self, df):
        events = {e for e in df.values}
        mapping = {event: i for i, event in enumerate(events)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
                x[i, mapping[col]] = 1
        return x


class IdentityEncoder:
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)


def RetailRocketWrapper(root="./data/RetailRocket/raw"):
    events_path = osp.join(root, 'events.csv')
    user_x, user_mapping = load_node_csv(events_path, index_col='visitorid')


    item_x , item_mapping = load_node_csv(events_path, index_col='itemid')


    edge_index, edge_label = load_edge_csv(
        events_path,
        src_index_col='visitorid',
        src_mapping=user_mapping,
        dst_index_col='itemid',
        dst_mapping=item_mapping,
        encoders={'event': EventEncoder(),
                'timestamp': IdentityEncoder(dtype=torch.long)},
    )

    data = HeteroData()
    data['user'].num_nodes = len(user_mapping)  # Users do not have any features.
    data['item'].num_nodes = len(item_mapping)  # Items do not have any features.
    #data['item'].x = item_x  # Items do not have any features.
    data['user', 'rates', 'item'].edge_index = edge_index
    data['user', 'rates', 'item'].edge_label = edge_label
    data["user", "rates", "item"].edge_label_index = edge_index
    print(data)
    return data

    # We can now convert `data` into an appropriate format for training a
    # graph-based machine learning model:

    # # 1. Add a reverse ('movie', 'rev_rates', 'user') relation for message passing.
    # data = ToUndirected()(data)
    # del data['movie', 'rev_rates', 'user'].edge_label  # Remove "reverse" label.

    # # 2. Perform a link-level split into training, validation, and test edges.
    # transform = RandomLinkSplit(
    #     num_val=0.05,
    #     num_test=0.1,
    #     neg_sampling_ratio=0.0,
    #     edge_types=[('user', 'rates', 'movie')],
    #     rev_edge_types=[('movie', 'rev_rates', 'user')],
    # )
    # train_data, val_data, test_data = transform(data)
    # print(train_data)
    # print(val_data)
    # print(test_data)

