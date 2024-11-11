
import pandas
import os.path as osp

from datetime import datetime
import pandas as pd
import torch
import numpy as np

from torch_geometric.data import HeteroData, download_url, extract_zip
from torch_geometric.transforms import RandomLinkSplit, ToUndirected


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
    def __call__(self, df):
        events =  {e for e in df.values} #CHECK THIS
        mapping = {event: i for i, event in enumerate(events)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for event in col:
                x[i, mapping[event]] = 1
        return x


class IdentityEncoder:
    # The 'IdentityEncoder' takes the raw column values and converts them to
    # PyTorch tensors.
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        #values = [int(value) for value in df.values]
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)
    
class TimeStampEncoder:
    def __init__(self):
        pass

    def __call__(self, df):
        # Extract the timestamp string from the DataFrame
        timestamp_array = [
               datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
                for timestamp in df.values
            ]
        return torch.from_numpy(np.array(timestamp_array)).view(-1, 1).to(torch.long)
        

def YOUCHOOSEWRAPPER():
    #buy_path = osp.join( "data", "YOUCHOOSE", "raw", "yoochoose-buys.dat")
    #click_path = osp.join( "data", "YOUCHOOSE", "raw", "yoochoose-clicks.dat")

    #buy_columns = ['sessionid', 'timestamp', 'itemid', 'price', 'quantity']
    #df_buy = pd.read_csv(buy_path, header=None, names=buy_columns)


    #click_columns = ['sessionid', 'timestamp', 'itemid', 'Category']
    #df_click = pd.read_csv(click_path, header=None, names=click_columns)
    #remap category to -1 in specific cases
    #df_click['Category'] = df_click['Category'].replace('S', -1)

    #train_df = pd.concat([df_click, df_buy], ignore_index=True)
    #fill nan values with -1
    #print(train_df.head())
    #train_df = train_df.fillna(0)
    #train_df.to_csv(osp.join("data", "YOUCHOOSE", "raw", "yoochoose-train.dat"), index=False, header=False)
    #del train_df

    train_columns = ['sessionid', 'timestamp', 'itemid', 'Category', 'price', 'quantity']
    train_path = osp.join("data", "YOUCHOOSE", "raw", "yoochoose-train.dat")
    test_path = osp.join("data", "YOUCHOOSE", "raw","yoochoose-test.dat")


    user_x, user_mapping = load_node_csv(train_path, index_col='sessionid', delimiter=",", names=train_columns)
    item_x , item_mapping = load_node_csv(train_path, index_col='itemid', delimiter=",", names=train_columns,
                                        encoders={'price': IdentityEncoder(dtype=torch.int64),
                                                  "Category": IdentityEncoder(dtype=torch.long)})


    edge_index, edge_label = load_edge_csv(
        train_path,
        names=train_columns,
        src_index_col='sessionid',
        src_mapping=user_mapping,
        dst_index_col='itemid',
        dst_mapping=item_mapping,
        encoders={'timestamp': TimeStampEncoder(),
                'quantity': IdentityEncoder(dtype=torch.int64)}
    )

    data = HeteroData()
    data['user'].num_nodes = len(user_mapping)  # Users do not have any features.
    data['item'].x = item_x
    data['user', 'rates', 'item'].edge_index = edge_index
    data['user', 'rates', 'item'].edge_label = edge_label
    print(data)
    return data

    # # We can now convert `data` into an appropriate format for training a
    # # graph-based machine learning model:

    # # 1. Add a reverse ('movie', 'rev_rates', 'user') relation for message passing.
    # data = ToUndirected()(data)
    # del data['item', 'rated_by', 'user'].edge_label  # Remove "reverse" label.

    # # 2. Perform a link-level split into training, validation, and test edges.
    # transform = RandomLinkSplit(
    #     num_val=0.05,
    #     num_test=0.1,
    #     neg_sampling_ratio=0.0,
    #     edge_types=[('user', 'rates', 'item')],
    #     rev_edge_types=[('item', 'rev_rates', 'user')],
    # )
    # train_data, val_data, test_data = transform(data)
    # print(train_data)
    # print(val_data)
    # print(test_data)

if __name__ == '__main__':
    YOUCHOOSEWRAPPER()