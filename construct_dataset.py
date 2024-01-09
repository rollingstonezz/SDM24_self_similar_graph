import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.utils import train_test_split_edges
from itertools import combinations
import time
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU
import argparse
import os, sys
from tqdm import tqdm
import time
import pickle as pkl
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.utils import subgraph 
from torch.autograd import Variable
from torch_geometric.data import Data, DataLoader
import os.path as osp
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import Batch
from torch_geometric.loader import ClusterData, ClusterLoader, DataLoader
from torch_geometric.nn import BatchNorm, SAGEConv
from torch_sparse import SparseTensor, cat
import numpy as np
import os.path as osp
from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset, MoleculeNet, GEDDataset
from utils.UPFD import UPFD
from utils.GNNBenchmarkDataset import GNNBenchmarkDataset
from torch_geometric.loader import ClusterData, ClusterLoader, DataLoader, ShaDowKHopSampler
from torch_geometric.nn import BatchNorm, SAGEConv
from torch_geometric.utils import to_undirected
from torch_scatter import scatter
import copy

class MyClusterLoader(torch.utils.data.DataLoader):
    def __init__(self, cluster_data, **kwargs):
        self.cluster_data = cluster_data

        super().__init__(range(len(cluster_data)), collate_fn=self.__collate__,
                         **kwargs)

    def __collate__(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        N = self.cluster_data.data.num_nodes
        E = self.cluster_data.data.num_edges
        
        #print(batch)

        start = self.cluster_data.partptr[batch].tolist()
        end = self.cluster_data.partptr[batch + 1].tolist()
        
        node_idx = torch.cat([torch.arange(s, e) for s, e in zip(start, end)])

        data = copy.copy(self.cluster_data.data)
        del data.num_nodes
        adj, data.adj = self.cluster_data.data.adj, None
        adj = cat([adj.narrow(0, s, e - s) for s, e in zip(start, end)], dim=0)
        adj = adj.index_select(1, node_idx)
        row, col, edge_idx = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)

        for key, item in data:
            if isinstance(item, torch.Tensor) and item.size(0) == N:
                data[key] = item[node_idx]
            elif isinstance(item, torch.Tensor) and item.size(0) == E:
                data[key] = item[edge_idx]
            else:
                data[key] = item
                
        nodes_length = end[0] - start[0]
        #assert len(data.edge_index.shape) == 2
        if data.edge_index.shape[0] == 1 and data.edge_index.shape[1] == 0:
            data.edge_index = torch.tensor([[],[]], dtype=torch.long)
            data.edge_attr = torch.empty((0,data.num_edge_features))
            
        if (data.edge_index[1][data.edge_index[0] < nodes_length] >= nodes_length).any():
            return data, True
        else:
            return data, False
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--idx_dir', type=str, default='./splits')
    parser.add_argument('--save_dir', type=str, default='./processed')
    parser.add_argument('--num_clusters', type=int, default=5)
    parser.add_argument('--dataset', type=str, default='BACE')
    args = parser.parse_args()
    dataset_name = args.dataset

    DATA_PATH = args.data_dir
    idx_path = os.path.join(args.idx_dir, dataset_name ,'10fold_idx')

    if dataset_name in ['DD', 'MUTAG', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'COLLAB']:
        path = osp.join(DATA_PATH, 'TUDataset')
        dataset = TUDataset(path, name=dataset_name)
        num_splits = 10
    elif dataset_name in ['brain']:
        path = osp.join(DATA_PATH, 'brain')
        with open(os.path.join(path, 'sc.pkl'), 'rb') as f:
            dataset = pkl.load(f)
        num_splits = 10
    elif dataset_name in ['UPFD']:
        path = osp.join(DATA_PATH, 'UPFD')
        dataset_train = UPFD(path, feature='content', name='politifact', split='train')
        dataset_val = UPFD(path, feature='content', name='politifact', split='val')
        dataset_test = UPFD(path, feature='content', name='politifact', split='test')
    elif dataset_name in ['MNIST', 'CIFAR10']:
        path = osp.join(DATA_PATH, 'GNNBenchmarkDataset')
        dataset_train = GNNBenchmarkDataset(path, name=dataset_name, split='train')
        dataset_val = GNNBenchmarkDataset(path, name=dataset_name, split='val')
        dataset_test = GNNBenchmarkDataset(path, name=dataset_name, split='test')
    elif dataset_name in ['hiv', 'bace', 'bbbp']:
        path = osp.join(DATA_PATH, 'PygGraphPropPredDataset')
        dataset = PygGraphPropPredDataset(name = 'ogbg-mol'+dataset_name, root = path)   
        split_indices = dataset.get_idx_split() 
        train_indices = split_indices["train"]
        val_indices = split_indices["valid"]
        test_indices = split_indices["test"]
        dataset_train = dataset[train_indices]
        dataset_val = dataset[val_indices]
        dataset_test = dataset[test_indices]


    def transform_to_batch_cluster(dataset):
        idx = 0
        if args.dataset in ['DD', 'MUTAG', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'UPFD']:
            batch_size = len(dataset)
        else:
            batch_size = 128

        batched_cluster_node_data_list = []
        batched_cluster_edge_data_list = []
        batched_cluster_edge_index = []
        batched_batch_list = []
        batched_y_list = []
        batched_num_edge_list = []
        num_clusters = args.num_clusters

        while idx < len(dataset):
            # define for a batch
            cluster_node_data_list = []
            cluster_edge_data_list = []
            cluster_edge_index = []
            batch_list = []
            y_list = []
            num_edge_list = []
            accumulated_num = 0

            current_batch_size = batch_size if idx+batch_size < len(dataset) else len(dataset)-idx
            # in one batch
            for i in range(current_batch_size):
                data = dataset[idx]

                num_splits = data.num_nodes if data.num_nodes < num_clusters else num_clusters
                cluster_data = ClusterData(data, num_parts=num_splits, recursive=True)
                train_loader = MyClusterLoader(cluster_data, batch_size=1, shuffle=False,
                                             num_workers=1)
                # super node
                for t in range(num_splits):
                    batch = torch.tensor([t])
                    item, _ = train_loader.collate_fn(batch)
                    cluster_node_data_list.append(item)

                # super edge
                batches_directed = torch.tensor(list(combinations(range(num_splits), 2)), dtype=torch.long)
                batches = to_undirected(batches_directed.T).T
                cluster_edge_exist_list = []
                for batch in batches:
                    item, cluster_edge_exist_flag = train_loader.collate_fn(batch)
                    cluster_edge_exist_list.append(cluster_edge_exist_flag)
                    if cluster_edge_exist_flag:
                        cluster_edge_data_list.append(item)
                cluster_edge_index_item = batches[cluster_edge_exist_list]

                batch_list += [i] * num_splits
                cluster_edge_index += cluster_edge_index_item + accumulated_num
                accumulated_num += num_splits
                y_list.append(data.y)
                num_edge_list.append(len(cluster_edge_index_item))
                idx += 1
            # transform set
            batched_dataset_node_item = Batch.from_data_list(cluster_node_data_list)
            batched_dataset_edge_item = Batch.from_data_list(cluster_edge_data_list)
            clustered_edge_index_item = torch.cat(cluster_edge_index, dim=0).view(-1,2).T
            clustered_batch_item = torch.tensor(batch_list)
            y_true_item = torch.cat(y_list)

            batched_cluster_node_data_list.append(batched_dataset_node_item)
            batched_cluster_edge_data_list.append(batched_dataset_edge_item)
            batched_cluster_edge_index.append(clustered_edge_index_item)
            batched_batch_list.append(clustered_batch_item)
            batched_y_list.append(y_true_item)
            batched_num_edge_list.append(num_edge_list)
        return batched_cluster_node_data_list, batched_cluster_edge_data_list, batched_cluster_edge_index, batched_batch_list, batched_y_list, batched_num_edge_list

    if dataset_name in ['MNIST', 'CIFAR10', 'hiv', 'bace', 'bbbp', 'UPFD']:
        start_time = time.time()
        cluster_data_list_train_node, cluster_data_list_train_edge, cluster_edge_index_train, batch_list_train, y_list_train, num_edge_list_train = transform_to_batch_cluster(dataset_train)
        cluster_data_list_val_node, cluster_data_list_val_edge, cluster_edge_index_val, batch_list_val, y_list_val, num_edge_list_val = transform_to_batch_cluster(dataset_val)
        cluster_data_list_test_node, cluster_data_list_test_edge, cluster_edge_index_test, batch_list_test, y_list_test, num_edge_list_test = transform_to_batch_cluster(dataset_test)
        print("--- %s seconds ---" % (time.time() - start_time))

        store_dataset = {
            'batched_dataset_train_node': cluster_data_list_train_node,
            'batched_dataset_train_edge': cluster_data_list_train_edge,
            'clustered_edge_index_train': cluster_edge_index_train,
            'clustered_batch_train': batch_list_train,
            'y_true_train': y_list_train,
            'batched_dataset_val_node': cluster_data_list_val_node,
            'batched_dataset_val_edge': cluster_data_list_val_edge,
            'clustered_edge_index_val': cluster_edge_index_val,
            'clustered_batch_val': batch_list_val,
            'y_true_val': y_list_val,
            'batched_dataset_test_node': cluster_data_list_test_node,
            'batched_dataset_test_edge': cluster_data_list_test_edge,
            'clustered_edge_index_test': cluster_edge_index_test,
            'clustered_batch_test': batch_list_test,
            'y_true_test': y_list_test,
            'type': 'train_val_test'
        }
        path = osp.join(args.save_dir, dataset_name)
        if not os.path.exists(path):
            os.mkdir(path)
        with open(osp.join(path, 'batched_data_cluster'+str(args.num_clusters)+'.pkl'), 'wb') as f:
            pkl.dump(store_dataset, f)
    else:
        for i in range(1,num_splits+1):
            if dataset_name in ['DD', 'MUTAG', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'COLLAB']:
                with open(os.path.join(idx_path, 'train_idx-'+str(i)+'.txt'), 'r') as f:
                    train_indices = list(map(int, f.readlines()))
                with open(os.path.join(idx_path, 'test_idx-'+str(i)+'.txt'), 'r') as f:
                    test_indices = list(map(int, f.readlines()))
            elif dataset_name in ['brain', 'REDDIT-BINARY']:
                test_indices = list(np.random.choice(len(dataset), len(dataset)//10, replace=False))
                train_indices = list(set(np.arange(len(dataset)))-set(test_indices))

            start_time = time.time()
            if dataset_name in ['brain']:
                dataset_train = []
                dataset_test = []
                for t in range(len(dataset)):
                    if t in train_indices:
                        dataset_train.append(dataset[t])
                    else:
                        dataset_test.append(dataset[t])
            else:
                dataset_train = dataset[train_indices]
                dataset_test = dataset[test_indices]

            start_time = time.time()
            cluster_data_list_train_node, cluster_data_list_train_edge, cluster_edge_index_train, batch_list_train, y_list_train, num_edge_list_train = transform_to_batch_cluster(dataset[train_indices])
            cluster_data_list_test_node, cluster_data_list_test_edge, cluster_edge_index_test, batch_list_test, y_list_test, num_edge_list_test = transform_to_batch_cluster(dataset[test_indices])
            print("--- %s seconds ---" % (time.time() - start_time))

            store_dataset = {
                'batched_dataset_train_node': cluster_data_list_train_node,
                'batched_dataset_train_edge': cluster_data_list_train_edge,
                'clustered_edge_index_train': cluster_edge_index_train,
                'clustered_batch_train': batch_list_train,
                'y_true_train': y_list_train,
                'batched_dataset_test_node': cluster_data_list_test_node,
                'batched_dataset_test_edge': cluster_data_list_test_edge,
                'clustered_edge_index_test': cluster_edge_index_test,
                'clustered_batch_test': batch_list_test,
                'y_true_test': y_list_test,
                'type': '10_fold'
            }
            path = osp.join(args.save_dir, dataset_name)
            if not os.path.exists(path):
                os.mkdir(path)
            with open(osp.join(path, 'batched_data_'+str(i)+'_cluster'+str(args.num_clusters)+'.pkl'), 'wb') as f:
                pkl.dump(store_dataset, f)