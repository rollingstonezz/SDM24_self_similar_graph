import torch
from torch_geometric.nn import GCNConv, GINConv, GraphConv
from torch_geometric.utils import train_test_split_edges
from itertools import combinations
import time
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU
import argparse
import os, sys
import pickle as pkl
from tqdm import tqdm
from ogb.graphproppred import PygGraphPropPredDataset
from torch.autograd import Variable
from torch_geometric.data import Data, DataLoader
from networkx.algorithms.components import strongly_connected_components
import numpy as np
import os.path as osp
from sklearn.metrics import f1_score
from torch_geometric.data import Batch
from torch_geometric.datasets import PPI
from torch_geometric.loader import ClusterData, ClusterLoader, DataLoader
from torch_geometric.nn import BatchNorm, SAGEConv
import os.path as osp
from sklearn.metrics import f1_score
import pandas as pd
from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset, MoleculeNet, GEDDataset
from utils.GNNBenchmarkDataset import GNNBenchmarkDataset
from utils.UPFD import UPFD
from torch_geometric.loader import ClusterData, ClusterLoader, DataLoader, ShaDowKHopSampler
from torch_geometric.nn import BatchNorm, SAGEConv
from torch_geometric.utils import contains_isolated_nodes
from torch_sparse import SparseTensor, cat
from torch_scatter import scatter
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from models.model import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./data/processed')
    parser.add_argument('--write_dir', type=str, default='./results')
    parser.add_argument('--dataset', type=str, default='BACE')
    parser.add_argument('--idx', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_clusters', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--number_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()
    dataset_name = args.dataset
    DATA_PATH = args.data_dir
    idx = args.idx

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
    if idx>1 and not dataset_name in ['DD', 'MUTAG', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI','COLLAB', 'brain', 'REDDIT-BINARY']:
        raise ValueError(dataset_name + 'does not have 10-fold validation.')
        exit()
        
    if dataset_name in ['MNIST', 'CIFAR10', 'hiv', 'bace', 'bbbp', 'UPFD']:

        path = osp.join(args.save_dir, dataset_name)

        assert os.path.exists(path)

        with open(osp.join(path, 'batched_data_cluster'+str(args.num_clusters)+'.pkl'), 'rb') as f:
            store_dataset = pkl.load(f)
        
        batched_dataset_train_node = store_dataset['batched_dataset_train_node']
        batched_dataset_train_edge = store_dataset['batched_dataset_train_edge']
        clustered_edge_index_train = store_dataset['clustered_edge_index_train']
        clustered_batch_train = store_dataset['clustered_batch_train']
        y_true_train = store_dataset['y_true_train']
        batched_dataset_val_node = store_dataset['batched_dataset_val_node']
        batched_dataset_val_edge = store_dataset['batched_dataset_val_edge']
        clustered_edge_index_val = store_dataset['clustered_edge_index_val']
        clustered_batch_val = store_dataset['clustered_batch_val']
        y_true_val = store_dataset['y_true_val']
        batched_dataset_test_node = store_dataset['batched_dataset_test_node']
        batched_dataset_test_edge = store_dataset['batched_dataset_test_edge']
        clustered_edge_index_test = store_dataset['clustered_edge_index_test']
        clustered_batch_test = store_dataset['clustered_batch_test']
        y_true_test = store_dataset['y_true_test']
    elif dataset_name in ['DD', 'MUTAG', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI','COLLAB', 'brain', 'REDDIT-BINARY']:
        i = idx
        path = osp.join(args.save_dir, dataset_name)
        with open(osp.join(path, 'batched_data_'+str(i)+'_cluster'+str(args.num_clusters)+'.pkl'), 'rb') as f:
            store_dataset = pkl.load(f)

        batched_dataset_train_node = store_dataset['batched_dataset_train_node']
        batched_dataset_train_edge = store_dataset['batched_dataset_train_edge']
        clustered_edge_index_train = store_dataset['clustered_edge_index_train']
        clustered_batch_train = store_dataset['clustered_batch_train']
        y_true_train = store_dataset['y_true_train']
        batched_dataset_test_node = store_dataset['batched_dataset_test_node']
        batched_dataset_test_edge = store_dataset['batched_dataset_test_edge']
        clustered_edge_index_test = store_dataset['clustered_edge_index_test']
        clustered_batch_test = store_dataset['clustered_batch_test']
        y_true_test = store_dataset['y_true_test']

    test_accuracy_list_all = []
    hidden_channels = args.hidden_dim
    try:
        in_channels_nodes = dataset.num_node_features
        in_channels_edges = dataset.num_edge_features
        num_classes = dataset.num_classes
    except:
        in_channels_nodes = dataset_train.num_node_features
        in_channels_edges = dataset_train.num_edge_features
        num_classes = dataset_train.num_classes

    if dataset_name in ['MNIST', 'CIFAR10']:
        in_channels_nodes += 2
    if in_channels_nodes == 0:
        nodes_feature_flag = False
        in_channels_nodes = 1
    else:
        nodes_feature_flag = True
        
    if in_channels_edges == 0:
        edge_feature_flag = False
        in_channels_edges = 1
    else:
        edge_feature_flag = True

    model = ClusterModel(
        in_channels_nodes=in_channels_nodes, 
        in_channels_edges=in_channels_edges, 
        hidden_channels=hidden_channels, 
        num_classes=num_classes, 
        num_layers=args.number_layers,
        edge_feature_flag=edge_feature_flag,
    )

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:'+str(args.gpu))
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()#torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def train(batched_dataset_node, batched_dataset_edge, batched_clustered_edge_index, batched_clustered_batch, batched_y_true):
        loss_value_list = []
        model.train()
        for data_node, data_edge, clustered_edge_index, clustered_batch, y_true in zip(batched_dataset_node, batched_dataset_edge, batched_clustered_edge_index, batched_clustered_batch, batched_y_true):
            edge_index_node, batch_node = data_node.edge_index, data_node.batch
            edge_index_edge, batch_edge = data_edge.edge_index, data_edge.batch
            if nodes_feature_flag:
                x_node = data_node.x.float()
                x_edge = data_edge.x.float()
            else:
                x_node = torch.zeros((data_node.num_nodes, 1))
                x_edge = torch.zeros((data_edge.num_nodes, 1))
            if edge_feature_flag:
                edge_attr_node = data_node.edge_attr.float()
                edge_attr_edge = data_edge.edge_attr.float()
            else:
                edge_attr_node = torch.zeros((data_node.num_edges, 1))
                edge_attr_edge = torch.zeros((data_edge.num_edges, 1))

            x_node, edge_index_node, edge_attr_node, batch_node = x_node.to(device), edge_index_node.to(device), edge_attr_node.to(device), batch_node.to(device)
            x_edge, edge_index_edge, edge_attr_edge, batch_edge = x_edge.to(device), edge_index_edge.to(device), edge_attr_edge.to(device), batch_edge.to(device) 

            if len(edge_attr_node.shape) == 1:
                edge_attr_node = edge_attr_node.view(-1,1)
                edge_attr_edge = edge_attr_edge.view(-1,1)
            if dataset_name in ['MNIST', 'CIFAR10']:
                pos_node = data_node.pos.float().to(device)
                x_node = torch.cat([x_node, pos_node], dim=-1)
                pos_edge = data_edge.pos.float().to(device)
                x_edge = torch.cat([x_edge, pos_edge], dim=-1)
            clustered_edge_index, clustered_batch = clustered_edge_index.to(device), clustered_batch.to(device)
            y_true = y_true.view(-1).to(device)

            out = model(
                x_node, edge_index_node, edge_attr_node, batch_node,
                x_edge, edge_index_edge, edge_attr_edge, batch_edge,
                clustered_edge_index, clustered_batch
            )
            loss = criterion(out, y_true)  # Compute the loss.
            loss_value = float(loss)
            loss_value_list.append(loss_value)
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
        return loss_value_list

    def test(batched_dataset_node, batched_dataset_edge, batched_clustered_edge_index, batched_clustered_batch, batched_y_true):
        y_pred_list = []
        model.eval()
        for data_node, data_edge, clustered_edge_index, clustered_batch, y_true in zip(batched_dataset_node, batched_dataset_edge, batched_clustered_edge_index, batched_clustered_batch, batched_y_true):
            edge_index_node, batch_node = data_node.edge_index, data_node.batch
            edge_index_edge, batch_edge = data_edge.edge_index, data_edge.batch
            if nodes_feature_flag:
                x_node = data_node.x.float()
                x_edge = data_edge.x.float()
            else:
                x_node = torch.zeros((data_node.num_nodes, 1))
                x_edge = torch.zeros((data_edge.num_nodes, 1))
            if edge_feature_flag:
                edge_attr_node = data_node.edge_attr.float()
                edge_attr_edge = data_edge.edge_attr.float()
            else:
                edge_attr_node = torch.zeros((data_node.num_edges, 1))
                edge_attr_edge = torch.zeros((data_edge.num_edges, 1))

            x_node, edge_index_node, edge_attr_node, batch_node = x_node.to(device), edge_index_node.to(device), edge_attr_node.to(device), batch_node.to(device)
            x_edge, edge_index_edge, edge_attr_edge, batch_edge = x_edge.to(device), edge_index_edge.to(device), edge_attr_edge.to(device), batch_edge.to(device) 

            if len(edge_attr_node.shape) == 1:
                edge_attr_node = edge_attr_node.view(-1,1)
                edge_attr_edge = edge_attr_edge.view(-1,1)
            if dataset_name in ['MNIST', 'CIFAR10']:
                pos_node = data_node.pos.float().to(device)
                x_node = torch.cat([x_node, pos_node], dim=-1)
                pos_edge = data_edge.pos.float().to(device)
                x_edge = torch.cat([x_edge, pos_edge], dim=-1)
            clustered_edge_index, clustered_batch = clustered_edge_index.to(device), clustered_batch.to(device)
            out = model(
                x_node, edge_index_node, edge_attr_node, batch_node,
                x_edge, edge_index_edge, edge_attr_edge, batch_edge,
                clustered_edge_index, clustered_batch
            )
            #pred = out.argmax(dim=1)  # Use the class with highest probability.
            #y_pred_list.append(pred.detach().cpu())
            out = out.softmax(dim=1)
            y_pred_list.append(out.detach().cpu())

        y_pred = torch.cat(y_pred_list)
        return y_pred 

    start_time = time.time()
    test_accuracy_list = []

    directory = os.path.join(args.write_dir, args.dataset, str(args.num_clusters), str(args.idx), str(args.hidden_dim)+'_'+str(args.number_layers)+'_'+str(args.lr))
    print(args.dataset, str(args.num_clusters), str(args.idx), str(args.hidden_dim), str(args.number_layers), str(args.lr))
    if not os.path.exists(directory):
        os.makedirs(directory)

    f = open(os.path.join(directory, 'log.txt'), 'a')
    if dataset_name in ['hiv', 'bace', 'bbbp', 'UPFD']:
        print('Epoch, Loss, Train rocauc, Val rocauc, Test rocauc, Time', file=f)
    elif dataset_name in ['MNIST', 'CIFAR10']:
        print('Epoch, Loss, Train Accuracy, Val Accuracy, Test Accuracy, Time', file=f)
    else:
        print('Epoch, Loss, Train Accuracy, Test Accuracy, Time', file=f)
    
    results = []
    for epoch in tqdm(range(200)):
        loss_value_list = train(batched_dataset_train_node, batched_dataset_train_edge, clustered_edge_index_train, clustered_batch_train, y_true_train)
        y_pred_train = test(batched_dataset_train_node, batched_dataset_train_edge, clustered_edge_index_train, clustered_batch_train, y_true_train)
        y_pred_test = test(batched_dataset_test_node, batched_dataset_test_edge, clustered_edge_index_test, clustered_batch_test, y_true_test)

        loss_value = np.mean(loss_value_list)

        # reshape y_true
        y_true_train_1d = torch.cat(y_true_train).view(-1).numpy()
        y_true_test_1d = torch.cat(y_true_test).view(-1).numpy()
        y_true_train_2d = np.zeros((y_true_train_1d.size, y_true_train_1d.max()+1))
        y_true_train_2d[np.arange(y_true_train_1d.size),y_true_train_1d] = 1
        y_true_test_2d = np.zeros((y_true_test_1d.size, y_true_test_1d.max()+1))
        y_true_test_2d[np.arange(y_true_test_1d.size),y_true_test_1d] = 1

        if dataset_name in ['MNIST', 'CIFAR10', 'hiv', 'bace', 'bbbp', 'UPFD']:
            y_pred_val = test(batched_dataset_val_node, batched_dataset_val_edge, clustered_edge_index_val, clustered_batch_val, y_true_val)
            y_true_val_1d = torch.cat(y_true_val).view(-1).numpy()
            y_true_val_2d = np.zeros((y_true_val_1d.size, y_true_val_1d.max()+1))
            y_true_val_2d[np.arange(y_true_val_1d.size),y_true_val_1d] = 1

        if dataset_name in ['hiv', 'bace', 'bbbp', 'UPFD']:
            train_accuracy = roc_auc_score(y_true_train_2d, y_pred_train.numpy())
            val_accuracy = roc_auc_score(y_true_val_2d, y_pred_val.numpy())
            test_accuracy = roc_auc_score(y_true_test_2d, y_pred_test.numpy())
            results.append([epoch, loss_value, train_accuracy, val_accuracy, test_accuracy, time.time()-start_time])
            print("{:03d}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(epoch, loss_value, train_accuracy, val_accuracy, test_accuracy, time.time()-start_time), file=f)
        elif dataset_name in ['MNIST', 'CIFAR10']:
            y_pred_train, y_pred_val, y_pred_test = y_pred_train.argmax(dim=1), y_pred_val.argmax(dim=1), y_pred_test.argmax(dim=1)
            train_accuracy = accuracy_score(y_true_train_1d, y_pred_train.numpy())
            val_accuracy = accuracy_score(y_true_val_1d, y_pred_val.numpy())
            test_accuracy = accuracy_score(y_true_test_1d, y_pred_test.numpy())
            results.append([epoch, loss_value, train_accuracy, val_accuracy, test_accuracy, time.time()-start_time])
            print("{:03d}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(epoch, loss_value, train_accuracy, val_accuracy, test_accuracy, time.time()-start_time), file=f)
        else:
            y_pred_train, y_pred_test = y_pred_train.argmax(dim=1), y_pred_test.argmax(dim=1)
            train_accuracy = accuracy_score(y_true_train_1d, y_pred_train.numpy())
            test_accuracy = accuracy_score(y_true_test_1d, y_pred_test.numpy())
            results.append([epoch, loss_value, train_accuracy, test_accuracy, time.time()-start_time])
            print("{:03d}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(epoch, loss_value, train_accuracy, test_accuracy, time.time()-start_time), file=f)
            
    if dataset_name in ['hiv', 'bace', 'bbbp', 'UPFD']:
        df = pd.DataFrame(data=results, columns=['Epoch', 'Loss', 'Train rocauc', 'Val rocauc', 'Test rocauc', 'Time'])
    elif dataset_name in ['MNIST', 'CIFAR10']:
        df = pd.DataFrame(data=results, columns=['Epoch', 'Loss', 'Train Accuracy', 'Val Accuracy', 'Test Accuracy', 'Time'])
    else:
        df = pd.DataFrame(data=results, columns=['Epoch', 'Loss', 'Train Accuracy', 'Test Accuracy', 'Time'])
        
    df.to_csv(os.path.join(directory, 'results.csv'), index=False)


