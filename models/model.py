import torch
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GINConv, GraphConv, BatchNorm, SAGEConv
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU
from torch.autograd import Variable
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter
from torch_geometric.data import Batch
from torch_geometric.loader import ClusterData, ClusterLoader, DataLoader
from torch_sparse import SparseTensor, cat
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import ClusterData, ClusterLoader, DataLoader, ShaDowKHopSampler
from torch_geometric.nn import BatchNorm, SAGEConv
from torch_sparse import SparseTensor, cat
from torch_scatter import scatter
from torch_geometric.nn import GINConv, GINEConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from typing import Callable, Optional, Tuple, Union
from torch import Tensor
from torch.nn import LSTM
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.mlp = ModuleList()
        self.mlp.append(Linear(in_channels, hidden_channels))
        if num_layers>=2:
            for _ in range(num_layers-2):
                self.mlp.append(Linear(hidden_channels, hidden_channels))
        self.mlp.append(Linear(hidden_channels, out_channels))
    
    def forward(self, x):
        for layer in self.mlp[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.mlp[-1](x)
        return x
    
class GraphConvNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.lin_init = Linear(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.convs.append(GraphConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, batch):
        x = self.lin_init(x)
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = x + conv(x, edge_index) 
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        x = self.convs[-1](x, edge_index)
        
        return global_add_pool(x, batch)
    
class SAGEConvNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.lin_init = Linear(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, batch):
        x = self.lin_init(x)
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = x + conv(x, edge_index) 
            #x = batch_norm(x)
            x = F.relu(x)
            #x = F.dropout(x, p=0.2, training=self.training)
        x = self.convs[-1](x, edge_index)
        
        return global_add_pool(x, batch)

class GINEConv(MessagePassing):
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 edge_dim: Optional[int] = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            if hasattr(self.nn[0], 'in_features'):
                in_channels = self.nn[0].in_features
            else:
                in_channels = self.nn[0].in_channels
            self.lin = Linear(edge_dim, in_channels)
        else:
            self.lin = None

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)


    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)
            
        #print(x_j.shape, edge_attr.shape)

        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'
    
class GINNet(torch.nn.Module):
    def __init__(self, in_channels_nodes, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.lin_init = MLP(in_channels_nodes, hidden_channels, hidden_channels, num_layers=2)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            nn = MLP(hidden_channels, hidden_channels, hidden_channels, num_layers=2)
            self.convs.append(GINConv(nn=nn, eps=0.1))
            self.batch_norms.append(BatchNorm(hidden_channels))
            
        nn = MLP(hidden_channels, hidden_channels, out_channels, num_layers=2)
        self.convs.append(GINConv(nn=nn, eps=0.1))

    def forward(self, x, edge_index, batch):
        x = self.lin_init(x)
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = x + conv(x, edge_index) 
            #x = batch_norm(x)
            x = F.relu(x)
            #x = F.dropout(x, p=0.2, training=self.training)
        x = self.convs[-1](x, edge_index)
        
        return global_add_pool(x, batch)
    
class GINENet(torch.nn.Module):
    def __init__(self, in_channels_nodes, in_channels_edges, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.lin_init_node = MLP(in_channels_nodes, hidden_channels, hidden_channels, num_layers=2)
        self.lin_init_edge = MLP(in_channels_edges, hidden_channels, hidden_channels, num_layers=2)
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            nn = MLP(hidden_channels, hidden_channels, hidden_channels, num_layers=2)
            self.convs.append(GINEConv(nn=nn, eps=0.1))
            self.batch_norms.append(BatchNorm(hidden_channels))
            
        nn = MLP(hidden_channels, hidden_channels, out_channels, num_layers=2)
        self.convs.append(GINEConv(nn=nn, eps=0.1))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.lin_init_node(x)
        edge_attr = self.lin_init_edge(edge_attr)
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = x + conv(x, edge_index, edge_attr) 
            #x = batch_norm(x)
            x = F.relu(x)
            #x = F.dropout(x, p=0.2, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr)
        
        return global_add_pool(x, batch)
    
class ClusterModel(torch.nn.Module):
    def __init__(self, in_channels_nodes, in_channels_edges, hidden_channels, num_classes, num_layers, edge_feature_flag):
        super().__init__()
        if edge_feature_flag:
            self.net1 = GINENet(in_channels_nodes, in_channels_edges, hidden_channels, hidden_channels, num_layers)
        else:
            self.net1 = GINNet(in_channels_nodes, hidden_channels, hidden_channels, num_layers)
            
        self.net2 = GINENet(hidden_channels, hidden_channels, hidden_channels, num_classes, num_layers)
        self.edge_feature_flag = edge_feature_flag
        
    def forward(self, 
                x_node, edge_index_node, edge_attr_node, batch_node,
                x_edge, edge_index_edge, edge_attr_edge, batch_edge,
                edge_index_cluster, batch_cluster):
        if self.edge_feature_flag:
            # with edge feature
            edge_attr_cluster = self.net1(x_edge, edge_index_edge, edge_attr_edge, batch_edge)
            x_cluster = self.net1(x_node, edge_index_node, edge_attr_node, batch_node)
        else: 
            # no edge feature
            edge_attr_cluster = self.net1(x_edge, edge_index_edge, batch_edge)
            x_cluster = self.net1(x_node, edge_index_node, batch_node)
            
        out = self.net2(x_cluster, edge_index_cluster, edge_attr_cluster, batch_cluster)
        return out


class SAGEEdgeConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: str = 'mean',
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        kwargs['aggr'] = aggr if aggr != 'lstm' else None
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if self.project:
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if self.aggr is None:
            self.fuse = False  # No "fused" message_and_aggregate.
            self.lstm = LSTM(in_channels[0], in_channels[0], batch_first=True)

        self.lin_t = Linear(in_channels[0], in_channels[0], bias=bias)
        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        if self.project:
            self.lin.reset_parameters()
        if self.aggr is None:
            self.lstm.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()
    
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out
    
    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return x_j + self.lin_t(edge_attr)

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def aggregate(self, x: Tensor, index: Tensor, ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        if self.aggr is not None:
            return scatter(x, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)

        # LSTM aggregation:
        if ptr is None and not torch.all(index[:-1] <= index[1:]):
            raise ValueError(f"Can not utilize LSTM-style aggregation inside "
                             f"'{self.__class__.__name__}' in case the "
                             f"'edge_index' tensor is not sorted by columns. "
                             f"Run 'sort_edge_index(..., sort_by_row=False)' "
                             f"in a pre-processing step.")

        x, mask = to_dense_batch(x, batch=index, batch_size=dim_size)
        out, _ = self.lstm(x)
        return out[:, -1]

    def __repr__(self) -> str:
        aggr = self.aggr if self.aggr is not None else 'lstm'
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={aggr})')
class SAGENet(torch.nn.Module):
    def __init__(self, in_channels_nodes, in_channels_edges, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.lin_init_node = MLP(in_channels_nodes, hidden_channels, hidden_channels, num_layers=2)
        self.lin_init_edge = MLP(in_channels_edges, hidden_channels, hidden_channels, num_layers=2)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(SAGEEdgeConv(hidden_channels, hidden_channels))
            
        self.convs.append(SAGEEdgeConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.lin_init_node(x)
        edge_attr = self.lin_init_edge(edge_attr)
        for conv in self.convs[:-1]:
            x = x + conv(x, edge_index, edge_attr) 
            x = F.relu(x)
        x = self.convs[-1](x, edge_index, edge_attr)
        
        return global_mean_pool(x, batch) 

    
class ClusterSageModel(torch.nn.Module):
    def __init__(self, in_channels_nodes, in_channels_edges, hidden_channels, num_classes, num_layers, edge_feature_flag):
        super().__init__()
        self.net1 = SAGENet(in_channels_nodes, in_channels_edges, hidden_channels, hidden_channels, num_layers)
            
        self.net2 = SAGENet(hidden_channels, hidden_channels, hidden_channels, num_classes, num_layers)
        self.edge_feature_flag = edge_feature_flag
        
    def forward(self, 
                x_node, edge_index_node, edge_attr_node, batch_node,
                x_edge, edge_index_edge, edge_attr_edge, batch_edge,
                edge_index_cluster, batch_cluster):
        edge_attr_cluster = self.net1(x_edge, edge_index_edge, edge_attr_edge, batch_edge)
        x_cluster = self.net1(x_node, edge_index_node, edge_attr_node, batch_node)
            
        out = self.net2(x_cluster, edge_index_cluster, edge_attr_cluster, batch_cluster)
        return out
    
    
class EdgeConvConv(MessagePassing):
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 edge_dim: Optional[int] = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            if hasattr(self.nn[0], 'in_features'):
                in_channels = self.nn[0].in_features
            else:
                in_channels = self.nn[0].in_channels
            self.lin = Linear(edge_dim, in_channels)
        else:
            self.lin = None

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        return out


    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
           
        temp = torch.cat([x_i, x_j, edge_attr], dim=1)

        return self.nn(temp)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'
    
class EdgeConvNet(torch.nn.Module):
    def __init__(self, in_channels_nodes, in_channels_edges, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.lin_init_node = MLP(in_channels_nodes, hidden_channels, hidden_channels, num_layers=2)
        self.lin_init_edge = MLP(in_channels_edges, hidden_channels, hidden_channels, num_layers=2)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            nn = MLP(hidden_channels*3, hidden_channels, hidden_channels, num_layers=2)
            self.convs.append(EdgeConvConv(nn=nn, eps=0.1))
            
        nn = MLP(hidden_channels*3, hidden_channels, out_channels, num_layers=2)
        self.convs.append(EdgeConvConv(nn=nn, eps=0.1))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.lin_init_node(x)
        edge_attr = self.lin_init_edge(edge_attr)
        for conv in self.convs[:-1]:
            x = x + conv(x, edge_index, edge_attr) 
            x = F.relu(x)
        x = self.convs[-1](x, edge_index, edge_attr)
        
        return global_add_pool(x, batch)
    
class ClusterEdgeConvModel(torch.nn.Module):
    def __init__(self, in_channels_nodes, in_channels_edges, hidden_channels, num_classes, num_layers, edge_feature_flag):
        super().__init__()
        self.net1 = EdgeConvNet(in_channels_nodes, in_channels_edges, hidden_channels, hidden_channels, num_layers)
            
        self.net2 = EdgeConvNet(hidden_channels, hidden_channels, hidden_channels, num_classes, num_layers)
        self.edge_feature_flag = edge_feature_flag
        
    def forward(self, 
                x_node, edge_index_node, edge_attr_node, batch_node,
                x_edge, edge_index_edge, edge_attr_edge, batch_edge,
                edge_index_cluster, batch_cluster):
        edge_attr_cluster = self.net1(x_edge, edge_index_edge, edge_attr_edge, batch_edge)
        x_cluster = self.net1(x_node, edge_index_node, edge_attr_node, batch_node)
            
        out = self.net2(x_cluster, edge_index_cluster, edge_attr_cluster, batch_cluster)
        return out