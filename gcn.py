import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import scipy.sparse as sp
import dgl
import math

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_normal_(self.weight)
        # nn.init.kaiming_normal_(self.bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        output = torch.mm(input, self.weight)
        # output = torch.sparse.mm(adj, output)
        # try:
        row, column = adj.coalesce().indices()
        g = dgl.graph((column, row), num_nodes=adj.shape[0], device=adj.device)
        output = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=output, rhs_data=adj.coalesce().values())
        # except:
        #     row, column, value = adj.coo()
        #     g = dgl.graph((column, row), num_nodes=adj.size(0), device=adj.device())
        #     output = dgl.ops.gspmm(g, 'mul', 'sum', lhs_data=output, rhs_data=value)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, layer_norm_first=False, use_ln=True):
        super(GCN, self).__init__()
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln
        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConvolution(in_channels, hidden_channels))
        self.lns = torch.nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(in_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GraphConvolution(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.convs.append(GraphConvolution(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()

    def forward(self, x, adj_t, layers=-1):
        if self.layer_norm_first:
            x = self.lns[0](x)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.use_ln:
                x = self.lns[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # obtain output from the i-th layer
            if layers == i+1:
                return x
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

    def con_forward(self,x,adj_t,layers=-1):
        if self.layer_norm_first and layers==1:
            x = self.lns[0](x)
        for i in range(layers-1,len(self.convs)-1):
            x = self.convs[i](x, adj_t)
            if self.use_ln:
                x = self.lns[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
