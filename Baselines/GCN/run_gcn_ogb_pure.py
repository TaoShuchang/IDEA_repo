import os,sys
#设置当前工作目录，放再import其他路径模块之前
os.chdir(sys.path[0])
#注意默认工作目录在顶层文件夹，这里是相对于顶层文件夹的目录
sys.path.append("./")
import numpy as np
import scipy.sparse as sp
import time
import csv
import pandas as pd
import os, sys
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from deeprobust.graph.utils import *
# from deeprobust.graph.defense import GraphConvolution 
from deeprobust.graph.data import Dataset
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from torch_sparse import SparseTensor
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
sys.path.append('../../')
from utils import *

setup_seed(123)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        # self.convs.append(GraphConvolution(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            # self.convs.append(GraphConvolution(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
        # self.convs.append(GraphConvolution(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

def train(model, feat, adj_tensor, labels, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(feat, adj_tensor)
    loss = F.nll_loss(out[train_idx], labels.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model, feat, adj_tensor, labels, split_idx, evaluator):
    model.eval()

    out = model(feat, adj_tensor)
    y_pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        'y_true': labels[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': labels[split_idx['val']],
        'y_pred': y_pred[split_idx['val']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': labels[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

def main(opts):
    dataset= opts['dataset']
    connect = opts['connect']
    suffix = opts['suffix']
    lr = opts['lr']
    weight_decay = opts['decay']
    nepochs = opts['nepochs']
    prefile = '../../'
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print('device', device)
    device = torch.device(device)
    
    if args.dataset in ['citeseer','cora', 'pubmed']:
        data = Dataset(root=prefile+'datasets/', name=args.dataset, setting='nettack')
        adj, features, labels_np = data.adj, data.features, data.labels
        train_mask, val_mask, test_mask = data.idx_train, data.idx_val, data.idx_test
        n = adj.shape[0]
        adj_tensor, feat, labels = preprocess(adj, features, labels_np, preprocess_adj=False, sparse=True, device=device)
    elif args.dataset in ['ogbarxiv']:
        args.dataset = 'ogbarxiv_oset'
        split = np.aload(prefile + 'splits/split_118/' + args.dataset+ '_split.npy').item()
        train_mask, val_mask, test_mask = split['train'], split['val'], split['test']
        dataset = PygNodePropPredDataset(name = "ogbn-arxiv", root = '../../dataset/', transform=T.ToSparseTensor())
        data = dataset[0]
        data = data.to(device)
        adj_tensor = data.adj_t.to_symmetric()
        feat, labels = data.x, data.y
    else:
        adj, features, labels_np = load_npz(prefile + f'datasets/{args.dataset}.npz')
        adj, features, labels_np, n = lcc_graph(adj, features, labels_np)
        split = np.aload(prefile + 'splits/split_118/' + args.dataset+ '_split.npy').item()
        train_mask, val_mask, test_mask = split['train'], split['val'], split['test']
        adj_tensor, feat, labels = preprocess(adj, features, labels_np, preprocess_adj=False, sparse=True, device=device)

    save_file = 'checkpoint/' + args.dataset + '/' + suffix
    graph_file = prefile + f'Attacks/metattack/new_graphs/{args.dataset}/{args.atk_suffix}_'
    for pert in np.arange(0, 0.3, 0.05):
        print('------------------- Perturbation', pert, '-------------------')
        if pert > 0:
            print('dataset',dataset)
            # if dataset == 'pubmed':
            #     new_adj_sp = sp.load_npz(prefile + f'final_graphs/meta/{args.dataset}_meta_adj_{pert:.2f}.npz')
            #     adj_tensor = sparse_mx_to_torch_sparse_tensor(new_adj_sp).to(device)
            # else:
            #     new_adj = torch.load(graph_file + f'{pert:.2f}.pt')
            #     print('adj',new_adj.to_dense().sum())
            #     adj_tensor = new_adj.to(device)
            if args.dataset == 'pubmed':
                adj = sp.load_npz(prefile + f'final_graphs/meta/{args.dataset}_meta_adj_{pert:.2f}.npz')
                # adj_tensor = sparse_mx_to_torch_sparse_tensor(new_adj_sp).to(device)
            elif 'ogbarxiv' in args.dataset:
                break
            else:
                new_adj = torch.load(graph_file + f'{pert:.2f}.pt')
                print('adj',new_adj.to_dense().sum())
                adj = sparse_tensor_to_torch_sparse_mx(new_adj)
        evaluator = Evaluator(name='ogbn-arxiv')
        seeds = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
        # seeds = [120]
        acc_test_arr = []
        for i in seeds:
            setup_seed(i)
            print('----------- Seed ',i, '-----------')
            # gcn = GCN(nfeat=features.shape[1], nhid=args.hid, nclass=labels.max().item() + 1, dropout=0.5, lr=args.lr, weight_decay=args.decay, device=device)
            # gcn = gcn.to(device)
            # gcn.fit(features, adj, labels_np, train_mask, val_mask, train_iters=2000, patience=500, verbose=True)
            model = GCN(feat.shape[1], args.hid,
                    labels.max().item() + 1, args.num_layers,
                    args.dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            for epoch in range(1, 1 + nepochs):
                loss = train(model, feat, adj_tensor, labels, train_mask, optimizer)
                
                # y_pred = logits.argmax(dim=-1, keepdim=True)
                if epoch % 10 == 0:
                    train_acc, val_acc, test_acc = test(model, feat, adj_tensor, labels, split, evaluator)
                    print("IT {:05d} | TrainLoss {:.4f} | TrainAcc {:.5f} | ValAcc: {:.5f} | TestAcc: {:.5f}".format(
                    epoch, loss, train_acc, val_acc, test_acc))
                    # print("Validate accuracy {:.4f}".format(val_acc))
                    model.eval()
                    logits = model(feat, adj_tensor)
                    val_acc = accuracy(logits[val_mask], labels.squeeze(1)[val_mask])
                    train_acc = accuracy(logits[train_mask], labels.squeeze(1)[train_mask])
                    train_acc = accuracy(logits[test_mask], labels.squeeze(1)[test_mask])
                    print("IT {:05d} | TrainLoss {:.4f} | TrainAcc {:.5f} | ValAcc: {:.5f} | TestAcc: {:.5f}".format(
                    epoch, loss, train_acc, val_acc, test_acc))

            torch.save(model.state_dict(), save_file + f'_pert{pert:.2f}_seed' + str(i) + '_checkpoint.pt')  
            model.eval()
            train_acc, val_acc, test_acc = test(model, feat, adj_tensor, labels, split, evaluator)
            print("Validate accuracy {:.4f}".format(val_acc))
            print("Test accuracy {:.4f}".format(test_acc))
            val_acc = accuracy(logits[val_mask], labels.squeeze(1)[val_mask])
            test_acc = accuracy(logits[test_mask], labels.squeeze(1)[test_mask])
            acc_test_arr.append(test_acc)
            print("Validate accuracy {:.4f}".format(val_acc))
            print("Test accuracy {:.4f}".format(test_acc))
            
        file = 'log/' + args.dataset + '/' + args.dataset + '.csv'
        
        nseed = len(seeds)
        ncol = int(len(acc_test_arr)/nseed)
        acc_test_arr = np.array(acc_test_arr).reshape(nseed, ncol) * 100
        acc_test_f = np.concatenate((acc_test_arr, acc_test_arr.mean(0).reshape(1, ncol), acc_test_arr.std(0).reshape(1, ncol)))
        print('acc_test_arr', acc_test_arr.shape)
        dataframe_test =  pd.DataFrame(acc_test_f[-2:])
        with open(file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['=====',args.suffix, f'_pert{pert:.2f}_seed','====='])
            writer.writerow(['---Test ACC---'])
        # dataframe = pd.DataFrame({u'graph_name_arr':graph_name_arr, u'acc_test':acc_test_arr, u'acc_target':acc_tar_arr})
        dataframe_test.to_csv(file, mode='a', index=False)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='GCN')

    # configure
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--connect', default=True, type=bool, help='lcc')
    parser.add_argument('--suffix', type=str, default='_', help='suffix of the checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='device')

    # dataset
    parser.add_argument('--dataset', default='ogbarxiv',
                        help='dataset to use')
    parser.add_argument('--optimizer', choices=['Adam','SGD', 'RMSprop','Adadelta', 'AdamW'], default='RMSprop',
                        help='optimizer')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--hid', default=128, type=int, help='hidden dimension')
    parser.add_argument('--decay', default=0, type=float, help='weight decay')
    parser.add_argument('--num_layers', default=3, type=int, help='number of layers')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout')
    
    parser.add_argument('--nepochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--runs', type=int, default=10, help='runs number')
    parser.add_argument('--atk_flag', type=int, default=0, help='Whether is attack')
    parser.add_argument('--atk_suffix', type=str, default='', help='Whether is attack')
    args = parser.parse_args()
    opts = args.__dict__.copy()
    print('opts', opts)
    main(opts)

'''

nohup python -u run_gcn_ogb_pure.py --dataset ogbarxiv --suffix ogb_pure_oset_256_eval --lr 0.01 --hid 512 --atk_suffix pre --gpu 5 > log/ogbarxiv_oset/ogb_pure_oset_256_eval.log 2>&1 &

'''