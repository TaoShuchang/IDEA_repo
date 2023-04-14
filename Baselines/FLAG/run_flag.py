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
from copy import deepcopy
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse

from attacks import *
from gcn_pyg import GCN

sys.path.append('../../')
from utils import *

setup_seed(123)

def training(net, nor_adj_tensor, feat, labels, train_mask, val_mask, test_mask, net_save_file, device):
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    stopper = EarlyStop(patience=1000)
    for epoch in range(1, args.epochs + 1):
        net.train()

        train_loss = train_flag(net, nor_adj_tensor, feat, labels, train_mask, optimizer, device, args)

        net.eval()
        # val_loss = train_flag(net, nor_adj_tensor, feat, labels, val_mask, optimizer, device, args)
        logits = net(feat, nor_adj_tensor)
        train_acc = accuracy(logits[train_mask], labels[train_mask])
        val_acc = accuracy(logits[val_mask], labels[val_mask])
        # test_acc = accuracy(logits[test_mask], labels[test_mask])

    # acc = accuracy(logits, labels)
        if epoch % 500 == 0:
            print("Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.5f} | Val Acc: {:.5f} ".format(
                epoch, train_loss, train_acc, val_acc))
        es = 1-val_acc

        if stopper.step(es, net, net_save_file):   
            break
    net.load_state_dict(torch.load(net_save_file+'_checkpoint.pt'))
    net.eval()
    logits = net(feat, nor_adj_tensor)
    logp = F.log_softmax(logits, dim=1)
    val_acc = accuracy(logp[val_mask], labels[val_mask])
    test_acc = accuracy(logp[test_mask], labels[test_mask])
    return test_acc


def train_flag(model, adj_tensor, feat, labels, train_idx, optimizer, device, args):
    
    y = labels[train_idx]
    if 'prod' in args.suffix:
        loss, _ = flag_products(model, feat, y,  adj_tensor, args, optimizer, device, F.cross_entropy, train_idx=train_idx)
    else:
        forward = lambda perturb : model(feat+perturb, adj_tensor)[train_idx]
        model_forward = (model, forward)

        loss, _ = flag(model_forward, feat.shape, y, args, optimizer, device, F.cross_entropy)
    return loss.item()

def main(opts):
    dataset= opts['dataset']
    suffix = opts['suffix']


    prefile = '../../'
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print('device', device)
    device = torch.device(device)
    save_file = 'checkpoint/' + dataset + '/' + suffix
    graph_file = prefile + f'Attacks/metattack/new_graphs/{args.dataset}/{args.atk_suffix}_'
    # if args.atk_flag == 0:
    if args.dataset in ['citeseer','cora', 'pubmed']:
        data = Dataset(root=prefile+'datasets/', name=args.dataset, setting='nettack')
        adj, features, labels_np = data.adj, data.features, data.labels
        train_mask, val_mask, test_mask = data.idx_train, data.idx_val, data.idx_test
        n = adj.shape[0]
        print('loading')
    else:
        adj, features, labels_np = load_npz(prefile + f'datasets/{args.dataset}.npz')
        adj, features, labels_np, n = lcc_graph(adj, features, labels_np)
        split = np.aload(prefile + 'splits/split_118/' + args.dataset+ '_split.npy').item()
        train_mask, val_mask, test_mask = split['train'], split['val'], split['test']
    adj_tensor, nor_adj_tensor, feat, labels = graph_to_tensor(adj, features, labels_np, device)
    

    acc_test_arr = []
    pert_rate = np.arange(0, 0.3, 0.05)
    for pert in pert_rate:
        print('------------------- Perturbation', pert, '-------------------')
        if pert > 0:
            if dataset == 'pubmed':
                adj = sp.load_npz(prefile + f'final_graphs/meta/{args.dataset}_meta_adj_{pert:.2f}.npz')
                adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
            elif dataset == 'ogbarxiv':
                pert_rate = np.array([0])
                break
            else:
                adj_tensor = torch.load(graph_file + f'{pert:.2f}.pt').to(device)
                # print('adj',new_adj.to_dense().sum())
                # adj = sparse_tensor_to_torch_sparse_mx(new_adj)
                # adj_tensor = new_adj.to(device)
            nor_adj_tensor = normalize_tensor(adj_tensor)
        
        # seeds = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
        seeds = [120]
        for i in seeds:
            setup_seed(i)
            print('----------- Seed ',i, '-----------')
            net = GCN(features.shape[1], args.hidden_channels, labels.max().item() + 1, args.num_layers, dropout=args.dropout, layer_norm_first=args.layer_norm_first, use_ln=args.use_ln).float().to(device)
            start_time = time.time()
            test_acc = training(net, nor_adj_tensor, feat, labels, train_mask, val_mask, test_mask, net_save_file=save_file + f'_pert{pert:.2f}_seed' + str(i) , device=device)
            end_time = time.time()
            during = end_time - start_time
            print('During Time:', during, 'seconds;', during/60, 'minutes;', during/60/60, 'hours;')
            
            print("Test accuracy {:.4f}".format(test_acc))
            acc_test_arr.append(test_acc)

            
            
    file = 'log/' + args.dataset + '/' + args.dataset + '.csv'
    
    nseed = len(seeds)
    nrow = int(len(acc_test_arr)/nseed)
    acc_test_arr = np.array(acc_test_arr).reshape(nrow, nseed) * 100
    # acc_test_f = np.concatenate((acc_test_arr, acc_test_arr.mean(1).reshape(nrow,1), acc_test_arr.std(0).reshape(nrow, 1)))
    print('acc_test_arr', acc_test_arr.shape)
    dataframe_test =  pd.DataFrame({u'pert_rate':pert_rate, u'mean':acc_test_arr.mean(1), u'std':acc_test_arr.std(1)})
    with open(file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['=====',args.suffix, f'_pert{pert:.2f}_seed','====='])
        writer.writerow(['---Test ACC---'])
    # dataframe = pd.DataFrame({u'graph_name_arr':graph_name_arr, u'acc_test':acc_test_arr, u'acc_target':acc_tar_arr})
    dataframe_test.to_csv(file, mode='a', index=False)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='model')

    # configure
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--connect', default=True, type=bool, help='lcc')
    parser.add_argument('--suffix', type=str, default='_', help='suffix of the checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='device')

    # dataset
    parser.add_argument('--dataset', default='citeseer',
                        help='dataset to use')
    parser.add_argument('--optimizer', choices=['Adam','SGD', 'RMSprop','Adadelta'], default='RMSprop',
                        help='optimizer')
    parser.add_argument('--patience', default=500, type=int, help='patience')
    parser.add_argument('--runs', type=int, default=10, help='runs number')
    parser.add_argument('--atk_flag', type=int, default=0, help='Whether is attack')
    parser.add_argument('--atk_suffix', type=str, default='', help='Whether is attack')

    parser.add_argument('--perturb_size', type=float, default=1e-3)
    parser.add_argument('-m', type=int, default=3)
    parser.add_argument('--amp', type=int, default=2)
    parser.add_argument('--test-freq', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=2000)
  
    parser.add_argument('--layer_norm_first', action="store_true")
    # put layer norm between layers or not
    parser.add_argument('--use_ln', type=int,default=0)

    
    args = parser.parse_args()
    opts = args.__dict__.copy()
    print('opts', opts)
    main(opts)

'''
CUDA_VISIBLE_DEVICES=2 nohup python -u run_flag.py --epochs 2000 --dataset ogbarxiv --suffix testtime_flag1 --atk_suffix seed123 > log/ogbarxiv/testtime_flag1.log 2>&1 & 


CUDA_VISIBLE_DEVICES=1 nohup python -u run_flag.py --dataset ogbarxiv --suffix flag --atk_suffix seed123 > log/ogbarxiv/flag.log 2>&1 & 
CUDA_VISIBLE_DEVICES=1 nohup python -u run_flag.py --dataset 12k_reddit --suffix flag --atk_suffix seed123 > log/12k_reddit/flag.log 2>&1 & 
CUDA_VISIBLE_DEVICES=4 nohup python -u run_flag.py --dataset 10k_ogbproducts --suffix flag --atk_suffix seed123 > log/10k_ogbproducts/flag.log 2>&1 & 

CUDA_VISIBLE_DEVICES=0 nohup python -u run_flag.py --dataset citeseer --suffix flag --atk_suffix seed123 > log/citeseer/flag.log 2>&1 & 
CUDA_VISIBLE_DEVICES=4 nohup python -u run_flag.py --dataset cora --suffix flag --atk_suffix seed123 > log/cora/flag.log 2>&1 & 
CUDA_VISIBLE_DEVICES=3 nohup python -u run_flag.py --dataset pubmed --suffix flag --atk_suffix pre > log/pubmed/flag.log 2>&1 & 

'''