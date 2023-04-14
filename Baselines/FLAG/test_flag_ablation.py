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

def main(opts):
    dataset= opts['dataset']
    suffix = opts['suffix']

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print('device', device)
    device = torch.device(device)
    prefile = '../../'
    save_file = 'checkpoint/' + dataset + '/' + suffix
    graphpath = prefile + 'final_graphs/' + args.dataset + '/' 
    target_nodes = np.load(prefile + 'splits/target_nodes/' + args.dataset+ '_tar.npy')
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
    
    graph_save_file = get_filelist(graphpath, [], name='s_')
    graph_save_file.sort()
    seeds = [123]
    
    acc_tar_arr = []
    for i in seeds:
        graph_name_arr = []
        setup_seed(i)
        print('----------- Seed ',i, '-----------')
        net = GCN(features.shape[1], args.hidden_channels, labels.max().item() + 1, args.num_layers, dropout=args.dropout, layer_norm_first=args.layer_norm_first, use_ln=args.use_ln).float().to(device)
        
        net.load_state_dict(torch.load(save_file + f'_pert0.00_seed' + str(i) + '_checkpoint.pt'))
        net.eval()

        logits = net(feat, nor_adj_tensor, layers=1)

        print("logits shape: ", logits.shape)

        clean_z = logits.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        np.save("../../visualization/FLAG/clean_z.npy" ,clean_z)
        np.save("../../visualization/FLAG/labels.npy" ,labels)

        n_clean_nodes = logits.size(0)

        for graph in graph_save_file:
            graph_name = graph.split('/')[-1]
            graph_name_arr.append(graph_name)
            print('inject attack',graph_name)
            new_adj, new_features, new_labels_np = load_npz(graph)
            
            new_adj_tensor = sparse_mx_to_torch_sparse_tensor(new_adj).to(device)
            new_nor_adj_tensor = normalize_tensor(new_adj_tensor)
            new_feat = torch.from_numpy(new_features.toarray().astype('double')).float().to(device)
            new_logits = net(new_feat, new_nor_adj_tensor, layers=1)

            pert_z = new_logits[:n_clean_nodes, :]
            pert_z = pert_z.cpu().detach().numpy()
            np.save(f"../../visualization/FLAG/pert_{graph_name[:-4]}.npy" ,pert_z)
            

           
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
  
    parser.add_argument('--layer_norm_first', default=True, action="store_true")
    # put layer norm between layers or not
    parser.add_argument('--use_ln', type=int,default=0)

    
    args = parser.parse_args()
    opts = args.__dict__.copy()
    print('opts', opts)
    main(opts)

'''
CUDA_VISIBLE_DEVICES=2 nohup python -u test_flag_ablation.py --dataset ogbarxiv --suffix flag --atk_suffix seed123 > log/ogbarxiv/evasion_flag.log 2>&1 & 
CUDA_VISIBLE_DEVICES=1 nohup python -u test_flag_ablation.py --dataset 12k_reddit --suffix flag --atk_suffix seed123 > log/12k_reddit/evasion_flag.log 2>&1 & 
CUDA_VISIBLE_DEVICES=3 nohup python -u test_flag_ablation.py --dataset 10k_ogbproducts --suffix flag --atk_suffix seed123 > log/10k_ogbproducts/evasion_flag.log 2>&1 & 

CUDA_VISIBLE_DEVICES=0 nohup python -u test_flag_ablation.py --dataset citeseer --suffix flag --atk_suffix seed123 > log/citeseer/evasion_flag.log 2>&1 & 
CUDA_VISIBLE_DEVICES=4 nohup python -u test_flag_ablation.py --dataset cora --suffix flag --atk_suffix seed123 > log/cora/evasion_flag.log 2>&1 & 
CUDA_VISIBLE_DEVICES=3 nohup python -u test_flag_ablation.py --dataset pubmed --suffix flag --atk_suffix pre > log/pubmed/evasion_flag.log 2>&1 & 

'''