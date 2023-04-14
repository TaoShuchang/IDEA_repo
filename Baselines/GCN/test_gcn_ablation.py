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
from deeprobust.graph.defense import GCN
from deeprobust.graph.data import Dataset
sys.path.append('../../')
from utils import *

setup_seed(123)

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

def main(opts):
    dataset= opts['dataset']
    connect = opts['connect']
    suffix = opts['suffix']

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print('device', device)
    device = torch.device('cpu')
    prefile = '../../'
    save_file = 'checkpoint/' + dataset + '/' + suffix
    graphpath = prefile + 'final_graphs/' + args.dataset + '/' 
    target_nodes = np.load(prefile + 'splits/target_nodes/' + args.dataset+ '_tar.npy')

    if args.dataset in ['citeseer','cora', 'pubmed']:
        data = Dataset(root=prefile+'datasets/', name=args.dataset, setting='nettack')
        adj, features, labels_np = data.adj, data.features, data.labels
        train_mask, val_mask, test_mask = data.idx_train, data.idx_val, data.idx_test
        n = adj.shape[0]
    else:
        adj, features, labels_np = load_npz(prefile + f'datasets/{args.dataset}.npz')
        adj, features, labels_np, n = lcc_graph(adj, features, labels_np)
        split = np.aload(prefile + 'splits/split_118/' + args.dataset+ '_split.npy').item()
        train_mask, val_mask, test_mask = split['train'], split['val'], split['test']
    
    labels = torch.LongTensor(labels_np).to(device)
    graph_save_file = get_filelist(graphpath, [], name='s_')
    graph_save_file.sort()
    seeds = [123]
    
    acc_tar_arr = []
    for i in seeds:
        acc_tar_arr_in = []
        graph_name_arr = []
        setup_seed(i)
        print('----------- Seed ',i, '-----------')
        gcn = GCN(nfeat=features.shape[1], nhid=args.hid, nclass=labels.max().item() + 1, dropout=0.5, device=device)
        gcn.load_state_dict(torch.load(save_file + f'_pert0.00_seed' + str(i) + '_checkpoint.pt'))
        gcn = gcn.to(device)
        logits = gcn.predict(features, adj, output_hidden=True)

        clean_z = logits.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        np.save("../../visualization/GCN/clean_z.npy" ,clean_z)
        np.save("../../visualization/GCN/labels.npy" ,labels)
        
        n_clean_nodes = logits.size(0)
        
        for graph in graph_save_file:
            graph_name = graph.split('/')[-1]
            graph_name_arr.append(graph_name)
            print('inject attack',graph_name)
            new_adj, new_features, new_labels_np = load_npz(graph)
            new_logits = gcn.predict(new_features, new_adj, output_hidden=True)

            print((logits == new_logits[:n_clean_nodes, :]).all())

            pert_z = new_logits[:n_clean_nodes, :]
            pert_z = pert_z.cpu().detach().numpy()
            np.save(f"../../visualization/GCN/pert_{graph_name[:-4]}.npy" ,pert_z)
            


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='GCN')

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
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--hid', default=16, type=int, help='hidden dimension')
    parser.add_argument('--decay', default=5e-4, type=float, help='weight decay')
    
    parser.add_argument('--nepochs', type=int, default=10000, help='number of epochs')
    parser.add_argument('--runs', type=int, default=10, help='runs number')
    parser.add_argument('--atk_flag', type=int, default=0, help='Whether is attack')
    parser.add_argument('--atk_suffix', type=str, default='', help='Whether is attack')
    args = parser.parse_args()
    opts = args.__dict__.copy()
    print('opts', opts)
    main(opts)

'''
nohup python -u test_gcn.py --dataset ogbarxiv --suffix gcn --atk_suffix seed123 --gpu 1 > log/ogbarxiv/evasion_gcn.log 2>&1 & 
nohup python -u test_gcn.py --dataset 12k_reddit --suffix gcn --atk_suffix seed123 --gpu 0 > log/12k_reddit/evasion_gcn.log 2>&1 & 
nohup python -u test_gcn.py --dataset 10k_ogbproducts --suffix gcn --atk_suffix seed123 --gpu 1 > log/10k_ogbproducts/evasion_gcn.log 2>&1 & 
nohup python -u test_gcn.py --dataset citeseer --suffix gcn --atk_suffix seed123 --gpu 1 > log/citeseer/evasion_gcn.log 2>&1 & 
nohup python -u test_gcn_ablation.py --dataset cora --suffix gcn --atk_suffix seed123 --gpu 0 > log/cora/evasion_gcn.log 2>&1 & 
nohup python -u test_gcn.py --dataset pubmed --suffix gcn --atk_suffix pre --gpu 2 > log/pubmed/evasion_gcn.log 2>&1 & 
'''