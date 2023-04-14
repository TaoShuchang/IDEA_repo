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

from deeprobust.graph.data import Dataset
from deeprobust.graph.data import Dataset
from util import Logger, str2bool
from elasticgnn import get_model
from train_eval import train
from torch_sparse import SparseTensor

sys.path.append('../../')
from utils import *

setup_seed(123)

def main(opts):
    dataset= opts['dataset']
    connect = opts['connect']
    suffix = opts['suffix']

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    print('device', device)
    device = torch.device(device)
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
    adj_tensor, feat, labels = preprocess(adj, features, labels_np, preprocess_adj=False, sparse=True, device=device)
    feat = feat.to_dense()
    adj_t = SparseTensor(row=adj_tensor.coalesce().indices()[0], col=adj_tensor.coalesce().indices()[1], value=adj_tensor.coalesce().values())

    graph_save_file = get_filelist(graphpath, [], name='s_1')
    graph_save_file.sort()
    seeds = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
    

    acc_tar_arr = []
    for i in seeds:
        acc_tar_arr_in = []
        graph_name_arr = []
        setup_seed(i)
        print('----------- Seed ',i, '-----------')
        elastic = get_model(args, nfeat=features.shape[1], nclass=labels_np.max()+1)
        elastic.load_state_dict(torch.load(save_file + f'_pert0.00_seed' + str(i) + '_checkpoint.pt'))
        # print('elastic',elastic)
        # print('feat',feat)
        # print('adj_t',adj_t)
        logits = elastic(feat, adj_t)
        # logits = elastic.predict(feat, adj_tensor)
        val_acc = accuracy(logits[val_mask], labels[val_mask])
        test_acc = accuracy(logits[test_mask], labels[test_mask])
        tar_acc = accuracy(logits[target_nodes], labels[target_nodes])
        graph_name_arr.append('Clean')
        acc_tar_arr.append(tar_acc)
        print("Validate accuracy {:.4%}".format(val_acc))
        print("Test accuracy {:.4%}".format(test_acc))
        print("Target accuracy {:.4%}".format(tar_acc))
        
        
        for graph in graph_save_file:
            graph_name = graph.split('/')[-1]
            graph_name_arr.append(graph_name)
            print('inject attack',graph_name)
            new_adj, new_features, new_labels_np = load_npz(graph)
            new_adj_tensor, new_feat, new_labels = preprocess(new_adj, new_features, new_labels_np, preprocess_adj=False, sparse=True, device=device)
            new_feat = new_feat.to_dense()
            new_adj_t = SparseTensor(row=new_adj_tensor.coalesce().indices()[0], col=new_adj_tensor.coalesce().indices()[1], value=new_adj_tensor.coalesce().values())

            logits = elastic(new_feat, new_adj_t)

            # adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
            # nor_adj_tensor = normalize_tensor(adj_tensor)
            # feat = torch.from_numpy(features.toarray().astype('double')).float().to(device)
            # logits = elastic.predict(feat, adj_tensor)
            logp = F.log_softmax(logits, dim=1)
            acc = accuracy(logp[target_nodes], labels[target_nodes])
            acc_tar_arr.append(acc)
            print("Target acc {:.4%}".format(acc))
            print()    
    file = 'log/' + args.dataset + '/' + args.dataset + '_evasion.csv'
    nseed = len(seeds)
    ncol = int(len(acc_tar_arr)/nseed)
    acc_tar_arr = np.array(acc_tar_arr).reshape(nseed, ncol) * 100
    print('acc_tar_arr', acc_tar_arr.shape)
    print('graph_name_arr',graph_name_arr)
    print('acc_tar_arr.mean(0)',acc_tar_arr.mean(0))
    print('acc_tar_arr.std(0)',acc_tar_arr.std(0))
    dataframe_test =  pd.DataFrame({u'pert_rate':graph_name_arr, u'mean':acc_tar_arr.mean(0), u'std':acc_tar_arr.std(0)})
    with open(file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['=====',args.suffix, f'all','====='])
        writer.writerow(['---Test ACC---'])
    # dataframe = pd.DataFrame({u'graph_name_arr':graph_name_arr, u'acc_test':acc_tar_arr, u'acc_target':acc_tar_arr})
    dataframe_test.to_csv(file, mode='a', index=False)




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='elastic')

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
    
    parser.add_argument('--nepochs', type=int, default=10000, help='number of epochs')
    parser.add_argument('--runs', type=int, default=10, help='runs number')
    parser.add_argument('--atk_flag', type=int, default=0, help='Whether is attack')
    parser.add_argument('--atk_suffix', type=str, default='', help='Whether is attack')

    parser.add_argument('--log_steps', type=int, default=200)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=2000)
  
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--lambda1', type=float, default=3)
    parser.add_argument('--lambda2', type=float, default=3)
    parser.add_argument('--L21', type=str2bool, default=True)
    args = parser.parse_args()
    opts = args.__dict__.copy()
    print('opts', opts)
    main(opts)

'''
CUDA_VISIBLE_DEVICES=2 nohup python -u test_elastic.py --dataset ogbarxiv --suffix elastic --atk_suffix seed123  > log/ogbarxiv/evasion_elastic.log 2>&1 & 
CUDA_VISIBLE_DEVICES=2 nohup python -u test_elastic.py --dataset 12k_reddit --suffix elastic --atk_suffix seed123  > log/12k_reddit/evasion_elastic.log 2>&1 & 
CUDA_VISIBLE_DEVICES=4 nohup python -u test_elastic.py --dataset 10k_ogbproducts --suffix elastic --atk_suffix seed123  > log/10k_ogbproducts/evasion_elastic.log 2>&1 & 
CUDA_VISIBLE_DEVICES=2 nohup python -u test_elastic.py --dataset citeseer --suffix elastic --atk_suffix seed123  > log/citeseer/evasion_elastic.log 2>&1 & 
CUDA_VISIBLE_DEVICES=2 nohup python -u test_elastic.py --dataset cora --suffix elastic --atk_suffix seed123  > log/cora/evasion_elastic.log 2>&1 & 
CUDA_VISIBLE_DEVICES=2 nohup python -u test_elastic.py --dataset pubmed --suffix elastic --atk_suffix pre  > log/pubmed/evasion_elastic.log 2>&1 & 
'''