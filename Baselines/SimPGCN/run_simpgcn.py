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
from deeprobust.graph.defense import SimPGCN
from deeprobust.graph.data import Dataset
sys.path.append('../../')
from utils import *

setup_seed(123)

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
    save_file = 'checkpoint/' + dataset + '/' + suffix
    graph_file = prefile + f'Attacks/metattack/new_graphs/{args.dataset}/{args.atk_suffix}_'
    # if args.atk_flag == 0:
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
    for pert in np.arange(0, 0.3, 0.05):
        print('------------------- Perturbation', pert, '-------------------')
        if pert > 0:
            if dataset == 'pubmed':
                adj = sp.load_npz(prefile + f'final_graphs/meta/{args.dataset}_meta_adj_{pert:.2f}.npz')
                # adj_tensor = sparse_mx_to_torch_sparse_tensor(new_adj_sp).to(device)
            elif dataset == 'ogbarxiv':
                break
            else:
                new_adj = torch.load(graph_file + f'{pert:.2f}.pt')
                print('adj',new_adj.to_dense().sum())
                adj = sparse_tensor_to_torch_sparse_mx(new_adj)
                # adj_tensor = new_adj.to(device)

        seeds = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
        acc_test_arr = []
        for i in seeds:
            setup_seed(i)
            print('----------- Seed ',i, '-----------')
            model = SimPGCN(nnodes=features.shape[0], nfeat=features.shape[1], nclass=labels_np.max()+1, nhid=16, device=device)
            model = model.to(device)
            model.fit(features, adj, labels_np, train_mask, val_mask, train_iters=2000, patience=500, verbose=True) # train with validation model picking
            torch.save(model.state_dict(), save_file + f'_pert{pert:.2f}_seed' + str(i) + '_checkpoint.pt')  
            # model.test(test_mask)
            # logits = model.predict()
            logits = model.output
            val_acc = accuracy(logits[val_mask], labels[val_mask])
            test_acc = accuracy(logits[test_mask], labels[test_mask])
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
nohup python -u run_simpgcn.py --dataset ogbarxiv --suffix simpgcn_new --atk_suffix seed123 --gpu 0 > log/ogbarxiv/simpgcn_new.log 2>&1 & 
nohup python -u run_simpgcn.py --dataset citeseer --suffix try --atk_suffix seed123 --gpu 1 > log/citeseer/try.log 2>&1 & 
nohup python -u run_simpgcn.py --dataset citeseer --suffix simpgcn --atk_suffix seed123 --gpu 0 > log/citeseer/simpgcn.log 2>&1 & 
nohup python -u run_simpgcn.py --dataset cora --suffix simpgcn --atk_suffix seed123 --gpu 4 > log/cora/simpgcn.log 2>&1 & 
nohup python -u run_simpgcn.py --dataset pubmed --suffix simpgcn --atk_suffix pre --gpu 1 > log/pubmed/simpgcn.log 2>&1 & 
nohup python -u run_simpgcn.py --dataset 10k_ogbproducts --suffix simpgcn --atk_suffix seed123 --gpu 1 > log/10k_ogbproducts/simpgcn.log 2>&1 & 
nohup python -u run_simpgcn.py --dataset 12k_reddit --suffix simpgcn --atk_suffix seed123 --gpu 2 > log/12k_reddit/simpgcn.log 2>&1 & 

'''