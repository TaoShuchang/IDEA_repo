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
from util import Logger, str2bool
from elasticgnn import get_model
from train_eval import train
from torch_sparse import SparseTensor

sys.path.append('../../')
from utils import *

setup_seed(123)

def training(model, feat, adj_tensor, labels, train_mask, val_mask, test_mask, optimizer):
    best_loss_val = 9999
    best_acc_val = 0
    cur_patience = args.patience
    for epoch in range(args.epochs):
        model.train()
        loss = train(model, feat, adj_tensor, labels, train_mask, optimizer)
        model.eval()
        logits = model(feat, adj_tensor)
        val_acc = accuracy(logits[val_mask], labels[val_mask])
        test_acc = accuracy(logits[test_mask], labels[test_mask])
        val_loss = F.nll_loss(logits[val_mask], labels[val_mask])
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            weights = deepcopy(model.state_dict())
            flag = 1
        if val_acc > best_acc_val:
            best_acc_val = val_acc
            weights = deepcopy(model.state_dict())
            flag = 1
        cur_patience = args.patience if flag==1 else cur_patience - 1
        if cur_patience <= 0:
            break
        if args.log_steps > 0:
            if epoch % args.log_steps == 0:
                print(f'Epoch: {epoch:03d}, '
                    f'Loss: {loss:.4f}, '
                    f'Valid: {100 * val_acc:.2f}% '
                    f'Test: {100 * test_acc:.2f}%')
        
    model.load_state_dict(weights)
    # torch.save(weights, './save_model/%s_%s_%s.pth' % (args.attack, args.dataset, args.ptb_rate))
    test_acc = accuracy(logits[test_mask], labels[test_mask])
    # print("===== Test Accuracy {:.4f}=====".format(test_acc))
    return test_acc

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
    adj_tensor, feat, labels = preprocess(adj, features, labels_np, preprocess_adj=False, sparse=True, device=device)
    feat = feat.to_dense()

    train_mask, val_mask, test_mask = torch.tensor(train_mask).to(device), torch.tensor(val_mask).to(device), torch.tensor(test_mask).to(device)

    acc_test_arr = []
    pert_rate = np.arange(0, 0.3, 0.05)
    for pert in pert_rate:
        print('------------------- Perturbation', pert, '-------------------')
        if pert > 0:
            if dataset == 'pubmed':
                adj = sp.load_npz(prefile + f'final_graphs/meta/{args.dataset}_meta_adj_{pert:.2f}.npz')
                adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
            elif dataset == 'ogbarxiv':
                break
            else:
                adj_tensor = torch.load(graph_file + f'{pert:.2f}.pt').to(device)
                # print('adj',new_adj.to_dense().sum())
                # adj = sparse_tensor_to_torch_sparse_mx(new_adj)
                # adj_tensor = new_adj.to(device)
        
        adj_t = SparseTensor(row=adj_tensor.coalesce().indices()[0], col=adj_tensor.coalesce().indices()[1], value=adj_tensor.coalesce().values())

        seeds = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
        acc_test_arr_in = []
        for i in seeds:
            setup_seed(i)
            print('----------- Seed ',i, '-----------')
            model = get_model(args, nfeat=features.shape[1], nclass=labels_np.max()+1)
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            test_acc = training(model, feat, adj_t, labels, train_mask, val_mask, test_mask, optimizer)
            
            print("Test accuracy {:.4f}".format(test_acc))
            acc_test_arr.append(test_acc)
            acc_test_arr_in.append(test_acc)
            torch.save(model.state_dict(), save_file + f'_pert{pert:.2f}_seed' + str(i) + '_checkpoint.pt')  

        file = 'log/' + args.dataset + '/' + args.dataset + '.csv'
        
        nseed = len(seeds)
        ncol = int(len(acc_test_arr_in)/nseed)
        acc_test_arr_in = np.array(acc_test_arr_in).reshape(nseed, ncol) * 100
        acc_test_f_in = np.concatenate((acc_test_arr_in, acc_test_arr_in.mean(0).reshape(1, ncol), acc_test_arr_in.std(0).reshape(1, ncol)))
        print('acc_test_arr_in', acc_test_arr_in.shape)
        dataframe_test =  pd.DataFrame(acc_test_f_in[-2:])
        with open(file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['=====',args.suffix, f'_pert{pert:.2f}_seed','====='])
            writer.writerow(['---Test ACC---'])
        dataframe_test.to_csv(file, mode='a', index=False)
            
            
    file = 'log/' + args.dataset + '/' + args.dataset + '.csv'
    
    nseed = len(seeds)
    nrow = int(len(acc_test_arr)/nseed)
    acc_test_arr = np.array(acc_test_arr).reshape(nrow, nseed) * 100
    # acc_test_f = np.concatenate((acc_test_arr, acc_test_arr.mean(1).reshape(nrow,1), acc_test_arr.std(0).reshape(nrow, 1)))
    print('acc_test_arr', acc_test_arr.shape)
    dataframe_test =  pd.DataFrame({u'pert_rate':pert_rate, u'mean':acc_test_arr.mean(1), u'std':acc_test_arr.std(1)})
    with open(file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['=====',args.suffix, 'all','====='])
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
CUDA_VISIBLE_DEVICES=2 nohup python -u run_elastic.py --dataset ogbarxiv --suffix elastic --atk_suffix seed123 > log/ogbarxiv/elastic.log 2>&1 & 
CUDA_VISIBLE_DEVICES=1 nohup python -u run_elastic.py --dataset 12k_reddit --suffix elastic --atk_suffix seed123 > log/12k_reddit/elastic.log 2>&1 & 
CUDA_VISIBLE_DEVICES=4 nohup python -u run_elastic.py --dataset 10k_ogbproducts --suffix elastic --atk_suffix seed123 > log/10k_ogbproducts/elastic.log 2>&1 & 

CUDA_VISIBLE_DEVICES=0 nohup python -u run_elastic.py --dataset citeseer --suffix elastic --atk_suffix seed123 > log/citeseer/elastic.log 2>&1 & 
CUDA_VISIBLE_DEVICES=2 nohup python -u run_elastic.py --dataset cora --suffix elastic --atk_suffix seed123 > log/cora/elastic.log 2>&1 & 
CUDA_VISIBLE_DEVICES=3 nohup python -u run_elastic.py --dataset pubmed --suffix elastic --atk_suffix pre > log/pubmed/elastic.log 2>&1 & 

'''