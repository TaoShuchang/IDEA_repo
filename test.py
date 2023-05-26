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
import torch.utils.data as Data
from copy import deepcopy
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from idea import IDEA
sys.path.append('..')
from utils import *
from utils import _fetch_data

def main(opts):
    dataset= opts['dataset']
    suffix = opts['suffix']
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print('device', device)
    device = torch.device(device)
    prefile = './'
    save_file = 'checkpoint/' + dataset + suffix
    graphpath = prefile + 'attacked_graphs/evasion/' + args.dataset + '/' 
    log_file = 'log/' + args.dataset + '_' + args.dataset + '_evasion.csv'
    target_nodes = np.load(prefile + 'splits/target_nodes/' + args.dataset+ '_tar.npy')

    # if args.atk_idea == 0:
    if args.dataset in ['citeseer','cora']:
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
    adj.setdiag(1)
    adj_tensor, nor_adj_tensor, feat, labels = graph_to_tensor(adj, features, labels_np, device)
    if n > 100000:
        _, _, use_tr_mask = load_npz(prefile + f'splits/perturbation/{args.dataset}_pert.npz')
        tr_n, use_tr_n = train_mask.shape[0], use_tr_mask.shape[0]        
    else:
        tr_n, use_tr_n = n, n
    
    graph_save_file = get_filelist(graphpath, [], name='s_')
    graph_save_file.sort()

    seeds = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
    
    
    acc_tar_arr = []
    for i in seeds:
        graph_name_arr = []
        setup_seed(i)
        print('----------- Seed ',i, '-----------')
        model = IDEA(features.shape[1], args.hidden_channels, labels.max().item()+1, args.dropout, tr_n=tr_n, use_tr_n=use_tr_n, n=n, opts=opts, device=device).to(device)
        model.load_state_dict(torch.load(save_file + f'_pert0.00_seed' + str(i) + '_checkpoint.pt'))
        model.eval()

        logits = model.predict(feat, nor_adj_tensor)
        logp = F.log_softmax(logits, dim=1)
        val_acc = accuracy(logp[val_mask], labels[val_mask])
        test_acc = accuracy(logp[test_mask], labels[test_mask])
        tar_acc = accuracy(logp[target_nodes], labels[target_nodes])
        graph_name_arr.append('Clean')
        acc_tar_arr.append(tar_acc)
        # print("Validate accuracy {:.4%}".format(val_acc))
        # print("Test accuracy {:.4%}".format(test_acc))
        print("Target accuracy {:.4%}".format(tar_acc))
        
        for graph in graph_save_file:
            graph_name = graph.split('/')[-1]
            graph_name_arr.append(graph_name)
            print('inject attack',graph_name)
            new_adj, new_features, labels_np = load_npz(graph)
            print('new_adj',new_adj.shape)
            print('new_features',new_features.shape)
            
            new_adj_tensor = sparse_mx_to_torch_sparse_tensor(new_adj).to(device)
            new_nor_adj_tensor = normalize_tensor(new_adj_tensor)
            new_feat = torch.from_numpy(new_features.toarray().astype('double')).float().to(device)
            new_logits = model.predict(new_feat, new_nor_adj_tensor)
            new_logp = F.log_softmax(new_logits, dim=1)
            tar_acc = accuracy(new_logp[target_nodes], labels[target_nodes])
            acc_tar_arr.append(tar_acc)

            print("Target acc {:.4%}".format(tar_acc))
            print()    
    
    nseed = len(seeds)
    ncol = int(len(acc_tar_arr)/nseed)
    acc_tar_arr = np.array(acc_tar_arr).reshape(nseed, ncol) * 100
    print('acc_tar_arr', acc_tar_arr.shape)
    dataframe_test =  pd.DataFrame({u'pert_rate':graph_name_arr, u'mean':acc_tar_arr.mean(0), u'std':acc_tar_arr.std(0)})
    with open(log_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['=====',args.suffix, f'all','====='])
        writer.writerow(['---Target ACC---'])

    dataframe_test.to_csv(log_file, mode='a', index=False)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='IDEA')

    # configure
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--counter', type=int, default=0, help='counter')
    parser.add_argument('--best_score', type=float, default=0., help='best score')
    parser.add_argument('--st_epoch', type=int, default=0, help='start epoch')

    parser.add_argument('--perturb_size', type=float, default=1e-3)
    parser.add_argument('--m', type=int, default=3)
    parser.add_argument('--attack', type=str, default='flag')
    parser.add_argument('--dataset', type=str, default='ogbarxiv')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--atk_suffix', type=str, default='', help='Whether is attack')
    # put a layer norm right after input
    parser.add_argument('--layer_norm_first', default=True)
    # put layer norm between layers or not
    parser.add_argument('--use_ln', type=int,default=0)
    parser.add_argument('--batch_size', type=int,default=256)
    parser.add_argument('--patience', type=int,default=500)
    # IDEA
    parser.add_argument('--alpha', type=int,default=10)
    parser.add_argument('--enable_bn', type=bool,default=True)
    parser.add_argument('--num_mlp_layers', type=int,default=2)
    parser.add_argument('--num_atks', type=int,default=3)
    parser.add_argument('--lr_f', type=float, default=1e-4,  help='learning rate for features')
    parser.add_argument('--lr_e', type=float, default=1e-4,  help='learning rate for inferring environment')
    parser.add_argument('--hidden_dim_infdom', type=int, default=16)
    parser.add_argument('--dom_num', type=int, default=10)
    parser.add_argument('--clf_dropout', type=float, default=0)
    
     # for graph edit model
    parser.add_argument('--K', type=int, default=1,
                        help='num of views for data augmentation')
    parser.add_argument('--T', type=int, default=3,
                        help='steps for graph learner before one step for GNN')
    parser.add_argument('--num_sample', type=int, default=8,
                        help='num of samples for each node with graph edit, attack budget')
    parser.add_argument('--lr_a', type=float, default=1e-4,
                        help='learning rate for graph learner with graph edit')
    args = parser.parse_args()
    opts = args.__dict__.copy()
    print('opts', opts)
    main(opts)
