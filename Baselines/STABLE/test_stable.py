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
from model import GCN, LogReg
from copy import deepcopy
import scipy
from robcon import get_contrastive_emb
from utils_stable import *
sys.path.append('../../')
from utils import *

setup_seed(123)

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
    adj_tensor, feat, labels = preprocess(adj, features, labels_np, preprocess_adj=False, sparse=True, device=device)
    adj_pre = preprocess_adj(features, adj, threshold=args.jt)
    adj_delete = adj - adj_pre
    # _, features = to_tensor(adj, features)
    print('===start getting contrastive embeddings===')
    embeds, _ = get_contrastive_emb(adj_pre, feat.unsqueeze(dim=0).to_dense(), adj_delete=adj_delete, lr=0.001, weight_decay=0.0, nb_epochs=10000, beta=args.beta)
    embeds = embeds.squeeze(dim=0)
    embeds = embeds.to('cpu')
    embeds = to_scipy(embeds)
    adj_clean = preprocess_adj(embeds, adj, jaccard=False, threshold=args.cos)
    embeds = torch.FloatTensor(embeds.todense()).to(device)
    adj_clean = sparse_mx_to_sparse_tensor(adj_clean).to_dense().to(device)
    labels = torch.LongTensor(labels_np).to(device)
    graph_save_file = get_filelist(graphpath, [], name='s_')
    graph_save_file.sort()
    seeds = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
    
    acc_tar_arr = []
    for i in seeds:
        acc_tar_arr_in = []
        graph_name_arr = []
        setup_seed(i)
        print('----------- Seed ',i, '-----------')
        adj_temp = adj_clean.clone()
        gcn = GCN(embeds.shape[1], args.hid, labels_np.max()+1)
        adj_temp = adj_new_norm(adj_temp, args.alpha, device)
       
        gcn.load_state_dict(torch.load(save_file + f'_pert0.00_seed' + str(i) + '_checkpoint.pt'))
        gcn = gcn.to(device)
        gcn.eval()
        logits = gcn(adj_temp,embeds)
        # logits = gcn.predict(feat, adj_tensor)
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
            new_adj_pre = preprocess_adj(new_features, new_adj, threshold=args.jt)
            new_adj_delete = new_adj - new_adj_pre
            print('===start getting contrastive embeddings===')
            new_feat = sparse_mx_to_torch_sparse_tensor(new_features).to(device)
            new_embeds, _ = get_contrastive_emb(new_adj_pre, new_feat.unsqueeze(dim=0).to_dense(), adj_delete=new_adj_delete, lr=0.001, weight_decay=0.0, nb_epochs=10000, beta=args.beta)
            new_embeds = new_embeds.squeeze(dim=0)
            new_embeds = new_embeds.to('cpu')
            new_embeds = to_scipy(new_embeds)
            new_adj_clean = preprocess_adj(new_embeds, new_adj, jaccard=False, threshold=args.cos)
            new_embeds = torch.FloatTensor(new_embeds.todense()).to(device)
            new_adj_clean = sparse_mx_to_sparse_tensor(new_adj_clean).to_dense().to(device)
            new_adj_temp = new_adj_clean.clone()
            new_adj_temp = adj_new_norm(new_adj_temp, args.alpha, device)
            logits = gcn(new_adj_temp,new_embeds)
            # new_adj_tensor = sparse_mx_to_sparse_tensor(new_adj).to_dense().to(device)
            # new_feat = sparse_mx_to_sparse_tensor(new_features).to_dense().to(device)

            # logits = gcn(new_adj_tensor,new_feat)

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

    parser.add_argument('--threshold', type=float, default=1,  help='threshold')
    parser.add_argument('--jt', type=float, default=0.03,  help='jaccard threshold')
    parser.add_argument('--cos', type=float, default=0.1,  help='cosine similarity threshold')
    parser.add_argument('--k', type=int, default=3 ,  help='add k neighbors')
    parser.add_argument('--alpha', type=float, default=0.3,  help='add k neighbors')
    parser.add_argument('--beta', type=float, default=2,  help='the weight of selfloop')
    args = parser.parse_args()
    opts = args.__dict__.copy()
    print('opts', opts)
    main(opts)

'''
CUDA_VISIBLE_DEVICES=3 nohup python -u test_stable.py --dataset ogbarxiv --suffix stable --atk_suffix seed123 --gpu 0 > log/ogbarxiv/evasion_stable.log 2>&1 & 
CUDA_VISIBLE_DEVICES=3 nohup python -u test_stable.py --dataset 12k_reddit --suffix stable --atk_suffix seed123 --gpu 0 > log/12k_reddit/evasion_stable.log 2>&1 & 
CUDA_VISIBLE_DEVICES=2 nohup python -u test_stable.py --dataset 10k_ogbproducts --suffix stable --atk_suffix seed123 --gpu 0 > log/10k_ogbproducts/evasion_stable.log 2>&1 & 
CUDA_VISIBLE_DEVICES=3 nohup python -u test_stable.py --dataset citeseer --suffix stable --atk_suffix seed123 --gpu 0 > log/citeseer/evasion_stable_nopre.log 2>&1 & 
CUDA_VISIBLE_DEVICES=0 nohup python -u test_stable.py --dataset cora --suffix stable --atk_suffix seed123 --gpu 0 > log/cora/evasion_stable.log 2>&1 & 
CUDA_VISIBLE_DEVICES=1 nohup python -u test_stable.py --dataset pubmed --suffix stable --atk_suffix pre --gpu 0 > log/pubmed/evasion_stable.log 2>&1 & 

'''