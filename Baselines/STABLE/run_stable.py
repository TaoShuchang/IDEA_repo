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

def train(model, optim, adj, embeds, labels, train_mask, val_mask, test_mask, loss, verbose=True):
    best_loss_val = 9999
    best_acc_val = 0
    cur_patience = args.patience
    for epoch in range(args.nepochs):
        flag = 0
        model.train()
        logits = model(adj, embeds)
        l = loss(logits[train_mask], labels[train_mask])
        optim.zero_grad()
        l.backward()
        optim.step()
        acc = evaluate(model, adj, embeds, labels, val_mask)
        val_loss = loss(logits[val_mask], labels[val_mask])
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            weights = deepcopy(model.state_dict())
            flag = 1
        if acc > best_acc_val:
            best_acc_val = acc
            weights = deepcopy(model.state_dict())
            flag = 1
        cur_patience = args.patience if flag==1 else cur_patience - 1
        if cur_patience <= 0:
            break
        if verbose:
            if epoch % 10 == 0:
                print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}".format(epoch, l.item(), acc))
    model.load_state_dict(weights)
    # torch.save(weights, './save_model/%s_%s_%s.pth' % (args.attack, args.dataset, args.ptb_rate))
    acc = evaluate(model, adj, embeds, labels, test_mask)
    print("===== Test Accuracy {:.4f}=====".format(acc))
    return acc

def main(opts):
    dataset= opts['dataset']
    connect = opts['connect']
    suffix = opts['suffix']
    lr = opts['lr']
    weight_decay = opts['decay']

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
    train_mask, val_mask, test_mask = idx_to_mask(train_mask, n).to(device), idx_to_mask(val_mask, n).to(device), idx_to_mask(test_mask, n).to(device)
    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)
    loss = nn.CrossEntropyLoss()
    
    for pert in [args.pert]:
    # for pert in np.arange(0, 0.3, 0.05):
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
        
                
        print('===start preprocessing the graph===')
        if args.dataset == 'polblogs':
            args.jt = 0
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
                
        seeds = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
        acc_test_arr = []
        for i in seeds:
            setup_seed(i)
            print('----------- Seed ',i, '-----------')
            adj_temp = adj_clean.clone()
            model = GCN(embeds.shape[1], args.hid, labels_np.max()+1)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            adj_temp = adj_new_norm(adj_temp, args.alpha, device)
            test_acc = train(model, optimizer, adj_temp, embeds,labels, train_mask, val_mask, test_mask, loss)
            print("Test accuracy {:.4f}".format(test_acc))
            acc_test_arr.append(test_acc)
            torch.save(model.state_dict(), save_file + f'_pert{pert:.2f}_seed' + str(i) + '_checkpoint.pt')  
            
            
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
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--hid', default=16, type=int, help='hidden dimension')
    parser.add_argument('--decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--patience', default=500, type=int, help='patience')
    parser.add_argument('--pert', default=0.00, type=float, help='perturbation rate')
    
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
CUDA_VISIBLE_DEVICES=2 nohup python -u run_stable.py --dataset ogbarxiv --suffix stable --atk_suffix seed123 --gpu 0 > log/ogbarxiv/stable.log 2>&1 & 
CUDA_VISIBLE_DEVICES=2 nohup python -u run_stable.py --dataset 12k_reddit --suffix stable --atk_suffix seed123 --gpu 0 > log/12k_reddit/stable.log 2>&1 & 
CUDA_VISIBLE_DEVICES=4 nohup python -u run_stable.py --dataset 10k_ogbproducts --suffix stable --atk_suffix seed123 --gpu 0 > log/10k_ogbproducts/stable.log 2>&1 & 

nohup python -u run_stable.py --dataset citeseer --suffix stable --atk_suffix seed123 --gpu 4 > log/citeseer/stable.log 2>&1 & 
nohup python -u run_stable.py --dataset cora --suffix stable --atk_suffix seed123 --gpu 4 > log/cora/stable.log 2>&1 & 
CUDA_VISIBLE_DEVICES=0 nohup python -u run_stable.py --dataset pubmed --suffix stable --pert 0.25 --atk_suffix pre --gpu 0 > log/pubmed/stable_0.25.log 2>&1 & 

'''