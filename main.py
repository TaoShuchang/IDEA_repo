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
from utils import *
from utils import _fetch_data

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader


def training(model, adj_tensor, nor_adj_tensor, feat, labels, atk_idx, train_mask, val_mask, test_mask, nei_list, pert_tensor, col_idx, use_tr_idx):
    torch_dataset = Data.TensorDataset(torch.LongTensor(train_mask))
    batch_x = torch.tensor(train_mask)
    # batch_size = opts['batch_size'] if opts['batch_size'] < train_mask.shape[0] else 64
    batch_loader = Data.DataLoader(dataset=torch_dataset, batch_size=opts['batch_size'], num_workers=24)
    iter_loader = iter(batch_loader)
    optimizer_aug = torch.optim.AdamW([{'params':model.gl.parameters(), 'lr':args.lr_a}, {'params':model.fl.parameters(), 'lr':args.lr_f}])
    optimizer_env = torch.optim.AdamW(model.infenv.parameters(), lr=args.lr_e)
    best_acc_val = 0
    cur_patience = args.patience
    for iteration in range(args.st_epoch, args.epochs + 1):
        model.train()
        for m in range(args.T):
            if opts['batch_size'] < train_mask.shape[0]:
                iter_loader, batch = _fetch_data(iter_dataloader=iter_loader, dataloader=batch_loader)
                batch_x = batch[0]
            total_loss, inv_loss, pearson_loss, irm_loss, logvar, z_max = model.all_update_env_nei(feat, adj_tensor, nor_adj_tensor, labels, atk_idx, batch_x, nei_list, pert_tensor, col_idx, use_tr_idx)
            if m == 0:
                model.optimizer.zero_grad()
                total_loss.backward()
                model.optimizer.step()
            elif m == 1:
                gen_loss = - inv_loss
                optimizer_aug.zero_grad()
                gen_loss.backward()
                optimizer_aug.step()
            else:
                infenv_loss = pearson_loss 
                optimizer_env.zero_grad()
                infenv_loss.backward()
                optimizer_env.step()
        model.eval()
        logits = model.predict(feat, nor_adj_tensor)
        train_acc = accuracy(logits[train_mask], labels[train_mask])
        val_acc = accuracy(logits[val_mask], labels[val_mask])
        if iteration % 200 == 0:
            if args.T == 2:
                print("IT {:05d} | TotalLoss {:.4f} | GenLoss {:.4f} | zMax {:.4f} | TrainAcc {:.5f} | ValAcc: {:.5f} ".format(
                    iteration, total_loss, gen_loss, z_max, train_acc, val_acc))
            else:
                print("IT {:05d} | TotalLoss {:.4f} | GenLoss {:.4f} | InfenvLoss {:.4f} | zMax {:.4f} | TrainAcc {:.5f} | ValAcc: {:.5f} ".format(
                    iteration, total_loss, gen_loss, infenv_loss, z_max, train_acc, val_acc))
        if val_acc > best_acc_val:
            best_acc_val = val_acc
            weights = deepcopy(model.state_dict())
            cur_patience = args.patience
        else:
            cur_patience =  cur_patience - 1
            # print('Early stopping cur_patience:', cur_patience)
        
        if cur_patience <= 0:
            break

    model.load_state_dict(weights)
    model.eval()
    logits = model.predict(feat, nor_adj_tensor)
    logp = F.log_softmax(logits, dim=1)
    val_acc = accuracy(logp[val_mask], labels[val_mask])
    test_acc = accuracy(logp[test_mask], labels[test_mask])

    print("Validate accuracy {:.4%}".format(val_acc))
    print("Test accuracy {:.4%}".format(test_acc))

    return test_acc



def main(opts):
    dataset= opts['dataset']
    suffix = opts['suffix']
    prefile = './'
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print('device', device)
    device = torch.device(device)

    save_file = 'checkpoint/' + args.dataset + suffix
    graph_file = f'attacked_graphs/poison/{args.dataset}/meta_' 
    nei_list = np.aload(prefile + 'splits/neighbors/' + args.dataset + '_nei.npy')
    target_nodes = np.load(prefile + 'splits/target_nodes/' + args.dataset+ '_tar.npy')

    # if args.atk_idea == 0:
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
    adj.setdiag(1)
    adj_tensor, nor_adj_tensor, feat, labels = graph_to_tensor(adj, features, labels_np, device)
    if n > 100000:
        pert_adj, _, use_tr_mask = load_npz(prefile + f'splits/perturbation/{args.dataset}_pert.npz')
        pert_tensor = sparse_mx_to_torch_sparse_tensor(pert_adj).to(device)
        col_idx = torch.LongTensor(train_mask.repeat(args.num_sample)).reshape(1,args.num_sample*train_mask.shape[0]).to(device)
        use_tr_idx = torch.LongTensor(use_tr_mask).to(device)
        tr_n, use_tr_n = train_mask.shape[0], use_tr_mask.shape[0]        
    else:
        pert_tensor = torch.ones(n, n, dtype=torch.int, device=device) - adj_tensor.to_dense() - adj_tensor.to_dense()
        col_idx = torch.arange(0, n).unsqueeze(1).repeat(1, args.num_sample)
        use_tr_idx = None
        tr_n, use_tr_n = n, n
    
    atk_idx = torch.diag(torch.arange(args.num_atks)).to(device)
    acc_test_arr = []
    atk_flag = 0
    pert_rate = np.arange(0.0, 0.3, 0.05)
    # pert_rate = np.arange(0.1, 0.2, 0.05)
    # pert_rate = np.arange(0.2, 0.3, 0.05)
    # pert_rate = np.array([0.25])
    for pert in pert_rate:
        print('------------------- Perturbation', pert, '-------------------')
        if pert > 0:
            if dataset == 'ogbarxiv':
                pert_rate = np.array([0])
                break
            else:
                adj_tensor = torch.load(graph_file + f'{pert:.2f}.pt').to(device)
                # print('adj',new_adj.to_dense().sum())
                adj = sparse_tensor_to_torch_sparse_mx(adj_tensor)
            adj.setdiag(1)
            adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
            nor_adj_tensor = normalize_tensor(adj_tensor)
            atk_flag = 1
        
        acc_test_arr_in = []
        seeds = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
        for i in seeds:
            setup_seed(i)
            print('----------- Seed ',i, '-----------')
            
            model = IDEA(features.shape[1], args.hidden_channels, labels.max().item()+1, args.dropout, tr_n=tr_n, use_tr_n=use_tr_n, n=n, opts=opts, device=device).to(device)
            
            test_acc = training(model, adj_tensor, nor_adj_tensor, feat, labels, atk_idx, train_mask, val_mask, test_mask, nei_list, pert_tensor, col_idx, use_tr_idx)
            if atk_flag == 0:
                torch.save(model.state_dict(), save_file + f'_pert{pert:.2f}_seed' + str(i) + '_checkpoint.pt')  
            print("Test accuracy {:.4f}".format(test_acc))
            acc_test_arr.append(test_acc)
            acc_test_arr_in.append(test_acc)
        file = 'log/' + args.dataset + '/' + args.dataset + '.csv'
        
        nseed = len(seeds)
        ncol = int(len(acc_test_arr_in)/nseed)
        acc_test_arr_in = np.array(acc_test_arr_in).reshape(nseed, ncol) * 100
        acc_test_f_in = np.concatenate((acc_test_arr_in, acc_test_arr_in.mean(0).reshape(1, ncol), acc_test_arr_in.std(0).reshape(1, ncol)))
        print('acc_test_arr', acc_test_arr_in.shape)
        dataframe_test =  pd.DataFrame(acc_test_f_in[-2:])
        with open(file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['=====',args.suffix, f'_pert{pert:.2f}_seed','====='])
            writer.writerow(['---Test ACC---'])
        # dataframe = pd.DataFrame({u'graph_name_arr':graph_name_arr, u'acc_test':acc_test_arr, u'acc_target':acc_tar_arr})
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
        writer.writerow(['=====',args.suffix, f'all','====='])
        writer.writerow(['---Test ACC---'])
    # dataframe = pd.DataFrame({u'graph_name_arr':graph_name_arr, u'acc_test':acc_test_arr, u'acc_target':acc_tar_arr})
    dataframe_test.to_csv(file, mode='a', index=False)



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
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--start-seed', type=int, default=0)
    parser.add_argument('--counter', type=int, default=0, help='counter')
    parser.add_argument('--best_score', type=float, default=0., help='best score')
    parser.add_argument('--st_epoch', type=int, default=0, help='start epoch')

    parser.add_argument('--perturb_size', type=float, default=1e-3)
    parser.add_argument('--test-freq', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='ogbarxiv')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--atk_suffix', type=str, default='seed123')
    # put a layer norm right after input
    parser.add_argument('--layer_norm_first', default=True)
    # put layer norm between layers or not
    parser.add_argument('--use_ln', type=int,default=0)
    parser.add_argument('--batch_size', type=int,default=256)
    parser.add_argument('--patience', type=int,default=500)
    # IDEA
    parser.add_argument('--lambda_inv_risks', type=int,default=10)
    parser.add_argument('--lambda_beta', type=float, default=1e-4)
    parser.add_argument('--enable_bn', type=bool,default=True)
    parser.add_argument('--num_mlp_layers', type=int,default=2)
    parser.add_argument('--num_atks', type=int,default=3)
    parser.add_argument('--lr_f', type=float, default=1e-4,  help='learning rate for features')
    parser.add_argument('--lr_e', type=float, default=1e-4,  help='learning rate for inferring environment')
    parser.add_argument('--hidden_dim_infenv', type=int, default=16)
    parser.add_argument('--env_num', type=int, default=10)
    parser.add_argument('--clf_dropout', type=float, default=0)
    
     # for graph edit model
    parser.add_argument('--K', type=int, default=1,
                        help='num of views for data augmentation')
    parser.add_argument('--T', type=int, default=3,
                        help='steps for graph learner before one step for GNN')
    parser.add_argument('--num_sample', type=int, default=8,
                        help='num of samples for each node with graph edit, attack budget')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='weight for mean of risks from multiple domains')
    parser.add_argument('--lr_a', type=float, default=1e-4,
                        help='learning rate for graph learner with graph edit')
    args = parser.parse_args()
    opts = args.__dict__.copy()
    print('opts', opts)
    main(opts)