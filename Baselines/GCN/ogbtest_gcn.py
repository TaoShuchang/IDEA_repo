from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_sparse import SparseTensor
print('OKOK')
import os,sys
#设置当前工作目录，放再import其他路径模块之前
os.chdir(sys.path[0])
#注意默认工作目录在顶层文件夹，这里是相对于顶层文件夹的目录
sys.path.append("./")
import argparse
import sys
import torch
import torch.nn.functional as F
import csv
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from logg import Logger
from copy import deepcopy
from ogbrun_gcn import GCN_bn, GCN_bnif, GCN_bnln
sys.path.append('../../')
from utils import *



def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item(), out


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    prefile = '../../'
    print('456')
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=5)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--suffix', type=str, default='ogbgcn_bnln_onlyln')
    parser.add_argument('--dataset', type=str, default='ogbarxiv')
    args = parser.parse_args()
    print(args)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print('device',device)

    adj, features, labels_np = load_npz(prefile + f'datasets/{args.dataset}.npz')
    adj, _ = preproc_adj(adj, lcc=False)
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    adj_tensor  = SparseTensor(row=adj_tensor.coalesce().indices()[0], col=adj_tensor.coalesce().indices()[1], value=adj_tensor.coalesce().values(),sparse_sizes=adj_tensor.shape)
    feat = torch.from_numpy(features.toarray().astype('double')).float().to(device)
    labels = torch.LongTensor(labels_np).to(device)
    split = np.aload(prefile + 'splits/split_118/' + args.dataset+ '_oset_split.npy').item()
    train_mask, val_mask, test_mask = split['train'], split['val'], split['test']
    # save_file = '/home/taoshuchang/rob_IRM/Defenses/GCN/checkpoint/ogbarxiv_oset/' + args.suffix
    
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root = '/home/taoshuchang/rob_IRM/dataset/', transform=T.ToSparseTensor())
    data = dataset[0]
    print('data',data)
    data.adj_t = data.adj_t.to_symmetric()
    data.adj_t = data.adj_t.set_diag(1) 
    data = data.to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    seeds = [120]
    save_file = 'checkpoint/' + args.dataset + '_oset/' + args.suffix
    graphpath = prefile + 'final_graphs/' + args.dataset + '/' 
    target_nodes = np.load(prefile + 'splits/target_nodes/' + args.dataset+ '_tar.npy')
    graph_save_file = get_filelist(graphpath, [], name='s_')
    graph_save_file.sort()
    acc_tar_arr = []
    for i in seeds:
        acc_tar_arr_in = []
        graph_name_arr = []
        setup_seed(i)
        print('----------- Seed ',i, '-----------')
        if 'bnif' in args.suffix:
            model = GCN_bnif(data.num_features, args.hidden_channels, dataset.num_classes, args.num_layers, args.dropout, use_bn=True).to(device)
        elif 'bnln' in args.suffix:
            if 'onlyln' in args.suffix:
                model = GCN_bnln(data.num_features, args.hidden_channels, dataset.num_classes, args.num_layers, args.dropout, use_bn=False, use_ln=True).to(device)
            elif 'onlybn' in args.suffix:
                model = GCN_bnln(data.num_features, args.hidden_channels, dataset.num_classes, args.num_layers, args.dropout, use_bn=True, use_ln=False).to(device)
            elif 'both' in args.suffix:
                model = GCN_bnln(data.num_features, args.hidden_channels, dataset.num_classes, args.num_layers, args.dropout, use_bn=True, use_ln=True).to(device)
            else:
                model = GCN_bnln(data.num_features, args.hidden_channels, dataset.num_classes, args.num_layers, args.dropout, use_bn=False, use_ln=False).to(device)
        elif 'bn' in args.suffix:
            model = GCN_bn(data.num_features, args.hidden_channels, dataset.num_classes, args.num_layers, args.dropout).to(device)
        model.load_state_dict(torch.load(save_file + '_checkpoint.pt'))
        model.eval()
        logits = model(feat, adj_tensor)
        logits2 = model(feat,  data.adj_t)
        logits3 = model(data.x, adj_tensor)
        logits4 = model(data.x, data.adj_t)
        val_acc = accuracy(logits[val_mask], labels[val_mask])
        test_acc = accuracy(logits[test_mask], labels[test_mask])
        tar_acc = accuracy(logits[target_nodes], labels[target_nodes])
        print("Validate accuracy {:.4%}".format(val_acc))
        print("Test accuracy {:.4%}".format(test_acc))
        print("Target accuracy {:.4%}".format(tar_acc))

        logits = model(data.x, data.adj_t)
        val_acc = accuracy(logits[val_mask], labels[val_mask])
        test_acc = accuracy(logits[test_mask], labels[test_mask])
        tar_acc = accuracy(logits[target_nodes], labels[target_nodes])
        print("data.x Validate accuracy {:.4%}".format(val_acc))
        print("Test accuracy {:.4%}".format(test_acc))
        print("Target accuracy {:.4%}".format(tar_acc))
        graph_name_arr.append('Clean')
        acc_tar_arr.append(tar_acc)
        
        for graph in graph_save_file:
            graph_name = graph.split('/')[-1]
            graph_name_arr.append(graph_name)
            print('inject attack',graph_name)
            new_adj, new_features, new_labels_np = load_npz(graph)
            new_adj_tensor = sparse_mx_to_torch_sparse_tensor(new_adj).to(device)
            new_adj_tensor = SparseTensor(row=new_adj_tensor.coalesce().indices()[0], col=new_adj_tensor.coalesce().indices()[1], value=new_adj_tensor.coalesce().values(),sparse_sizes=new_adj_tensor.shape)
            new_feat = torch.from_numpy(new_features.toarray().astype('double')).float().to(device)
            logits = model(new_feat, new_adj_tensor)
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


if __name__ == "__main__":
    print('123')
    main()


'''
nohup python -u ogbtest_gcn.py --suffix ogbgcn_bnif --hidden_channels 128 --runs 1 --device 1 > log/ogbarxiv_oset/evasion_ogbgcn_bnif.log 2>&1 &
nohup python -u ogbtest_gcn.py --suffix ogbgcn_bn --hidden_channels 128 --runs 1 --device 0 > log/ogbarxiv_oset/evasion_ogbgcn_bn.log 2>&1 &
nohup python -u ogbtest_gcn.py --suffix ogbgcn_bnln_onlyln --hidden_channels 128 --runs 1 --device 3 > log/ogbarxiv_oset/evasion_ogbgcn_bnln_onlyln.log 2>&1 &
nohup python -u ogbtest_gcn.py --suffix ogbgcn_bnln_onlybn --hidden_channels 128 --runs 1 --device 5 > log/ogbarxiv_oset/evasion_ogbgcn_bnln_onlybn.log 2>&1 &
nohup python -u ogbtest_gcn.py --suffix ogbgcn_bnln_both --hidden_channels 128 --runs 1 --device 0 > log/ogbarxiv_oset/evasion_ogbgcn_bnln_both.log 2>&1 &
nohup python -u ogbtest_gcn.py --suffix ogbgcn_bnln_no --hidden_channels 128 --runs 1 --device 3 > log/ogbarxiv_oset/evasion_ogbgcn_bnln_no.log 2>&1 &
'''