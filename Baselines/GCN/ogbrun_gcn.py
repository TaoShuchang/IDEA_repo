
from ogb.nodeproppred import PygNodePropPredDataset
print('OKOK')
from ogb.nodeproppred import Evaluator
import time
import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
# from logg import Logger
from copy import deepcopy


class GCN_bnln(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn=False, use_ln=False,catched=True):
        super(GCN_bnln, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=catched))
        if use_bn:
            print('use BatchNorm')
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        if use_ln:
            print('use LayerNorm')
            self.lns = torch.nn.ModuleList()
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=catched))
            if use_bn:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if use_ln:
                self.lns.append(torch.nn.LayerNorm(hidden_channels))
        if use_ln:
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=catched))
        self.dropout = dropout
        self.use_ln = use_ln
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.use_bn:
            for bn in self.bns:
                bn.reset_parameters()
        if self.use_ln:
            for ln in self.lns:
                ln.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.use_bn:
                x = self.bns[i](x)
            if self.use_ln:
                x = self.lns[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class GCN_bnif(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn=False, use_ln=False, catched=True):
        super(GCN_bnif, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=catched))
        if use_bn:
            self.bns = torch.nn.ModuleList()
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=catched))
            if use_bn:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=catched))

        self.dropout = dropout
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class GCN_bn(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, use_bn=False, use_ln=False,catched=True):
        super(GCN_bn, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=catched))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=catched))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=catched))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


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
    args = parser.parse_args()
    print(args)

    save_file = '/home/taoshuchang/rob_IRM/Defenses/GCN/checkpoint/ogbarxiv_oset/' + args.suffix
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print('device',device)
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root = '/home/taoshuchang/rob_IRM/dataset/',
                                     transform=T.ToSparseTensor())
    print('dataset',dataset)
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data.adj_t = data.adj_t.set_diag(1)   
    data = data.to(device)

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
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

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        best_acc_val = 0
        patience = 200
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        start_time = time.time()
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)
            logger.add_result(run, result)
            # torch.save(model.state_dict(), save_file  + '_checkpoint.pt')  
            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                # print('run', run, 'Epoch', epoch, 'loss',loss,'train',train_acc, 'valid_acc',valid_acc,'test_acc',test_acc)
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss[0]:.4f}, '
                      f'Out: {loss[1].min().item():.4f},'
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
            acc_val = result[1]
            if best_acc_val < acc_val:
                best_acc_val = acc_val
                weights = deepcopy(model.state_dict())
                patience = 200
                print('patience:',patience, 'best_acc_val:',best_acc_val)
            else:
                patience -= 1
        max_ep = logger.print_statistics(run)
        print('max_ep',max_ep)
        model.load_state_dict(weights)
        end_time = time.time()
        during = end_time - start_time
        print('During Time:', during, 'seconds;', during/60, 'minutes;', during/60/60, 'hours;')
        torch.save(model.state_dict(), save_file  + '_checkpoint.pt')  
        model.eval()
        result = test(model, data, split_idx, evaluator)
        print(f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
    # logger.print_statistics()


if __name__ == "__main__":
    print('123')
    main()


'''
nohup python -u ogbrun_gcn.py --suffix testtime_bn --hidden_channels 128 --runs 1 --device 0 > log/ogbarxiv_oset/testtime_bn.log 2>&1 &


nohup python -u ogbrun_gcn.py --suffix ogbgcn_bnif --hidden_channels 128 --runs 1 --device 1 > log/ogbarxiv_oset/1ogbgcn_bnif.log 2>&1 &
nohup python -u ogbrun_gcn.py --suffix ogbgcn_bn --hidden_channels 128 --runs 1 --device 0 > log/ogbarxiv_oset/1ogbgcn_bn.log 2>&1 &
nohup python -u ogbrun_gcn.py --suffix ogbgcn_bnln_no --hidden_channels 128 --runs 1 --device 2 > log/ogbarxiv_oset/1ogbgcn_bnln_no.log 2>&1 &
nohup python -u ogbrun_gcn.py --suffix ogbgcn_bnln_both --hidden_channels 128 --runs 1 --device 3 > log/ogbarxiv_oset/1ogbgcn_bnln_both.log 2>&1 &
nohup python -u ogbrun_gcn.py --suffix ogbgcn_bnln_onlybn --hidden_channels 128 --runs 1 --device 1 > log/ogbarxiv_oset/1ogbgcn_bnln_onlybn.log 2>&1 &
nohup python -u ogbrun_gcn.py --suffix ogbgcn_bnln_onlyln --hidden_channels 128 --runs 1 --device 0 > log/ogbarxiv_oset/1ogbgcn_bnln_onlyln.log 2>&1 &
'''