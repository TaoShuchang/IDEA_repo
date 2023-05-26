import os,sys
# os.chdir(sys.path[0])
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from audtorch.metrics.functional import pearsonr
# import networks

from utils import *
from gcn import GCN




###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout=0):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers
        self.dropout = dropout

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                h = F.dropout(x, self.dropout, training=self.training)
            return self.linears[self.num_layers - 1](h)


class Infdom(nn.Module):
    def __init__(self, opts, z_dim, class_num, dropout=0):
        super(Infdom, self).__init__()
        self.dropout = dropout
        self.lin1 = nn.Linear(z_dim, opts['hidden_dim_infdom'])
        self.lin2 = nn.Linear(opts['hidden_dim_infdom'], class_num)
        for lin in [self.lin1, self.lin2]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.lin_seq = nn.Sequential(
            self.lin1, nn.ReLU(True), nn.Dropout(dropout), self.lin2, nn.Softmax(dim=1))

    def forward(self, input):
        out = self.lin_seq(input)
        return out

class IDEA(nn.Module):
    """Invariant DEfense against Adversarial attack on graphs"""
    def __init__(self, input_dim, hid_dim, num_classes, dropout, tr_n, use_tr_n, n=2000, opts=None, device=None):
        super(IDEA, self).__init__()
        self.opts = opts
        self.device = device
        num_mlp_layers = opts['num_mlp_layers']
        self.featurizer = GCN(input_dim, hid_dim, hid_dim, opts['num_layers'], dropout).to(device)

        feat_dim = hid_dim
        # VIB archs
        if opts['enable_bn']:
            self.encoder = torch.nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.encoder = torch.nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True)
            )
        self.fc3_mu = nn.Linear(feat_dim, feat_dim) 
        self.fc3_logvar = nn.Linear(feat_dim, feat_dim) 
        self.g_classifier = MLP(num_mlp_layers, hid_dim, hid_dim, num_classes, dropout=self.opts['clf_dropout'])
        self.gd_classifier = MLP(num_mlp_layers, hid_dim+opts['dom_num'], hid_dim+opts['dom_num'], num_classes, dropout=self.opts['clf_dropout'])
        
        
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.g_classifier.parameters()) + list(
                self.gd_classifier.parameters()) + list(self.encoder.parameters()) + list(
                self.fc3_mu.parameters()) + list(self.fc3_logvar.parameters()),
            lr=self.opts["lr"],
            weight_decay=self.opts['weight_decay']
        )
        self.n = n
        self.gl = Graph_Editer(tr_n, use_tr_n, n, device)
        self.fl = Feat_Editer((n, input_dim), opts['perturb_size'], device)
        self.infdom = Infdom(opts, z_dim=hid_dim+opts['num_atks'], class_num=opts['dom_num'], dropout=dropout).cuda()

    def encoder_fun(self, res_feat):
        latent_z = self.encoder(res_feat)
        mu = self.fc3_mu(latent_z)
        logvar = self.fc3_logvar(latent_z)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar / 2)
            eps = torch.randn_like(std)
            return torch.add(torch.mul(std, eps), mu)
        else:
            return mu


    def all_update_dom_nei(self, feat, adj_tensor, nor_adj_tensor, labels, atk_idx, batch, nei_list, pert_tensor, col_idx, use_tr_idx):
        nei_batch, sub_batch, batch = sample_neighbor(batch, nei_list)
        atk_idx = atk_idx.repeat_interleave(batch.shape[0], dim=0)
        z = self.featurizer(feat, nor_adj_tensor)
        # structure
        atk_nor_adj_tensor = self.gl(adj_tensor, pert_tensor, col_idx, self.n, self.opts['num_sample'], use_tr_idx)
        atk_z_struc = self.featurizer(feat, atk_nor_adj_tensor)
        # attributes
        atk_feat = self.fl(feat)
        atk_z_attr = self.featurizer(atk_feat, nor_adj_tensor)

        ori_all_z = torch.cat((z[batch], atk_z_attr[batch], atk_z_struc[batch]))
        all_labels = torch.cat((labels[batch], labels[batch], labels[batch]))
        mu, logvar = self.encoder_fun(ori_all_z)
        all_z = self.reparameterize(mu, logvar)

        domain_class = self.infdom(torch.cat([all_z.detach(), atk_idx],1))
        # calculate loss by parts
        all_pred = self.g_classifier(all_z)
        inv_loss = F.cross_entropy(all_pred[sub_batch], all_labels[sub_batch])
        dom_loss = F.cross_entropy(self.gd_classifier(torch.cat([all_z[sub_batch], domain_class[sub_batch]], 1)), all_labels[sub_batch])
        irm_loss = (inv_loss - dom_loss) ** 2
        inv_loss_nei = F.cross_entropy(all_pred[nei_batch], all_labels[sub_batch])
        dom_loss_nei = F.cross_entropy(self.gd_classifier(torch.cat([all_z[nei_batch], domain_class[sub_batch]], 1)), all_labels[sub_batch])
        nei_loss = (inv_loss_nei - dom_loss_nei) ** 2
        # use beta to balance the info loss.

        total_loss = inv_loss + dom_loss + self.opts['alpha'] * irm_loss + self.opts['alpha'] * nei_loss
        tr = self.compute_dom_vect(all_pred[sub_batch], all_labels[sub_batch], ori_all_z[sub_batch], domain_class[sub_batch], self.device)
        pearson_arr = torch.cat([pearsonr(tr[i], tr[j]) for j in range(len(tr)) for i in range(j+1, len(tr))])
        pearson_loss = pearson_arr.sum()
        return total_loss, inv_loss, pearson_loss, irm_loss, logvar, all_z.max().item()
    

    def compute_dom_vect(self, pred, labels, all_z, domain_class, device):
        dom_label = domain_class.argmax(1)
        eps_num = 100
        tr = []
        z_num, z_dim = all_z.shape
        eps = torch.randn(eps_num, device=device)/100
        # x = torch.randn(z_dim,1).to(device)
        for idx in torch.unique(dom_label):
            domidx = torch.where(dom_label==idx)[0]
            XXx = (all_z[domidx].T.mul(pred[domidx, labels[domidx]] - 1)).sum(1)
            Xeps = all_z[domidx].sum(0)*eps.sum()
            tr.append(XXx - Xeps)
        return tr


    def rand_gen(self, feat, adj_tensor, perturb_size, num_sample, feat_budget):
        n, nfeat = feat.shape
        rand_pert_feat = torch.FloatTensor(*feat.shape).uniform_(-perturb_size, perturb_size).to(feat.device)
        P = torch.softmax(torch.randn(feat.shape), dim=1)
        S = torch.multinomial(P, num_samples=feat_budget)  # [n, s]
        M = torch.zeros(n, nfeat, dtype=torch.float).to(feat.device)
        row_idx = torch.arange(0, n).unsqueeze(1).repeat(1, feat_budget)
        M[row_idx, S] = 1.
        randatk_feat = feat + M * rand_pert_feat
        
        A = adj_tensor.to_dense()
        rand_pert_adj = torch.randn(*adj_tensor.shape).to(feat.device)
        A_c = torch.ones(n, n, dtype=torch.int).to(self.device) - A
        P = torch.softmax(rand_pert_adj, dim=0)
        S = torch.multinomial(P, num_samples=num_sample)  # [n, s]
        M = torch.zeros(n, n, dtype=torch.float).to(feat.device)
        col_idx = torch.arange(0, n).unsqueeze(1).repeat(1, num_sample)
        M[S, col_idx] = 1.
        randatk_adj_tensor = A + M * (A_c - A)

        randatk_nor_adj_tensor = normalize_tensor(randatk_adj_tensor.to_sparse())
        return randatk_feat, randatk_nor_adj_tensor
    
    def predict(self, feat, nor_adj_tensor):
        z = self.featurizer(feat, nor_adj_tensor)
        # mu, logvar = self.encoder_fun(z[batch])
        mu, logvar = self.encoder_fun(z)
        z = self.reparameterize(mu, logvar)
        y = self.g_classifier(z)
        return y


class Graph_Editer(nn.Module):
    def __init__(self, tr_n, use_tr_n, n, device):
        super(Graph_Editer, self).__init__()
        self.B = nn.Parameter(torch.FloatTensor(tr_n, use_tr_n))
        self.tr_n = tr_n
        self.n = n
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.B)

    def forward(self, adj_tensor, pert_tensor, col_idx, n, num_sample, use_tr_idx=None):
        if n > 100000:
            return self.sparse_forward(adj_tensor, pert_tensor, col_idx, n, num_sample, use_tr_idx)
        else:
            return self.dense_forward(adj_tensor, pert_tensor, col_idx, n, num_sample)
        
    def dense_forward(self, adj_tensor, pert_tensor, col_idx, n, num_sample):
        P = torch.softmax(self.B, dim=0)
        S = torch.multinomial(P, num_samples=num_sample)  # [n, s]
        M = torch.zeros(n, n, dtype=torch.float, device=self.device)
        M[S, col_idx] = 1.
        C = adj_tensor.to_dense() + M * pert_tensor
        return normalize_tensor(C.to_sparse())
    
    def sparse_forward(self, adj_tensor, pert_tensor, col_idx_sp, n, num_sample, use_tr_idx):
        tr_n = self.tr_n
        P = torch.softmax(self.B, dim=0)
        S_0_sp = torch.multinomial(P, num_samples=num_sample).flatten() 
        S_sp = use_tr_idx[S_0_sp].reshape(1, num_sample*tr_n)
        indices = torch.cat([S_sp, col_idx_sp],dim=0)
        values = torch.ones(tr_n*num_sample, device=self.device)
        M = torch.sparse.FloatTensor(indices, values, torch.Size([n,n]))
        return normalize_tensor(adj_tensor + M.mul(pert_tensor))


class Feat_Editer(nn.Module):
    def __init__(self, feat_shape, perturb_size, device):
        super(Feat_Editer, self).__init__()
        self.feat_perturb = nn.Parameter(torch.FloatTensor(torch.Size(feat_shape)).uniform_(-perturb_size, perturb_size).to(device))
        self.device = device

    def reset_parameters(self):
        nn.init.uniform_(self.B)

    # def forward(self, edge_index, n, num_sample, k):
    def forward(self, feat):
        return feat + self.feat_perturb
