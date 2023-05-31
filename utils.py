import os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import torch
import torch.nn.functional as F
import random

np_load_old = np.load
np.aload = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def _fetch_data(iter_dataloader, dataloader):
    """
    Fetches the next set of data and refresh the iterator when it is exhausted.
    Follows python EAFP, so no iterator.hasNext() is used.
    """
    try:
        real_batch = next(iter_dataloader)
    except StopIteration:
        iter_dataloader = iter(dataloader)
        real_batch = next(iter_dataloader)

    return iter_dataloader, real_batch


def save_weights_checkpoint(weights,file):
    '''Saves model when validation loss decrease.'''
    torch.save(weights, file+'_checkpoint.pt')

def save_checkpoint(model,file):
    '''Saves model when validation loss decrease.'''
    torch.save(model.state_dict(), file+'_checkpoint.pt')

def setup_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 
        
def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

# --------------------- Load data ----------------------

def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.
    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.aload(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels

def largest_connected_components(adj, n_components=1):
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep

    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


# ------------------------ Normalize -----------------------
# D^(-0.5) * A * D^(-0.5)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx

def normalize_tensor(sp_adj_tensor,edges=None, sub_graph_nodes=None,sp_degree=None,eps=1e-4, power=-0.5):
    edge_index = sp_adj_tensor.coalesce().indices()
    edge_weight = sp_adj_tensor.coalesce().values()
    shape = sp_adj_tensor.shape
    num_nodes= sp_adj_tensor.size(0)

    row, col = edge_index
    if sp_degree is None:
        # print('None')
        deg = torch.sparse.sum(sp_adj_tensor,1).to_dense().flatten()
    else:
        # print('sp')
        deg = sp_degree
        for i in range(len(edges)):
            idx = sub_graph_nodes[0,i]
            deg[idx] = deg[idx] + edges[i]
        last_deg = torch.sparse.sum(sp_adj_tensor[-1]).unsqueeze(0).data
        deg = torch.cat((deg,last_deg))
        
    deg_inv_sqrt = (deg + eps).pow(power)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    if power == -0.5:
        values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    else:
        values = deg_inv_sqrt[row] * edge_weight
    nor_adj_tensor = torch.sparse.FloatTensor(edge_index, values, shape)
    del edge_index, edge_weight, values, deg_inv_sqrt
    return nor_adj_tensor


def sparse_mx_to_torch_sparse_tensor(sparse_mx,device=None):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape,dtype=torch.float,device=device)


def sparse_tensor_to_torch_sparse_mx(sparse_tensor,device=None):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    row = sparse_tensor.coalesce().indices()[0].detach().cpu().numpy()
    col = sparse_tensor.coalesce().indices()[1].detach().cpu().numpy()
    data = sparse_tensor.coalesce().values().detach().cpu().numpy()
    shape = (sparse_tensor.shape[0],sparse_tensor.shape[1])
    return sp.coo_matrix((data, (row, col)),shape=shape)

# --------------------------------- Sub-graph ------------------------ 

def sample_neighbor(batch, nei_list, num_atks=3):
    nei = torch.tensor([nei_list[i][np.random.randint(len(nei_list[i]))] for i in batch], device=batch.device)
    all_batch = torch.cat((batch,nei))
    batch_n = batch.shape[0]
    all_n = all_batch.shape[0]
    sub_batch = torch.arange(batch_n, device=batch.device)
    nei_batch = torch.arange(batch_n, all_n, device=batch.device)
    if num_atks == 3:
        sub_batch = torch.cat([sub_batch ,sub_batch+all_n, sub_batch+2*all_n])
        nei_batch = torch.cat([nei_batch ,nei_batch+all_n, nei_batch+2*all_n])
    else:
        sub_batch = torch.cat([sub_batch ,sub_batch+all_n])
        nei_batch = torch.cat([nei_batch ,nei_batch+all_n])
    return sub_batch, nei_batch, all_batch



def get_filelist(cur_dir, Filelist, name=''):
    newDir = cur_dir
    if os.path.isfile(cur_dir) and name in cur_dir:
        Filelist.append(cur_dir)
        # Filelist.append(os.path.basename(dir))
    elif os.path.isdir(cur_dir):
        for s in os.listdir(cur_dir):
            if "others" in s:
                continue
            newDir=os.path.join(cur_dir,s)
            get_filelist(newDir, Filelist, name)
    return Filelist


def preproc_adj(adj, lcc=True):
    adj = adj + adj.T
    adj = adj.tolil()
    adj[adj > 1] = 1
    if lcc:
        lcc = largest_connected_components(adj)
        adj = adj[lcc][:, lcc]
        assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"
    # whether to set diag=0?
    adj.setdiag(1)
    adj = adj.astype("float32").tocsr()
    adj.eliminate_zeros()
    return adj, lcc

def lcc_graph(adj, features, labels_np, lcc=True):
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(adj.shape[0])
    adj, lcc = preproc_adj(adj, lcc)
    features = features[lcc]
    labels_np = labels_np[lcc]
    n = adj.shape[0]
    print('Nodes num:',n)
    return adj, features, labels_np, n

def obtain_train_graph(adj, features, labels_np, train_mask):
    train_adj = adj[train_mask][:,train_mask]
    train_features = features[train_mask]
    train_labels = labels_np[train_mask]
    print('train node num', train_adj.shape[0])
    return train_adj, train_features, train_labels

def graph_to_tensor(adj, features, labels_np, device):
    adj_tensor = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    nor_adj_tensor = normalize_tensor(adj_tensor)
    feat = torch.from_numpy(features.toarray().astype('double')).float().to(device)
    labels = torch.LongTensor(labels_np).to(device)
    return adj_tensor, nor_adj_tensor, feat, labels
