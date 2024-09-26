from sklearn import metrics
from munkres import Munkres
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.factory import KNNGraph
from torch_geometric.utils import remove_self_loops
import scipy.sparse as sp

def knngraph(pca_embed, num_neighbors):
    kg= KNNGraph(num_neighbors)
    dgl_adj= kg(pca_embed, dist= 'cosine')
    dgl_adj= dgl.remove_self_loop(dgl_adj)
    srt, dst= dgl_adj.edges()[0], dgl_adj.edges()[1]
    adj= torch.zeros(pca_embed.shape[0], pca_embed.shape[0])
    for i, j in zip(srt, dst):
        adj[i,j]=1
#    adj= adj_eye- torch.eye(adj_eye.shape[0])
    g_edges= torch.stack(dgl_adj.edges())
    return adj, g_edges

def laplacian_positional_encoding(g, pos_enc_dim): #两参数分别是邻接矩阵(需要np.array)和需要保留的特征值个数
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """
#    np.fill_diagonal(g, 0) #注意np.fill_diagonal函数对矩阵对角线的填充是就地进行的
    row, col = np.nonzero(g)
    values = g[row, col]
#    A = sp.coo_matrix((values, (row, col)), shape=g.shape)
    # Laplacian
    A = sp.coo_matrix(g)
    N = sp.diags(np.apply_along_axis(np.sum, 1, g).clip(1)**-0.5, dtype= float) #np.clip(1)的作用是将小于1的值指定为1，避免出现0元素影响逆矩阵运算
    L = sp.eye(g.shape[0]) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    lp_matrix = torch.from_numpy(EigVec[:,:pos_enc_dim]).float() 
    return L, lp_matrix

def cluster_acc(y_true, y_pred):
    y_true = y_true.numpy().astype(int)
    y_pred = np.array(y_pred)
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    numclass1 = len(l1)
    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('n_cluster is not valid')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    return acc

def sim(z1: torch.Tensor, z2: torch.Tensor, hidden_norm: bool = True):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())
