import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from utils import sim

def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, tau, adj, cls_adj, cls_thres=0.9,
                     mean: bool = True, hidden_norm: bool = True):
    l1 = nei_con_loss(z1, z2, tau, adj, cls_adj, cls_thres, hidden_norm)
    l2 = nei_con_loss(z2, z1, tau, adj, cls_adj, cls_thres, hidden_norm)
    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()
    return ret

def nei_con_loss(z1: torch.Tensor, z2: torch.Tensor, tau, adj_eye, cls_adj, cls_thres=0.9, hidden_norm: bool = True):
    '''neighbor contrastive loss'''
    adj_eye = torch.tensor(adj_eye, dtype=torch.float)
    adj = adj_eye - torch.diag_embed(adj_eye.diag())  # remove self-loop
    N = adj_eye.shape[0]
    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z1, z1, hidden_norm))
    inter_view_sim = f(sim(z1, z2, hidden_norm))
    loss = (inter_view_sim.diag() + (intra_view_sim.mul(adj).mul(cls_adj >= cls_thres)).sum(1) + (
        inter_view_sim.mul(adj).mul(cls_adj >= cls_thres)).sum(1)) / (
               (inter_view_sim.sum(1) + intra_view_sim.sum(1) - intra_view_sim.diag()))
    return torch.mean(-torch.log(loss))

class AGCLoss(nn.Module):
    def __init__(self, entropy_weight=1.0, temperature=1.0, device= 'cpu'):
        super(AGCLoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss()
        self.lamda = entropy_weight
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = temperature
        self.device= device
    def forward(self, ologits, plogits, adj_eye):
        adj = adj_eye.clone().detach()
        num_neighbors = torch.sum(adj, 1, keepdim=True)  # number of neighbors of each node
        ologits = self.softmax(ologits)
        plogits = self.softmax(plogits)
        d1 = torch.mm(adj, ologits) / num_neighbors  # average cluster representation 
        d2 = torch.mm(adj, plogits) / num_neighbors
        similarity = torch.mm(F.normalize(d1.t(), p=2, dim=1), F.normalize(d2, p=2, dim=0))/self.temperature  # a k*k tensor
        loss_ce = self.xentropy(similarity, torch.arange(similarity.size(0)).to(self.device))  
        d1 = d1.sum(0).view(-1)
        d1 /= d1.sum()
        loss_ne = math.log(d1.size(0)) +(d1 * (d1 + 0.01).log()).sum()  # use entropy as regularization
        loss = loss_ce + self.lamda * loss_ne
        return loss