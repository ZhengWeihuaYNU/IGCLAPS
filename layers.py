import torch.nn as nn
import torch
from torch_geometric.nn import TransformerConv

class DataAug(nn.Module):
    def __init__(self, dropout=0.9):
        super(DataAug, self).__init__()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        aug_data = self.drop(x)
        return aug_data

class GraphTransformerLayer(nn.Module):
    """
        Param:
    """

    def __init__(self, in_dim, out_dim, num_heads, attn_drop=0.6, add_drop=0.2, alpha=1.0):
        super().__init__()
        self.addnorm1 = AddNorm(add_drop, in_dim)
        self.addnorm2 = AddNorm(add_drop, in_dim)
        self.ffn = FFN(in_dim, in_dim * 2, in_dim, alpha)
        self.attention = TransformerConv(in_dim, out_dim, num_heads, dropout=attn_drop)  # 输出维度为out_dim*num_heads
        self.activate = torch.nn.ELU()
        self.O = nn.Linear(out_dim * num_heads, in_dim)  # 将维度重新变为in_dim以便叠加多个层

    #        self.layer_norm= LayerNorm(in_dim)

    def forward(self, h, g):
        h_in1 = h
        a = self.attention(h, g)  # h是边索引，是2行n列(n为样本数)的张量
        #        print(a.shape)
        a_trans = self.O(self.activate(a))
        h = self.addnorm1(h_in1, a_trans)
        h_in2 = h
        h = self.addnorm2(h_in2, self.ffn(h))

        return h


class FFN(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, alpha):
        super().__init__()
        self.dense1 = nn.Linear(dim_input, dim_hidden)
        self.elu = torch.nn.ELU(alpha)
        self.dense2 = nn.Linear(dim_hidden, dim_output)

    def forward(self, X):
        return self.dense2(self.elu(self.dense1(X)))

class AddNorm(nn.Module):
    def __init__(self, dropout, layer_dim):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(layer_dim)
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class GraphTransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        num_heads = net_params['num_heads']
        dropout = net_params['dropout']
        attn_drop = net_params['attn_drop']
        num_layers = net_params['num_layers']
        add_drop = net_params['add_drop']
        final_embed = net_params['final_embed']
        cluster = net_params['cluster']
#        self.embedding_h1 = nn.Sequential(nn.Linear(in_dim, in_dim//4),nn.Linear(in_dim//4, hidden_dim))
#        self.embedding_h2 = nn.Sequential(nn.Linear(in_dim, in_dim//4),nn.Linear(in_dim//4, hidden_dim))
        self.embedding_h1 = nn.Linear(in_dim, hidden_dim)
        self.embedding_h2 = nn.Linear(in_dim, hidden_dim)

        self.encoder_q = nn.ModuleList([GraphTransformerLayer(hidden_dim, out_dim, num_heads, attn_drop, add_drop)
                                        for _ in range(num_layers)])
        self.encoder_k = nn.ModuleList([GraphTransformerLayer(hidden_dim, out_dim, num_heads, attn_drop, add_drop)
                                        for _ in range(num_layers)])
        self.cluster_projector_q = nn.Linear(hidden_dim, cluster)
        self.cluster_projector_k = nn.Linear(hidden_dim, cluster)
        self.contrast_projector_q = nn.Linear(hidden_dim, final_embed)
        self.contrast_projector_k = nn.Linear(hidden_dim, final_embed)
        self.lap_pos_enc = net_params['lap_pos_enc']
        if self.lap_pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_lap_pos_enc1 = nn.Linear(pos_enc_dim, hidden_dim) #拉普拉斯编码后的投影层
            self.embedding_lap_pos_enc2 = nn.Linear(pos_enc_dim, hidden_dim) #拉普拉斯编码后的投影层


    def forward(self, h1, h2, g, lap= None):
        h1 = self.embedding_h1(h1)
        if self.lap_pos_enc:
            h_lap_pos_enc1 = self.embedding_lap_pos_enc1(lap) 
            h1 = h1 + h_lap_pos_enc1
        for conv in self.encoder_q:
            h1 = conv(h1, g)
        h1_contrast = self.contrast_projector_q(h1)
        h1_cluster = self.cluster_projector_q(h1)
        h2 = self.embedding_h2(h2)
        if self.lap_pos_enc:
            h_lap_pos_enc2 = self.embedding_lap_pos_enc2(lap) 
            h2 = h2 + h_lap_pos_enc2

        for conv in self.encoder_k:
            h2 = conv(h2, g)
        h2_contrast = self.contrast_projector_k(h2)
        h2_cluster = self.cluster_projector_k(h2)
        return h1, h2, h1_contrast, h2_contrast, h1_cluster, h2_cluster
