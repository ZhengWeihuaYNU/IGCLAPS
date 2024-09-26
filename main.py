# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from tqdm import tqdm
import time
import torch
import h5py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import get_laplacian
import warnings

from data import preprocess
from layers import GraphTransformerNet,  DataAug
from loss import AGCLoss, contrastive_loss
from utils import knngraph, sim, cluster_acc

def evaluate():
    #if gpu is availbale then use gpu
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    data_h5 = h5py.File('pbmc4k.h5') #data file, containing X as the cell*gene count matrix and Y as the real label
    expr = np.array(data_h5.get('X'))
    real_label = np.array(data_h5.get('Y')).reshape(-1)
    adata = preprocess(expr) #data preprocessing includes quality control, normalization, log-transformation and HVGs selection
    expr = torch.tensor(adata.X[:, adata.var['highly_variable']].astype(np.float32))
    adata.obs['Ground Truth'] = real_label
    pca_embed = torch.tensor(adata.obsm['X_pca'].copy())
    real_label = torch.tensor(LabelEncoder().fit_transform(adata.obs['Ground Truth']))
    num_neighbors= 400
    eig_num= 20
    gradient_clipping = 10
    num_epochs = 500
    batch_size= 512
    tau = 0.5
    lam = 0.5
    cls_thres = 0.5
    agc = AGCLoss(device= device)
    net_params = {'num_layers': 1, 'in_dim': expr.shape[1], 'hidden_dim': 32, 'out_dim': 32, 'final_embed': 16,
                  'num_heads': 4, 'dropout': 0.5, 'attn_drop': 0.5, 'add_drop': 0.0, 'lap_pos_enc': True, 
                  'pos_enc_dim':eig_num, 'cluster': len(np.unique(real_label.cpu()))}    
    adj, g_edges= knngraph(pca_embed, num_neighbors)
    lap_sym = get_laplacian(g_edges, normalization='sym')
    aaa= torch.zeros((adj.shape))
    aaa[lap_sym[0][0,:],lap_sym[0][1,:]]= lap_sym[1]
    val, vec= torch.linalg.eig(aaa)
    lp_matrix= (vec[:, torch.argsort(val.real)[:eig_num]].real)
    adj_eye= (adj+ torch.eye(adj.shape[0]))
    g_data= Data(x= expr, edge_index= g_edges, laplacian= lp_matrix)
    expr= expr.to(device)
    lp_matrix= lp_matrix.to(device)
    adj_eye= adj_eye.to(device)
    g_data= g_data.to(device)
    g_edges= g_edges.to(device)
    warnings.filterwarnings("ignore")
    min_train_loss = 100
    early_stop_counter = 50
    seed= 42
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed) # CPU
    loader = NeighborLoader(g_data, num_neighbors= [5]*net_params['num_layers'], batch_size=batch_size, shuffle=True)
    model= GraphTransformerNet(net_params).to(device)
    aug_model= DataAug(net_params['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay= 1e-4)
    best_t = -1
    counter = 0
    start_time= time.time()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            batch = batch.to(device)
            input1, input2 = aug_model(batch.x), aug_model(batch.x)
            output1, output2, contrast1, contrast2, cluster1, cluster2 = model(input1, input2, batch.edge_index, batch.laplacian)
            # 注意模型输出的行数不是batch_size，而是为了计算batch_size个节点的表示所需要的节点个数，这些节点的id是batch.n_id，其中前batch_size个
            # id就是最终得到的batch_size个节点的id，即batch.input_id
            final_cluster = ((cluster1 + cluster2) / 2)[:len(batch.input_id),
                            :].detach()  # 形状为(batch_size,net_params['clusters'])
            sub_adj_eye = adj_eye[batch.input_id, :][:, batch.input_id]
            sim_cls = sim(final_cluster, final_cluster) - torch.eye(sub_adj_eye.shape[0]).to(device)
            loss_instance = contrastive_loss(contrast1[:len(batch.input_id), :], contrast2[:len(batch.input_id), :],
                                             tau,
                                             sub_adj_eye, sim_cls, cls_thres)
            loss_cluster = agc(cluster1[:len(batch.input_id), :], cluster2[:len(batch.input_id), :], sub_adj_eye)
            loss = lam * loss_instance + (1 - lam) * loss_cluster
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            output1, output2, contrast1, contrast2, cluster1, cluster2 = model(expr, expr, g_edges, lp_matrix)
            final_cluster = ((cluster1 + cluster2) / 2)
            sim_cls = sim(final_cluster, final_cluster) - (torch.eye(adj_eye.shape[0])).to(device)
            loss_instance_train = contrastive_loss(contrast1, contrast2, tau,
                                                   adj_eye, sim_cls, cls_thres)
            loss_cluster_train = agc(cluster1, cluster2, adj_eye)
            loss_train = lam * loss_instance_train + (1 - lam) * loss_cluster_train
            ari = adjusted_rand_score(real_label.cpu(), torch.argmax(final_cluster, 1).cpu())
        if loss_train < min_train_loss:
            counter = 0
            min_train_loss = loss_train
            best_t = epoch
            torch.save(model.state_dict(), 'best_model.pkl')
        else:
            counter += 1
        if counter >= early_stop_counter:
            print('early stop')
            break    
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss_train:.4f}')
            print(f'Epoch {epoch + 1}/{num_epochs}, Instance Loss: {loss_instance_train:.4f}')
            print(f'Epoch {epoch + 1}/{num_epochs}, Cluster Loss: {loss_cluster_train:.4f}')
            print(f'Ari: {ari}')
            print(f'current best epoch: {best_t + 1}')
    print('Loading {}th epoch'.format(best_t))
    model.eval()
    model.load_state_dict(torch.load('best_model.pkl'))
    output1, output2, contrast1, contrast2, cluster1, cluster2 = model(expr, expr, g_edges, lp_matrix)
    final_output = (output1 + output2) / 2
    final_output = final_output.cpu().detach().numpy()
    final_cluster = ((cluster1 + cluster2) / 2).argmax(1).cpu()
    ari = adjusted_rand_score(real_label, final_cluster)
    nmi = normalized_mutual_info_score(real_label, final_cluster)
    ca = cluster_acc(real_label, final_cluster)
    end_time= time.time()
    print(f'time cost: {end_time- start_time}')
    print(f'evaluated ari: {ari}')
    print(f'evaluated nmi: {nmi}')
    print(f'evaluated ca: {ca}')
if __name__ == '__main__':
    evaluate()

