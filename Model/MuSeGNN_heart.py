'''
For the details of different functions, please check the TRIANGLE_all_tissues.py
'''
import numpy as np
import torch_geometric.nn
import torch_geometric.data as data
import networkx as nx
from torch_geometric.utils.convert import to_networkx

import torch
import torch.nn as nn
import scanpy as sc
import pandas as pd
import random
import argparse


from torch_geometric.nn import TransformerConv
import os

def sigmoid(x):
    return 1/(1+np.exp(-x))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description='Run encoder')

    parser.add_argument('--epoches', type=int, default=2000,
                        help='number of epoches')
    parser.add_argument('--lambdac', type=float, default=0.01,
                        help='weight for the contrastive learning') 
    parser.add_argument('--lr1', type=float, default=1e-4,
                        help='lr for encoder')     
    parser.add_argument('--lr2', type=float, default=1e-3,
                        help='lr for decoder')
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for the contrastive learning') 
    parser.add_argument('--samplesize', type=int, default=100,
                        help='sample size for contrastive learning')
    parser.add_argument('--dim', type=float, default=32,
                        help='latent dimensions') 
    parser.add_argument('--savepath', type=str, default="heart_global/heart_umi_musegnn.h5ad",
                        help='save path') 
    
    
    return parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#set_seed(0)

# # for specific encoder/decoder
# # tissue_list = { 
# #                "heart":[233, 676, 783, 947,266, 223, 233, 978, 928, 852, 839, 733]}

args = parse_args()

tissue_list = { 
               "scrna_heart":['D4',
 'H2',
 'H3',
 'D6',
 'D2',
 'H7',
 'D11',
 'D3',
 'D1',
 'D5',
 'H4',
 'D7',
 'H6',
 'H5',
 'G19'], 
#                    "scrna_pancreas":['2017', 'bTop3'], 
#          "scrna_kidney":["b1", "b2"],
               }

# construct graph batch
# based on simulation results
graph_list = []
cor_list = []
label_list = []
graph_networkx_list = []
count = 0

for tissue in tissue_list.keys():
    for i in tissue_list[tissue]:
        print(i)
        pathway_count = f"./{tissue.split('_')[1]}_atlas/{tissue}_" + i + "_rna_expression" + ".csv"
        pathway_matrix = f"./{tissue.split('_')[1]}_atlas/{tissue}_" + i + "_pvalue" + ".csv"

        pd_adata_new =  pd.read_csv(pathway_count, index_col=0)
        correlation = pd.read_csv(pathway_matrix, index_col=0)
        cor_list.append(correlation)

        print(correlation.shape)
        print(pd_adata_new.shape)
        adata = sc.AnnData(pd_adata_new)

        adata_new = adata.copy()
        edges_new = np.array([np.nonzero(correlation.values)[0],np.nonzero(correlation.values)[1]])
        graph = data.Data(x=torch.FloatTensor(adata_new.X.copy()), edge_index=torch.FloatTensor(edges_new).long())
        
        vis = nx.from_pandas_adjacency(correlation)
        graph_networkx_list.append(vis)
        
        graph.gene_list = pd_adata_new.index
        graph.show_index = tissue +"__" + str(i)
        
        graph_list.append(graph)
        label_list.append(tissue)
        
        count +=1


class GCNEncoder_Multiinput(torch.nn.Module):
    def __init__(self, out_channels, graph_list, label_list):
        super(GCNEncoder_Multiinput, self).__init__()
        self.activ = nn.Mish(inplace=True)
        
        conv_dict = {}
        conv_dict1 = {}
        for i in graph_list:
            conv_dict[i.show_index] = torch_geometric.nn.Sequential('x, edge_index', [(TransformerConv(i.x.shape[1], out_channels, heads=4),'x, edge_index-> x'),
                                                     (torch_geometric.nn.GraphNorm(out_channels*4), 'x -> x')])
        self.convl1 = nn.ModuleDict(conv_dict)
        self.convl2 = nn.ModuleDict(conv_dict1)
              
    def forward(self, x, edge_index, show_index):
        x = self.convl1[show_index](x, edge_index)
        x = self.activ(x)
        return x
    
class GCNEncoder_Commoninput(torch.nn.Module):
    def __init__(self, out_channels, graph_list, label_list):
        super(GCNEncoder_Commoninput, self).__init__()
        self.activ = nn.Mish(inplace=True)
        
        conv_dict_l2 = {}
        conv_dict_l3 = {}
        conv_dict_l4 = {}
        tissue_specific_list = list(set(label_list))
        
        for i in tissue_specific_list:
            conv_dict_l2[i] = torch_geometric.nn.Sequential('x, edge_index', [(TransformerConv(out_channels*4, out_channels, heads=2),'x, edge_index -> x'),
                                                     (torch_geometric.nn.GraphNorm(out_channels*2), 'x -> x')])
            conv_dict_l3[i] = TransformerConv(out_channels*2, out_channels)
            conv_dict_l4[i] = TransformerConv(out_channels*4, out_channels)
              
        self.convl2 = nn.ModuleDict(conv_dict_l2)
        self.convl3 = nn.ModuleDict(conv_dict_l3)
        self.convl4 = nn.ModuleDict(conv_dict_l4)
        
    def forward(self, x, edge_index, show_index):
        x_inp = x
        x = self.convl2[show_index.split('__')[0]](x, edge_index)
        x = self.activ(x)
        x = self.convl3[show_index.split('__')[0]](x, edge_index)
        return x + self.convl4[show_index.split('__')[0]](x_inp, edge_index)

class MLP_edge_Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, graph_list):
        super(MLP_edge_Decoder, self).__init__()
        
        dec_dict = {}
        
        self.activ = nn.Mish(inplace=True)
        for i in graph_list:
            dec_dict[i.show_index] = torch.nn.Sequential(
                                              nn.Linear(in_channels,  out_channels)
                                             , self.activ,
                                              nn.Linear(in_channels,  out_channels)
                                             , self.activ,
                                              nn.Linear(in_channels,  out_channels)
                                             )
        self.MLP = nn.ModuleDict(dec_dict)
        
    def forward(self, x, show_index):
        x = self.MLP[show_index](x)
        return x


gene_encoder_is = GCNEncoder_Multiinput(args.dim, graph_list, label_list).to(device)
gene_encoder_com =  GCNEncoder_Commoninput(args.dim, graph_list, label_list).to(device)

gene_decoder = MLP_edge_Decoder(1000,1000,graph_list).to(device)

optimizer_enc_is = torch.optim.Adam(gene_encoder_is.parameters(), lr=args.lr1)
optimizer_enc_com = torch.optim.Adam(gene_encoder_com.parameters(), lr=args.lr1)

optimizer_dec2 = torch.optim.Adam(gene_decoder.parameters(), lr=args.lr2)

import numpy as np
import networkx as nx
from concurrent.futures import ProcessPoolExecutor,as_completed, ThreadPoolExecutor
from multiprocessing import cpu_count

import process_pair

def compute_gene_sets(graph_list, graph_networkx_list, num_threads=cpu_count()):
    common_gene_set = {}
    common_gene_overlap = {}
    diff_gene_set = {}
    diff_gene_neighbor_set = {}

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(process_pair.process_pair, i, j, graph_list, graph_networkx_list)
            for i in range(len(graph_list))
            for j in range(len(graph_list))
            if i != j
        ]

        for future in as_completed(futures):
            i, j, common_set, common_overlap, diff_set, diff_neighbor_set = future.result()
            key = graph_list[i].show_index + graph_list[j].show_index
            common_gene_set[key] = common_set
            common_gene_overlap[key] = common_overlap
            diff_gene_set[key] = diff_set
            diff_gene_neighbor_set[key] = diff_neighbor_set

    return common_gene_set, common_gene_overlap, diff_gene_set, diff_gene_neighbor_set
common_gene_set, common_gene_overlap, diff_gene_set, diff_gene_neighbor_set = compute_gene_sets(graph_list, graph_networkx_list)

print("finish preprocessing")
print("start training")

loss_f = nn.BCEWithLogitsLoss()
loss_m = nn.CrossEntropyLoss()
from pytorch_metric_learning import losses
loss_func = losses.SelfSupervisedLoss(losses.NTXentLoss(temperature = args.temp))

lambda_infonce = args.lambdac
def penalize_data(z, graph_list,j):
    loss = torch.tensor(0.).to(device)
    graph_new = graph_list[j]

    x = graph_new.x.to(device)
    train_pos_edge_index = graph_new.edge_index.to(device)
    
    x = gene_encoder_is(x, train_pos_edge_index, graph_new.show_index)
    z_new = gene_encoder_com(x, train_pos_edge_index, graph_new.show_index)

    [index_i, index_j] = common_gene_set[graph.show_index + graph_new.show_index]
    if (len(index_i) ==0) or (len(index_j) == 0):
        return loss
    
    z_cor = z[index_i]
    z_new_cor = z_new[index_j]
    
    weight = torch.FloatTensor(common_gene_overlap[graph.show_index + graph_new.show_index]).to(device)
    cos_sim = torch.cosine_similarity(z_cor, z_new_cor, axis = 1)*weight

    loss += -cos_sim.mean()
    
    [index_i, index_j] = diff_gene_set[graph.show_index + graph_new.show_index]

    if (len(index_i) ==0) or (len(index_j) == 0):
        return loss
    
    opt_index = np.random.choice([i for i in range(len(index_i))], min(100, len(index_i)))
    
    z_diff = z[index_i[opt_index]]
    z_new_diff = z_new[index_j[opt_index]]
    
    [index_i, index_j] = diff_gene_neighbor_set[graph.show_index + graph_new.show_index]
    
    z_diff_true =  z[index_i[opt_index]]
    z_new_diff_true = z_new[index_j[opt_index]]
    
    cos_sim = torch.cosine_similarity(z_cor, z_new_cor, axis = 1)*weight

    loss += lambda_infonce * loss_func(torch.cat((z_diff,z_new_diff)), torch.cat((z_diff_true,z_new_diff_true)))
    return loss


# Contrastive learning
gene_encoder_is.train()
gene_encoder_com.train()
graph_index_list = [item for item in range(0, len(graph_list))]
edge_adj_list = [torch.FloatTensor(cor_list[i].values).to(device) for i in graph_index_list]

for epoch in range(args.epoches):
    loss = 0.
    for i in range(0,len(graph_index_list)):
        
        optimizer_enc_is.zero_grad(set_to_none=True)
        optimizer_enc_com.zero_grad(set_to_none=True)
        optimizer_dec2.zero_grad(set_to_none=True)

        graph = graph_list[i]

        x = graph.x.to(device)
        train_pos_edge_index = graph.edge_index.to(device)
        edge_adj = edge_adj_list[i]

        x = gene_encoder_is(x, train_pos_edge_index,  graph.show_index)
        z = gene_encoder_com(x, train_pos_edge_index, graph.show_index)
        
        adj = torch.matmul(z, z.t())
        edge_reconstruct = gene_decoder(adj, graph.show_index)
        
        loss = loss_f(edge_reconstruct.flatten(), edge_adj.flatten()) 

        if epoch % 200 ==0:
            print(loss.item())

        graph_index_list_copy = graph_index_list.copy()
        graph_index_list_copy.remove(i)

        j = random.sample(graph_index_list_copy, 1)
        loss += penalize_data(z, graph_list,j[0]) 

        del graph
        loss.backward()
        del loss
        
        optimizer_enc_is.step()
        optimizer_enc_com.step()
        optimizer_dec2.step()
        
    print("epoch finish")

torch.cuda.empty_cache()

emb_list = []
gene_list = []
tissue_list = []

# inference step
gene_encoder_is.eval()
gene_encoder_com.eval()
with torch.no_grad():
    for i in range(0,len(graph_list)):
        graph = graph_list[i]
        x = graph.x.to(device)
        train_pos_edge_index = graph.edge_index.long().to(device)
        
        x = gene_encoder_is(x, train_pos_edge_index,  graph.show_index)
        z = gene_encoder_com(x, train_pos_edge_index, graph.show_index)
        
        emb_list.append(z.cpu().numpy())
        gene_list.append(graph.gene_list)
        tissue_list.append([graph.show_index for j in range(len(x))])

adata = sc.AnnData(np.concatenate(emb_list))

adata.obs['gene'] = np.concatenate(gene_list)
adata.obs['tissue'] = np.concatenate(tissue_list)

sc.pp.neighbors(adata, use_rep='X')
sc.tl.umap(adata)

tissue_list1 = []
for i in list(set(adata.obs['tissue'])):
    tissue_list1.append(list(adata[adata.obs['tissue'] == i].obs['gene']))

common_gene_list = set(tissue_list1[0]).intersection(*tissue_list1[1:])

adata.obs['displaygene']= [True if i in common_gene_list else False for i in adata.obs['gene']]
adata.obs['displaygene']  = adata.obs['displaygene'].astype('category')

sc.tl.leiden(adata)

adata.obs['tissue_new'] = [i.split("__")[0] for i in adata.obs['tissue']]

adata.write_h5ad(args.savepath)