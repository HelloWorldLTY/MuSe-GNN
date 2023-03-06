import scanpy as sc
import torch_geometric.nn
import torch_geometric.datasets as datasets
import torch_geometric.data as data
import torch_geometric.transforms as transforms
import networkx as nx
from torch_geometric.utils.convert import to_networkx

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pickle

from torch_geometric.nn import TransformerConv


import random
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.utils import negative_sampling
from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.losses import SelfSupervisedLoss
from pytorch_metric_learning import losses, reducers

# store the name of files used in this task.
tissue_list = { 
               "scrna_heart":['D4',
 'H2',
 'H3',
 'D6',
 'D2',
],
    "scrna_lung":["BAL034", "A44-LNG-2-SC-45N-1","ND17494","BAL027","BT1294"],
    "scrna_kidney":["b1", "b2"],
        "scrna_liver":["A31", "A29", "A35", "A36", "A52", "640C", "637C"],
        "scrna_thymus":["A31"],
    "scrna_pbmcHealthy":[ 'H1', 'H2', 'H3', 'H4', 'H5', 'H6'],
    "scrna_pancreas":['2017'],
    "scrna_spleen":["A52", "640C", "A36", "A31", "A29", "637C"],
    "scatac_heart":['674', '328', '864'],
  "spatial_heart":['visium', 'visiumneighbor']
               }

# Data structure used to store the information related to the graph.
graph_list = []
cor_list = []
cor_dict = {}
label_list = []
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
        graph = data.Data(x=torch.FloatTensor(adata_new.X.copy()), edge_index=torch.FloatTensor(edges_new))

        vis = to_networkx(graph)
        graph.gene_list = pd_adata_new.index
        graph.show_index = tissue +"__" + str(i)
        cor_dict[graph.show_index] = correlation.copy()


        graph_list.append(graph)
        label_list.append(tissue)
        
        count +=1

# network construction: Dataset isolated GNN
class GCNEncoder_Multiinput(torch.nn.Module):
    def __init__(self, out_channels, graph_list, label_list):
        super(GCNEncoder_Multiinput, self).__init__()
        self.activ = nn.Mish(inplace=True)
        
        conv_dict = {}
        for i in graph_list:
            conv_dict[i.show_index] = torch_geometric.nn.Sequential('x, edge_index', [(TransformerConv(i.x.shape[1], out_channels, heads = 4),'x, edge_index -> x'),
                                                     (torch_geometric.nn.GraphNorm(out_channels*4), 'x -> x')])
        self.convl1 = nn.ModuleDict(conv_dict)
    
        
    def forward(self, x, edge_index, show_index):
        x = self.convl1[show_index](x, edge_index)
        x = self.activ(x)
        return x

# network construction: Tissue common GNN
class GCNEncoder_Commoninput(torch.nn.Module):
    def __init__(self, out_channels, graph_list, label_list):
        super(GCNEncoder_Commoninput, self).__init__()
        self.activ = nn.Mish(inplace=True)
        
        conv_dict_l2 = {}
        conv_dict_l3 = {}
        tissue_specific_list = list(set(label_list))
        
        for i in tissue_specific_list:
            conv_dict_l2[i] = torch_geometric.nn.Sequential('x, edge_index', [(TransformerConv(out_channels*4, out_channels, heads = 2),'x, edge_index -> x'),
                                                     (torch_geometric.nn.GraphNorm(out_channels*2), 'x -> x')])
            conv_dict_l3[i] = TransformerConv(out_channels*2, out_channels)
        self.convl2 = nn.ModuleDict(conv_dict_l2)
        self.convl3 = nn.ModuleDict(conv_dict_l3)
        
    def forward(self, x, edge_index, show_index):
        x = self.convl2[show_index.split('__')[0]](x, edge_index)
        x = self.activ(x)
        return self.convl3[show_index.split('__')[0]](x, edge_index)

# network construction: Dataset isolated MLP
class MLP_edge_Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, graph_list):
        super(MLP_edge_Decoder, self).__init__()
        
        dec_dict = {}
        for i in graph_list:
            dec_dict[i.show_index] = torch.nn.Sequential(
                                              nn.Linear(in_channels,  out_channels)
                                             , nn.Mish(inplace=True),
                                              nn.Linear(out_channels,  out_channels) 
                                              ,nn.Mish(inplace=True),
                                              nn.Linear(out_channels,  out_channels)
                                             )
        self.MLP = nn.ModuleDict(dec_dict)
        
    def forward(self, x, show_index):
        x = self.MLP[show_index](x)
        return x

# Instantiating different neural networks
gene_encoder_is = GCNEncoder_Multiinput(32, graph_list, label_list).to(device)
gene_encoder_com = GCNEncoder_Commoninput(32, graph_list, label_list).to(device)
gene_decoder = MLP_edge_Decoder(1000,1000,graph_list).to(device)

# Create optimizer for different neural networks
optimizer_enc_is = torch.optim.Adam(gene_encoder_is.parameters(), lr=1e-4)
optimizer_enc_com = torch.optim.Adam(gene_encoder_com.parameters(), lr=1e-4)
optimizer_dec2 = torch.optim.Adam(gene_decoder.parameters(), lr=1e-3)

# Create loss functions for different neural networks.
loss_f = nn.BCEWithLogitsLoss()
loss_m = nn.CrossEntropyLoss()
loss_func = SelfSupervisedLoss(NTXentLoss())

# A helper function to calcualte the neighbor overlap.
def find_neighbor_overlap(G1,G2, gene):
    G1_neg = list(G1.neighbors(gene))
    G2_neg = list(G2.neighbors(gene))
    overlap_score = len(set(G1_neg).intersection(set(G2_neg)))/len(set(G1_neg).union(set(G2_neg)))
    return overlap_score

# Calculate neighbor overlap
common_gene_set ={}
common_gene_overlap = {}
for i in range(0,len(graph_list)):
    graph = graph_list[i]
    G1 = nx.from_pandas_adjacency(cor_dict[graph.show_index])
    for j in range(0,len(graph_list)):
        graph_new = graph_list[j]
        genes_common = list(set(graph.gene_list).intersection(set(graph_new.gene_list)))
        index_i = graph.gene_list.get_indexer(genes_common)
        index_j = graph_new.gene_list.get_indexer(genes_common)
        common_gene_set[graph.show_index + graph_new.show_index] = [index_i, index_j]
        G2 = nx.from_pandas_adjacency(cor_dict[graph_new.show_index])
        value_list = []
        if i==j:
            for item in genes_common:
                value_list.append(1.0)
        else:
            for item in genes_common:
                value_list.append(find_neighbor_overlap(G1,G2, item))
        common_gene_overlap[graph.show_index + graph_new.show_index] = value_list
            
# Calculate the set for different genes.   
diff_gene_set ={}
for i in range(0,len(graph_list)):
    graph = graph_list[i]
    for j in range(0,len(graph_list)):
        graph_new = graph_list[j]
        index_i = graph.gene_list.get_indexer(set(graph.gene_list) - set(graph_new.gene_list))
        index_j = graph_new.gene_list.get_indexer(set(graph_new.gene_list) - set(graph.gene_list))
        diff_gene_set[graph.show_index + graph_new.show_index] = [index_i, index_j]

# A function used for regularization, including similarity learning and contrastive learning
lambda_infonce = 0.01
def penalize_data(z, graph_list,Z,i,j):
    loss = 0.
    graph_new = graph_list[j]
    x = graph_new.x.to(device)
    train_pos_edge_index = graph_new.edge_index.long().to(device)
    x = gene_encoder_is(x, train_pos_edge_index, graph_new.show_index)
    z_new = gene_encoder_com(x, train_pos_edge_index, graph_new.show_index)


    [index_i, index_j] = common_gene_set[graph.show_index + graph_new.show_index]
    
    z_cor = z[index_i]
    z_new_cor = z_new[index_j]
    
    weight = torch.FloatTensor(common_gene_overlap[graph.show_index + graph_new.show_index]).to(device)

    [index_i, index_j] = diff_gene_set[graph.show_index + graph_new.show_index]

    z_diff = z[index_i]
    z_new_diff = z_new[index_j]

    cos_sim = torch.cosine_similarity(z_cor, z_new_cor, axis = 1)*weight
    
    loss += -cos_sim.mean() + lambda_infonce * loss_func(z_diff, z_new_diff)
        
    return loss

# Start the training process
graph_index_list = [item for item in range(0 , len(graph_list))]

for epoch in range(2000):
     
    for i in range(0,len(graph_index_list)):
        
        optimizer_enc_is.zero_grad(set_to_none=True)
        optimizer_enc_com.zero_grad(set_to_none=True)
        optimizer_dec2.zero_grad(set_to_none=True)
        
        graph = graph_list[i]
        
        x = graph.x.to(device)
        
        train_pos_edge_index = graph.edge_index.long().to(device)
        edge_adj = torch.FloatTensor(cor_list[i].values).to(device)
        
        
        x = gene_encoder_is(x, train_pos_edge_index, graph.show_index)
        z = gene_encoder_com(x, train_pos_edge_index, graph.show_index)
        adj = torch.matmul(z, z.t())
        edge_reconstruct = gene_decoder(adj, graph.show_index)
        loss = loss_f(edge_reconstruct.flatten(), edge_adj.flatten())
        
        if epoch % 200 ==0:
            print(loss.item())
            
        graph_index_list_copy = graph_index_list.copy()
        graph_index_list_copy.remove(i)
        
        j = random.sample(graph_index_list_copy, 1)
        
        loss += penalize_data(z, graph_list,Z,i,j[0])
    
        
        del graph
        loss.backward()
        del loss
        
        optimizer_enc_is.step()
        optimizer_enc_com.step()
        optimizer_dec2.step()

        
    print("epoch finish")

# Store the gene embeddings
emb_list = []
gene_list = []
tissue_list = []

with torch.no_grad():
    for i in range(0,len(graph_list)-1):
        graph = graph_list[i].to(device)
        x = graph.x
        train_pos_edge_index = graph.edge_index.long()
        
        x = gene_encoder_is(x, train_pos_edge_index, graph.show_index)
        z = gene_encoder_com(x, train_pos_edge_index, graph.show_index)
        
        emb_list.append(z.cpu().numpy())
        
        gene_list.append(graph.gene_list)
        tissue_list.append([graph.show_index for j in range(len(x))])

# Post-processing
adata = sc.AnnData(np.concatenate(emb_list))
adata.obs['gene'] = np.concatenate(gene_list)
adata.obs['tissue'] = np.concatenate(tissue_list)
sc.pp.neighbors(adata, use_rep='X')
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=3) # 3 means we have three types of techiniques 

adata.obs['tissue_new'] = [i.split("__")[0] for i in adata.obs['tissue']]

# Write the gene embeddings into a file with .h5ad format.
adata.write_h5ad("/ysm-gpfs/pi/zhao/tl688/GIANT/GIANT/src/analysis/multi_folder/umi_SWMGNN_alltissue_withpancreas_neighborspatial_infoNCE_sample_2000.h5ad")