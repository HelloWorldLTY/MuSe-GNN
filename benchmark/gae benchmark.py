import scanpy as sc
import torch_geometric.nn
import torch_geometric.data as data
from torch_geometric.utils.convert import to_networkx
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from torch_geometric.nn import GAE
from torch_geometric.nn import GCNConv

import torch.distributed as dist
import random

import os


from torch import Tensor

def sigmoid(x):
    return 1/(1+np.exp(-x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


    # for specific encoder/decoder
    # tissue_list = { 
    #                "heart":[233, 676, 783, 947,266, 223, 233, 978, 928, 852, 839, 733]}
tissue_list_all = { 
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
    }

for item in tissue_list_all.keys():
    tissue_list = {}
    tissue_list[item] = tissue_list_all[item]

    # construct graph batch
    # based on simulation results
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
            
            cor_dict[graph.show_index] = correlation 

            graph_list.append(graph)
            label_list.append(tissue)
            
            count +=1
        

    class GCNEncoder_Multiinput(torch.nn.Module):
        def __init__(self, out_channels, graph_list, label_list):
            super(GCNEncoder_Multiinput, self).__init__()
            self.activ = nn.Mish()
            
            conv_dict = {}
            for i in graph_list:
                conv_dict[i.show_index] = GCNConv(i.x.shape[1], out_channels*4)
            self.convl1 = nn.ModuleDict(conv_dict)
            
        
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
    #         conv_dict_res = {}
            tissue_specific_list = list(set(label_list))
            
            for i in tissue_specific_list:
                conv_dict_l2[i] = GCNConv(out_channels*4, out_channels*2)
                conv_dict_l3[i] = GCNConv(out_channels*2, out_channels) #mu
                
            self.convl2 = nn.ModuleDict(conv_dict_l2)
            self.convl3 = nn.ModuleDict(conv_dict_l3)
    #         self.convl_res = nn.ModuleDict(conv_dict_res)
        
        def forward(self, x, edge_index, show_index):
            x = self.convl2[show_index.split('__')[0]](x, edge_index)
            x = self.activ(x)
            mu = self.convl3[show_index.split('__')[0]](x, edge_index)
            return mu


    
    loss_f = nn.BCEWithLogitsLoss()
    lambda_infonce = 0.01


    for seed in range(0,10):
        set_seed(seed)
        gene_encoder_is = GCNEncoder_Multiinput(32, graph_list, label_list).to(device)
        gene_encoder_com = GCNEncoder_Commoninput(32, graph_list, label_list).to(device)

        print(f"Let's use {torch.cuda.device_count()} GPUs!")

        optimizer_enc_is = torch.optim.Adam(gene_encoder_is.parameters(), lr=1e-4)
        optimizer_enc_com = torch.optim.Adam(gene_encoder_com.parameters(), lr=1e-4)


        # Contrastive learning
        graph_index_list = [item for item in range(0 , len(graph_list))]
        gene_encoder_is.train()
        gene_encoder_com.train()

        for epoch in range(2000):
            loss = 0.
            for i in range(0,len(graph_index_list)):
                
                optimizer_enc_is.zero_grad(set_to_none=True)
                optimizer_enc_com.zero_grad(set_to_none=True)

                graph = graph_list[i]

                x = graph.x.to(device)
                x_in = x

                train_pos_edge_index = graph.edge_index.long().to(device)
                edge_adj = torch.FloatTensor(cor_list[i].values).to(device)

                x = gene_encoder_is(x, train_pos_edge_index,  graph.show_index)
                z= gene_encoder_com(x, train_pos_edge_index, graph.show_index)
                adj = torch.sigmoid(torch.matmul(z, z.t()))
                loss = loss_f(adj.flatten(), edge_adj.flatten())

                if epoch % 200 ==0:
                    print(loss.item())


                del graph
                loss.backward()
                del loss

                optimizer_enc_is.step()
                optimizer_enc_com.step()


            print("epoch finish")

        torch.cuda.empty_cache()

        emb_list = []
        gene_list = []
        tissue_list = []
        
        gene_encoder_is.eval()
        gene_encoder_com.eval()

        with torch.no_grad():
            for i in range(0,len(graph_list)):
                graph = graph_list[i]
                x = graph.x.to(device)
                train_pos_edge_index = graph.edge_index.long().to(device)

                x = gene_encoder_is(x, train_pos_edge_index, graph.show_index)
                z= gene_encoder_com(x, train_pos_edge_index, graph.show_index)

                emb_list.append(z.cpu().numpy())

                gene_list.append(graph.gene_list)
                tissue_list.append([graph.show_index for j in range(len(x))])


        adata = sc.AnnData(np.concatenate(emb_list))
        adata.obs['gene'] = np.concatenate(gene_list)
        adata.obs['tissue'] = np.concatenate(tissue_list)


        sc.pp.neighbors(adata, use_rep='X')
        sc.tl.umap(adata)
        sc.tl.leiden(adata)

        adata.obs['tissue_new'] = [i.split("__")[0] for i in adata.obs['tissue']]

        adata.write_h5ad(f"gae_benchmark/{tissue.split('_')[1]}_umi_gae_{seed}.h5ad")