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
def sigmoid(x):
    return 1/(1+np.exp(-x))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_masked_data(data, prop=0.25):
    row,col = data.shape[0], data.shape[1]
    
    rowind = np.random.choice(row, int(prop*row*col))
    colind = np.random.choice(col, int(prop*row*col))
    
    data[rowind,colind] = 0
    
    return data

import pickle

from torch_geometric.nn import TransformerConv
from torch_geometric.nn import GATConv
from pytorch_metric_learning.losses import SelfSupervisedLoss
import os

import argparse
from pytorch_metric_learning import losses

import torch
from torch.nn import Module, Linear
from GPSConv import GPSConv
from torch_geometric.data import Data

from adabelief_pytorch import AdaBelief

class GPSTF(nn.Module):
    def __init__(self, 
                 input_dim, output_dim, 
                 heads=1) -> None:
        super(GPSTF, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.emb = nn.Linear(self.input_dim, self.output_dim)
        self.conv = GPSConv(
            channels = self.output_dim,
            conv = None,
            heads = self.heads
        )

    def forward(self, x, edge_index) -> torch.Tensor:
        x = self.emb(x)
        x = self.conv(x, edge_index)
        
        return x

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
    
    
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

args = parse_args()

for item in tissue_list_all.keys():
    tissue_list = {}
    tissue_list[item] = tissue_list_all[item]

    # construct graph batch
    # based on simulation results
    graph_list = []
    cor_list = []
    graph_networkx_list = []
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
            graph = data.Data(x=torch.FloatTensor(adata_new.X.copy()), edge_index=torch.FloatTensor(edges_new).long())

            vis = to_networkx(graph)
            graph.gene_list = pd_adata_new.index
            graph.show_index = tissue +"__" + str(i)

            graph_list.append(graph)
            label_list.append(tissue)
            
            count +=1


    class MLPEncoder_Multiinput(torch.nn.Module):
        def __init__(self, out_channels, graph_list, label_list):
            super(MLPEncoder_Multiinput, self).__init__()
            self.activ = nn.Mish()
            
            conv_dict = {}
            for i in graph_list:
                conv_dict[i.show_index] = nn.Linear(i.x.shape[1], out_channels*4)
            self.convl1 = nn.ModuleDict(conv_dict)
            
        def forward(self, x, edge_index, show_index):
            x = self.convl1[show_index](x)
            x = self.activ(x)
            return x

    class MLPEncoder_Commoninput(torch.nn.Module):
        def __init__(self, out_channels, graph_list, label_list):
            super(MLPEncoder_Commoninput, self).__init__()
            self.activ = nn.Mish()
            
            conv_dict_l2 = {}
            conv_dict_l3 = {}
            tissue_specific_list = list(set(label_list))
            
            for i in tissue_specific_list:
                conv_dict_l2[i] = nn.Linear(out_channels*4, out_channels*2)
                conv_dict_l3[i] = nn.Linear(out_channels*2, out_channels)
            self.convl2 = nn.ModuleDict(conv_dict_l2)
            self.convl3 = nn.ModuleDict(conv_dict_l3)
            
        
        def get_weight(self, show_index):
            return self.convl2[show_index.split('__')[0]].state_dict(), self.convl3[show_index.split('__')[0]].state_dict()
                
            
        def forward(self, x, edge_index, show_index):
            x = self.convl2[show_index.split('__')[0]](x)
            x = self.activ(x)
            return self.convl3[show_index.split('__')[0]](x)
        

    class MLP_edge_Decoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels, graph_list):
            super(MLP_edge_Decoder, self).__init__()
            
            dec_dict = {}
            for i in graph_list:
                dec_dict[i.show_index] = torch.nn.Sequential(
                                                nn.Linear(in_channels,  out_channels)
                                                , nn.Mish(),
                                                nn.Linear(out_channels,  out_channels) 
                                                ,nn.Mish(),
                                                nn.Linear(out_channels,  out_channels)
                                                )
            self.MLP = nn.ModuleDict(dec_dict)
            
        def forward(self, x, show_index):
            x = self.MLP[show_index](x)
            return torch.sigmoid(x)



    for seed in range(0, 10):
        set_seed(seed)
        gene_encoder_is = MLPEncoder_Multiinput(32, graph_list, label_list).to(device)
        gene_encoder_com = MLPEncoder_Commoninput(32, graph_list, label_list).to(device)

        gene_decoder = MLP_edge_Decoder(1000,1000,graph_list).to(device)

        optimizer_enc_is = torch.optim.Adam(gene_encoder_is.parameters(), lr=1e-4)
        optimizer_enc_com = torch.optim.Adam(gene_encoder_com.parameters(), lr=1e-4)

        optimizer_dec2 = torch.optim.Adam(gene_decoder.parameters(), lr=5e-4)

        loss_f = nn.BCELoss()

        
        # Contrastive learning
        gene_encoder_is.train()
        gene_encoder_com.train()
        graph_index_list = [item for item in range(0, len(graph_list))]
        edge_adj_list = [torch.FloatTensor(cor_list[i].values).to(device) for i in graph_index_list]

        for epoch in range(2000):
            loss = 0.
            for i in range(0,len(graph_index_list)):
                
                optimizer_enc_is.zero_grad(set_to_none=True)
                optimizer_enc_com.zero_grad(set_to_none=True)
                optimizer_dec2.zero_grad(set_to_none=True)

                graph = graph_list[i]

                x = graph.maskedata.to(device)
                
                train_pos_edge_index = graph.edge_index.to(device)
                x = gene_encoder_is(x, train_pos_edge_index, graph.show_index)
                z = gene_encoder_com(x, train_pos_edge_index, graph.show_index)

                edge_adj = torch.FloatTensor(cor_list[i].values).to(device)

                adj = torch.matmul(z, z.t())
                edge_reconstruct = gene_decoder(adj, graph.show_index)

                loss = loss_f(edge_reconstruct.flatten(), edge_adj.flatten())

                if epoch % 200 ==0:
                    print(loss.item())
                    
                loss.backward()
                    
                optimizer_enc_is.step()
                optimizer_enc_com.step()
                optimizer_dec2.step()

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
                x = graph.maskedata.to(device)
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

        finalname = str(args.lr1) + str(args.lr2) + str(args.epoches)
        adata.write_h5ad(f"mae_benchmark/{tissue.split('_')[1]}_umi_wsmaenew_{seed}.h5ad")