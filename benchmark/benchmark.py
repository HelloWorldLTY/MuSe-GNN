import scanpy as sc
import numpy as np
import pandas as pd
import networkx as nx

import scib

# for specific encoder/decoder


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
}

# construct graph batch
# based on simulation results
graph_list = []
cor_list = {}
label_list = [] 
count = 0
for tissue in tissue_list.keys():
    for i in tissue_list[tissue]:
        print(i)
        pathway_count = f"./heart_atlas/{tissue}_" + i + "_rna_expression" + ".csv"
        pathway_matrix = f"./heart_atlas/{tissue}_" + i + "_pvalue" + ".csv"
        correlation = pd.read_csv(pathway_matrix, sep=",", index_col=0)
        cor_list[tissue +"__" + str(i)] = correlation

        print(correlation.shape)

        label_list.append(tissue +"__" + str(i))
        
        count +=1

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def calculate_common_asw(adata):
    tissue_list = []
    for i in list(set(adata.obs['tissue'])):
        tissue_list.append(list(adata[adata.obs['tissue'] == i].obs['gene']))
        
    common_gene_list = set(tissue_list[0]).intersection(*tissue_list[1:])
    adata_new = adata[[True if i in common_gene_list else False for i in adata.obs['gene']]]
    adata_new.obsm['X_emb'] = adata_new.X
    
    result = scib.metrics.silhouette_batch(adata_new, batch_key='tissue', group_key='leiden', embed='X_emb')
    
    return result    



from sklearn.metrics import roc_auc_score
def calculate_AUC(adata, cor_list):
    tissue_list = list(set(cor_list.keys()))
    
    result = 0
    for i in tissue_list:
        adata_new = adata[adata.obs['tissue'] == i]
        rec_matrix = sigmoid(adata_new.X@adata_new.X.T).flatten()
        cor_matrix = cor_list[i].values.flatten()
        result += roc_auc_score(cor_matrix, rec_matrix)
    
    result = result/len(tissue_list)
    return result    


def calculate_iLISI(adata):
    tissue_list = []
    for i in list(set(adata.obs['tissue'])):
        tissue_list.append(list(adata[adata.obs['tissue'] == i].obs['gene']))
        
    common_gene_list = set(tissue_list[0]).intersection(*tissue_list[1:])
    adata_new = adata[[True if i in common_gene_list else False for i in adata.obs['gene']]]
    adata_new.obsm['X_emb'] = adata_new.X
    
    result = scib.metrics.ilisi_graph(adata_new, batch_key="tissue", type_="embed")
    
    return result   



def calculate_graph_connectivity(adata):
    tissue_list = []
    for i in list(set(adata.obs['tissue'])):
        tissue_list.append(list(adata[adata.obs['tissue'] == i].obs['gene']))
        
    common_gene_list = set(tissue_list[0]).intersection(*tissue_list[1:])
    adata_new = adata[[True if i in common_gene_list else False for i in adata.obs['gene']]]

    adata_new.obsm['X_emb'] = adata_new.X
    
    result = scib.metrics.graph_connectivity(adata_new,'leiden')
    
    return result    

def calculate_common_gene_propertion(adata):
    full_score = 0
    for i in list(set(adata.obs['leiden'])):
        adata_new = adata[adata.obs['leiden'] == i]
        
        gene_list = set(adata_new.obs['gene'])
        
        prop = 1 - len(gene_list)/len(adata_new.obs['gene'])
        print(prop)
        
        full_score += len(adata_new)/len(adata) * prop
        
    return full_score

def calculate_common_gene_cluster_propertion(adata):
    tissue_list = []
    for i in list(set(adata.obs['tissue'])):
        tissue_list.append(list(adata[adata.obs['tissue'] == i].obs['gene']))
        
    common_gene_list = set(tissue_list[0]).intersection(*tissue_list[1:])
    adata_new = adata[[True if i in common_gene_list else False for i in adata.obs['gene']]]

    result = len(set(adata_new.obs['leiden']))/len(set(adata.obs['leiden']))
    return result

def calculate_metric(adata, cor_list):
    asw = calculate_common_asw(adata)
    AUC = calculate_AUC(adata, cor_list)
    ilisi = calculate_iLISI(adata)
    gc = calculate_graph_connectivity(adata)
    
    percp = calculate_common_gene_cluster_propertion(adata)
    
    ratio = calculate_common_gene_propertion(adata)
    
    df = pd.DataFrame(np.array([asw,AUC,ilisi,gc,percp,ratio]))
    df.index = ['ASW', 'AUC', 'iLISI', 'GC','Common Prop cluster','Common ratio']
    return df

def calculate_overlap(G1,G2,g1,g2):
    G1_neg = list(G1.neighbors(g1))
    G2_neg = list(G2.neighbors(g2))
    
    overlap_score = len(set(G1_neg).intersection(set(G2_neg)))/len(set(G1_neg).union(set(G2_neg)))
    
    return overlap_score

graph_list = {}
for i in cor_list.keys():
    graph_list[i] = nx.from_pandas_adjacency(cor_list[i])

def calculate_common_neighbor_ovarlap(adata, cor_list):
    output_value = 0
    for i in list(set(adata.obs['leiden'])):
        adata_new = adata[adata.obs['leiden'] == i]
        
        tissue_list = list(adata_new.obs['tissue'])
        gene_list = list(adata_new.obs['gene'])
        
        overlap_value = 0
        dim_value = 0
        for num1,item1 in enumerate(gene_list):
            for num2,item2 in enumerate(gene_list):
                t1 = tissue_list[num1]
                t2 = tissue_list[num2]
                if t1 != t2:
                    g1 = graph_list[t1]
                    g2 = graph_list[t2]
                    temp_overlap = calculate_overlap(g1,g2,item1,item2)
                    overlap_value += temp_overlap
                    dim_value += 1.0
                
        print("finish one cluster")
        
        if dim_value == 0:
            overlap_value = 0
        else:
            overlap_value = overlap_value/dim_value
        output_value += overlap_value*len(adata_new)/len(adata)
        
    return output_value
        

# new loss
adata = sc.read_h5ad("heart_global/heart_umi_SWMGNN_cossim_infoNCE.h5ad")

print(calculate_metric(adata, cor_list))

print(calculate_common_neighbor_ovarlap(adata, cor_list))

