import scanpy as sc
import numpy as np
import pandas as pd
import scib

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

def calculate_metric(adata, cor_list):
    asw = calculate_common_asw(adata)
    AUC = calculate_AUC(adata, cor_list)
    ilisi = calculate_iLISI(adata)
    gc = calculate_graph_connectivity(adata)
    
    percp = calculate_common_gene_propertion(adata)
    
    df = pd.DataFrame(np.array([asw,AUC,ilisi,gc,percp]))
    df.index = ['ASW', 'AUC', 'iLISI', 'GC','Common Prop']
    return df