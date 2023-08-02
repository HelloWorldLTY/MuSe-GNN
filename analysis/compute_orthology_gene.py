# codes for finding the orthology genes proportion of gene embeddings
import scanpy as sc 
import pandas as pd 
import numpy as np

adata = sc.read_h5ad("multi_folder/all_umi_wslgnn_residual_3000_multispecies.h5ad")

specie = []
for item in adata.obs['tissue']:
    if 'mouse' in item:
        specie.append('mouse')
    elif 'lemur' in item:
        specie.append('lemur')
    else:
        specie.append('human')

adata.obs['species'] = specie

gene_list = pd.read_csv("orthology_gene_id/human_mouse_orthology.csv", index_col=0)

all_prop = []
for item in sorted(list(set(adata.obs.leiden))):
    
    adata_it = adata[adata.obs.leiden == item]
    
    adata_it_human = adata_it[adata_it.obs.species == 'human']
    adata_it_mouse = adata_it[adata_it.obs.species == 'mouse']
    
    if (len(adata_it_human) == 0) or (len(adata_it_mouse) == 0):
        continue
    
    count = 0
    for item1 in adata_it_human:
        for item2 in adata_it_mouse:
            if [item1.obs['gene'][0],item2.obs['gene'][0]] in gene_list.values.tolist():
                count += 1
                
    all_prop.append(count/(len(adata_it_human) * len(adata_it_mouse)))

print("human,mouse")
print(np.mean(all_prop))

gene_list = pd.read_csv("orthology_gene_id/human_lemur_orthology.csv", index_col=0)

all_prop = []
for item in sorted(list(set(adata.obs.leiden))):
    
    adata_it = adata[adata.obs.leiden == item]
    
    adata_it_human = adata_it[adata_it.obs.species == 'human']
    adata_it_mouse = adata_it[adata_it.obs.species == 'mouse']
    
    if (len(adata_it_human) == 0) or (len(adata_it_mouse) == 0):
        continue
    
    count = 0
    for item1 in adata_it_human:
        for item2 in adata_it_mouse:
            if [item1.obs['gene'][0],item2.obs['gene'][0]] in gene_list.values.tolist():
                count += 1
                
    all_prop.append(count/(len(adata_it_human) * len(adata_it_mouse)))

print("human,lemur")
print(np.mean(all_prop))

gene_list = pd.read_csv("orthology_gene_id/mouse_lemur_orthology.csv", index_col=0)

all_prop = []
for item in sorted(list(set(adata.obs.leiden))):
    
    adata_it = adata[adata.obs.leiden == item]
    
    adata_it_human = adata_it[adata_it.obs.species == 'mouse']
    adata_it_mouse = adata_it[adata_it.obs.species == 'lemur']
    
    if (len(adata_it_human) == 0) or (len(adata_it_mouse) == 0):
        continue
    
    count = 0
    for item1 in adata_it_human:
        for item2 in adata_it_mouse:
            if [item1.obs['gene'][0],item2.obs['gene'][0]] in gene_list.values.tolist():
                count += 1
                
    all_prop.append(count/(len(adata_it_human) * len(adata_it_mouse)))

print("mouse,lemur")
print(np.mean(all_prop))

