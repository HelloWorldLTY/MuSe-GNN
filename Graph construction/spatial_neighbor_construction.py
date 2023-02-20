import numpy as np

import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from scipy.spatial import Delaunay

import matplotlib.pyplot as plt


import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import scanpy as sc
import numpy as np
import pandas as pd

import scanpy as sc
import anndata2ri
import logging
from scipy.sparse import issparse
from CSCORE.CSCORE_IRLS import CSCORE_IRLS

import rpy2.rinterface_lib.callbacks as rcb
import rpy2.robjects as ro
import glob

import seaborn as sns

rcb.logger.setLevel(logging.ERROR)
ro.pandas2ri.activate()
anndata2ri.activate()
%load_ext rpy2.ipython




adata = sc.read_h5ad("heart_atlas/spatial_heart_visium.h5ad")

adata

adata.var_names_make_unique()

adata.var_names

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

adata = adata[adata.obs.n_genes_by_counts < 3000, :]
adata = adata[adata.obs.pct_counts_mt < 60, :]

adata = adata[:,adata.var['mt']==False]


adata.obs['num_index'] = [i for i in range(len(adata))]

tri = Delaunay(adata.obsm['spatial'])

def find_neighbors(pindex, triang):

    return triang.vertex_neighbor_vertices[1][triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex+1]]

adata_new_x = []
for index, item in enumerate(adata.obs['num_index']):
    neig_list = find_neighbors(item, tri)
    adata_neg = adata[neig_list]
    neg_value = np.mean(adata_neg.X, axis=0)

    adata_new_x.append(np.array(neg_value)[0])



adata_new = sc.AnnData(np.array(adata_new_x), obs=adata.obs, var=adata.var)

adata_new.obs_names = adata.obs_names 
adata_new.var_names = adata.var_names

adata_new

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

adata_filter = adata_new.copy()


ro.r('''
library(Seurat)
library(sctransform)
library(Hmisc)
''')


adata = adata_new.copy()

if issparse(adata.X):
    if not adata.X.has_sorted_indices:
        adata.X.sort_indices()
ro.globalenv["adata_new"] = adata_new

adata.obs['n_counts'] = np.array(np.sum(adata.X, axis = 1))


ro.r('''
seurat_obj = as.Seurat(adata_new, counts="X", data = NULL)
seurat_obj = RenameAssays(seurat_obj, originalexp = "RNA")
res = SCTransform(object=seurat_obj, vst.flavor = "v2", variable.features.n = 1000 , method = "glmGamPoi", verbose = FALSE)
''')

gene_list = list(ro.r("rownames(res@assays$SCT@scale.data)"))
norm_x = ro.r("res@assays$SCT@scale.data")
exp_matrix = pd.DataFrame(norm_x, index=gene_list)

adata_new = adata[:,gene_list]

adata_new.obs['n_counts'] = np.array(np.sum(adata.X, axis = 1))

counts = adata_new.X.todense()
seq_depth = adata_new.obs['n_counts'].values

B_cell_result = CSCORE_IRLS(np.array(counts), seq_depth)


p_value = B_cell_result[1]
cor_matrix = (p_value<0.005)*1
print(cor_matrix)

exp_matrix.to_csv(f"./heart_atlas/spatial_heart_visiumneighbor"+"_rna_expression.csv")
cor_matrix = pd.DataFrame(cor_matrix, index = gene_list, columns = gene_list)
cor_matrix.to_csv(f"./heart_atlas/spatial_heart_visiumneighbor"+"_pvalue.csv")