#!/usr/bin/env python
# coding: utf-8
# The codes here need rpy2.

# In[1]:


import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rpy2

import scanpy as sc
import numpy as np
import pandas as pd

import scanpy as sc
import anndata2ri
import logging
from scipy.sparse import issparse

import rpy2.rinterface_lib.callbacks as rcb
import rpy2.robjects as ro


# In[62]:


rcb.logger.setLevel(logging.ERROR)
ro.pandas2ri.activate()
anndata2ri.activate()
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[63]:


get_ipython().run_cell_magic('R', '', 'library(Seurat)\nlibrary(sctransform)\nlibrary(Hmisc)\n')


# In[4]:

# load one dataset in python
adata = sc.read_h5ad(f"heart_atlas/scrna_heart_D5.h5ad")


# In[5]:

# reorder the dataset and import the dataset into R environment.
if issparse(adata.X):
    if not adata.X.has_sorted_indices:
        adata.X.sort_indices()
ro.globalenv["adata"] = adata


# In[7]:

# Run sctransform and select highly variable genes.
get_ipython().run_cell_magic('R', '', 'seurat_obj = as.Seurat(adata, counts="X", assay = "RNA", data = NULL)\nseurat_obj = RenameAssays(seurat_obj, originalexp = "RNA")\nres = SCTransform(object=seurat_obj, vst.flavor = "v2", variable.features.n = 1000 , method = "glmGamPoi", verbose = FALSE)\n')


# In[15]:


get_ipython().run_cell_magic('R', '', '')


# In[10]:


adata.var_names


# In[28]:

# store the Pearson residuals
gene_list = list(ro.r("rownames(res@assays$SCT@scale.data)"))
norm_x = ro.r("res@assays$SCT@scale.data")
exp_matrix = pd.DataFrame(norm_x, index=gene_list)


# In[29]:


exp_matrix
adata = sc.read_h5ad("realdata_sctransform/scrna_heart_global.h5ad")


# In[7]:


sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)


# In[19]:


adata


# In[20]:


adata.write("realdata_sctransform/scrna_heart_processed.h5ad")


# In[23]:


adata.obs['n_counts']


# In[25]:


for i in list(set(adata.obs['donor'])):
    adata_new = adata[adata.obs['donor'] == i]
    adata_new.write_h5ad(f"heart_atlas/scrna_heart_{i}.h5ad")


# In[ ]:





# In[26]:


adata = sc.read_h5ad(f"heart_atlas/scrna_heart_{i}.h5ad")


# In[29]:


adata


# In[33]:


get_ipython().run_cell_magic('time', '', 'adata_sct = SCTransform(adata,\n                        n_genes=1000,\n                        n_cells=None, #use all cells\n                        bin_size=500,\n                        bw_adjust=3,\n                        inplace=False)\n')


# In[45]:





# In[50]:


counts = adata.X[:,0:1000].todense()
seq_depth = adata.obs['n_counts'].values


# In[65]:


counts.shape


# In[66]:


seq_depth.shape


# In[70]:


np.array(counts)


# In[52]:


from CSCORE.CSCORE_IRLS import CSCORE_IRLS


# In[71]:


B_cell_result = CSCORE_IRLS(np.array(counts), seq_depth)


# In[78]:


B_cell_result[1]


# In[1]:


get_ipython().run_cell_magic('R', '', '')


# In[ ]:





# # True pipeline

# In[2]:


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


# In[3]:


import seaborn as sns


# In[4]:


rcb.logger.setLevel(logging.ERROR)
ro.pandas2ri.activate()
anndata2ri.activate()
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[5]:


get_ipython().run_cell_magic('R', '', 'library(Seurat)\nlibrary(sctransform)\nlibrary(Hmisc)\n')


# In[56]:


tissue_id = "spleen"
identify = "637C"


# In[57]:


for i in glob.glob("./spleen/scrna_*.h5ad"):
    print(i)


# In[ ]:





# In[58]:


read_path = f"/ysm-gpfs/pi/zhao/tl688/GIANT/GIANT/src/analysis/{tissue_id}/scrna_{tissue_id}_{identify}.h5ad"


# In[59]:


identify


# In[60]:


adata = sc.read_h5ad(read_path)


# In[61]:


adata


# In[62]:


adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)


# In[63]:


sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')


# In[64]:


adata = adata[adata.obs.n_genes_by_counts < 4000, :]
adata = adata[adata.obs.pct_counts_mt < 20, :]


# In[65]:


adata = adata[:,adata.var['mt']==False]


# In[66]:


adata


# In[67]:


if issparse(adata.X):
    if not adata.X.has_sorted_indices:
        adata.X.sort_indices()
ro.globalenv["adata"] = adata


# In[68]:


adata.obs['n_counts'] = np.array(np.sum(adata.X, axis= 1))


# In[69]:


get_ipython().run_cell_magic('R', '', 'seurat_obj = as.Seurat(adata, counts="X", data = NULL)\nseurat_obj = RenameAssays(seurat_obj, originalexp = "RNA")\nres = SCTransform(object=seurat_obj, vst.flavor = "v2", variable.features.n = 1000 , method = "glmGamPoi", verbose = FALSE)\n')


# In[70]:


adata.var_names


# In[71]:


gene_list = list(ro.r("rownames(res@assays$SCT@scale.data)"))
norm_x = ro.r("res@assays$SCT@scale.data")
exp_matrix = pd.DataFrame(norm_x, index=gene_list)


# In[72]:


exp_matrix


# In[73]:


adata_new = adata[:,gene_list]


# In[74]:


adata_new


# In[75]:


counts = adata_new.X
seq_depth = adata_new.obs['n_counts'].values


# In[76]:


counts


# In[77]:


B_cell_result = CSCORE_IRLS(np.array(counts), seq_depth)


# In[78]:


p_value = B_cell_result[1]
cor_matrix = (p_value<0.005)*1
print(cor_matrix)


# In[79]:


exp_matrix.to_csv(f"./{tissue_id}_atlas/scrna_{tissue_id}_{identify}"+"_rna_expression.csv")
cor_matrix = pd.DataFrame(cor_matrix, index = gene_list, columns = gene_list)
cor_matrix.to_csv(f"./{tissue_id}_atlas/scrna_{tissue_id}_{identify}"+"_pvalue.csv")


# In[80]:


a = 1


# In[ ]:





# # Pipeline

# In[1]:


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


# In[2]:


import seaborn as sns


# In[3]:


rcb.logger.setLevel(logging.ERROR)
ro.pandas2ri.activate()
anndata2ri.activate()
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[19]:


get_ipython().run_cell_magic('R', '', 'library(Seurat)\nlibrary(sctransform)\nlibrary(Hmisc)\n')


# In[19]:


for i in glob.glob("./heart_atlas/*_pvalue.csv"):
    print(i)


# In[5]:


for i in glob.glob("./heart_atlas/*"):
    print(i)


# In[44]:


read_path = "./heart_atlas/scrna_heart_H3.h5ad"
identify = read_path.split('_')[3].split('.')[0]


# In[45]:


adata = sc.read_h5ad(read_path)


# In[46]:


adata


# In[47]:


if issparse(adata.X):
    if not adata.X.has_sorted_indices:
        adata.X.sort_indices()
ro.globalenv["adata"] = adata


# In[48]:


adata


# In[49]:


get_ipython().run_cell_magic('R', '', 'seurat_obj = as.Seurat(adata, counts="X", data = NULL)\nseurat_obj = RenameAssays(seurat_obj, originalexp = "RNA")\nres = SCTransform(object=seurat_obj, vst.flavor = "v2", variable.features.n = 1000 , method = "glmGamPoi", verbose = FALSE)\n')


# In[ ]:





# In[50]:


adata.var_names


# In[51]:


gene_list = list(ro.r("rownames(res@assays$SCT@scale.data)"))
norm_x = ro.r("res@assays$SCT@scale.data")
exp_matrix = pd.DataFrame(norm_x, index=gene_list)


# In[52]:


exp_matrix


# In[53]:


# adata_new = adata[:,gene_list]


# In[54]:


# counts = adata_new.X.todense()
# seq_depth = adata_new.obs['n_counts'].values


# In[55]:


# B_cell_result = CSCORE_IRLS(np.array(counts), seq_depth)


# In[56]:


# %%R
# correct_matrix = res@assays$SCT@scale.data

# correct_matrix = t(correct_matrix)

# correlaton_result = rcorr(correct_matrix, type="pearson")

# p_mat = correlaton_result$P

# p_mat[is.na(p_mat)] <- 0


# In[57]:


# cor_matrix = ro.r("p_mat")
# cor_matrix = (cor_matrix<0.005)*1


# In[58]:


# p_value = B_cell_result[1]
# cor_matrix = (p_value<0.005)*1
# print(cor_matrix)


# In[ ]:





# In[59]:


exp_matrix.to_csv(f"./heart_atlas_sctrans/scrna_{identify}"+"_gene_exp_new.csv")
cor_matrix = pd.DataFrame(cor_matrix, index = gene_list, columns = gene_list)
cor_matrix.to_csv(f"./heart_atlas_sctrans/scrna_{identify}"+"_cox_exp_new.csv")


# In[ ]:





# In[27]:


np.sum(cor_matrix)/(1000*1000)


# In[50]:


for i in cor_matrix:
    print(i)


# In[60]:


sns.heatmap(cor_matrix)


# In[ ]:





# In[ ]:





# In[1]:


import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# In[2]:


import seaborn as sns


# In[3]:


rcb.logger.setLevel(logging.ERROR)
ro.pandas2ri.activate()
anndata2ri.activate()
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[4]:


get_ipython().run_cell_magic('R', '', 'library(Seurat)\nlibrary(sctransform)\n')


# In[5]:


read_path = f"heart_atlas/scrna_heart_H2.h5ad"


# In[6]:


adata = sc.read_h5ad(read_path)
identify = read_path.split('_')[3].split('.')[0]


# In[7]:


identify


# In[8]:


adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)


# In[9]:


sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')


# In[10]:


adata = adata[adata.obs.n_genes_by_counts < 3000, :]
adata = adata[adata.obs.pct_counts_mt < 2.5, :]


# In[11]:


adata = adata[:,adata.var['mt']==False]


# In[12]:


adata


# In[13]:


if issparse(adata.X):
    if not adata.X.has_sorted_indices:
        adata.X.sort_indices()
ro.globalenv["adata"] = adata


# In[14]:


adata.obs['n_counts'] = np.array(np.sum(adata.X, axis= 1))


# In[15]:


adata


# In[16]:


get_ipython().run_cell_magic('R', '', 'seurat_obj = as.Seurat(adata, counts="X",  data = NULL)\nseurat_obj = RenameAssays(seurat_obj, originalexp = "RNA")\nres = SCTransform(object=seurat_obj, vst.flavor = "v2", variable.features.n = 1000 , method = "glmGamPoi", verbose = FALSE)\n')


# In[17]:


adata.var_names


# In[ ]:





# In[18]:


gene_list = list(ro.r("rownames(res@assays$SCT@scale.data)"))
norm_x = ro.r("res@assays$SCT@scale.data")
exp_matrix = pd.DataFrame(norm_x, index=gene_list)


# In[19]:


adata_new = adata[:,gene_list]


# In[20]:


adata_new


# In[21]:


counts = adata_new.X.todense()
seq_depth = adata_new.obs['n_counts'].values


# In[22]:


B_cell_result = CSCORE_IRLS(np.array(counts), seq_depth)


# In[23]:


# %%R
# correct_matrix = res@assays$SCT@scale.data

# correct_matrix = t(correct_matrix)

# correlaton_result = rcorr(correct_matrix, type="pearson")

# p_mat = correlaton_result$P

# p_mat[is.na(p_mat)] <- 0


# In[24]:


# cor_matrix = ro.r("p_mat")
# cor_matrix = (cor_matrix<0.005)*1


# In[25]:


p_value = B_cell_result[1]
cor_matrix = (p_value<0.005)*1
print(cor_matrix)


# In[26]:


exp_matrix.to_csv(f"./heart_atlas/scrna_heart_{identify}"+"_rna_expression.csv")
cor_matrix = pd.DataFrame(cor_matrix, index = gene_list, columns = gene_list)
cor_matrix.to_csv(f"./heart_atlas/scrna_heart_{identify}"+"_pvalue.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Lung

# In[1]:


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


# In[2]:


import seaborn as sns


# In[3]:


rcb.logger.setLevel(logging.ERROR)
ro.pandas2ri.activate()
anndata2ri.activate()
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[4]:


get_ipython().run_cell_magic('R', '', 'library(Seurat)\nlibrary(sctransform)\nlibrary(Hmisc)\n')


# In[5]:


for i in glob.glob("/ysm-gpfs/pi/zhao/tl688/GIANT/GIANT/src/analysis/lung/*.h5ad"):
    print(i)


# In[6]:


read_path = "/ysm-gpfs/pi/zhao/tl688/GIANT/GIANT/src/analysis/lung/scrna_lung_BAL034.h5ad"
identify = read_path.split('_')[2].split('.')[0]


# In[7]:


identify


# In[8]:


adata = sc.read_h5ad(read_path)


# In[9]:


adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)


# In[10]:


sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')


# In[10]:


adata = adata[adata.obs.n_genes_by_counts < 3000, :]
adata = adata[adata.obs.pct_counts_mt < 2.5, :]


# In[11]:


adata = adata[:,adata.var['mt']==False]


# In[12]:


adata


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[109]:


# del adata.obs


# In[110]:


if issparse(adata.X):
    if not adata.X.has_sorted_indices:
        adata.X.sort_indices()
ro.globalenv["adata"] = adata


# In[111]:


adata.obs['n_counts'] = np.array(np.sum(adata.X, axis= 1))


# In[112]:


get_ipython().run_cell_magic('R', '', 'seurat_obj = as.Seurat(adata, counts="X", data = NULL)\nseurat_obj = RenameAssays(seurat_obj, originalexp = "RNA")\nres = SCTransform(object=seurat_obj, vst.flavor = "v2", variable.features.n = 1000 , method = "glmGamPoi", verbose = FALSE)\n')


# In[113]:


adata.var_names


# In[114]:


gene_list = list(ro.r("rownames(res@assays$SCT@scale.data)"))
norm_x = ro.r("res@assays$SCT@scale.data")
exp_matrix = pd.DataFrame(norm_x, index=gene_list)


# In[115]:


exp_matrix


# In[116]:


adata_new = adata[:,gene_list]


# In[117]:


adata_new


# In[118]:


counts = adata_new.X.todense()
seq_depth = adata_new.obs['n_counts'].values


# In[119]:


B_cell_result = CSCORE_IRLS(np.array(counts), seq_depth)


# In[120]:


# %%R
# correct_matrix = res@assays$SCT@scale.data

# correct_matrix = t(correct_matrix)

# correlaton_result = rcorr(correct_matrix, type="pearson")

# p_mat = correlaton_result$P

# p_mat[is.na(p_mat)] <- 0


# In[121]:


# cor_matrix = ro.r("p_mat")
# cor_matrix = (cor_matrix<0.005)*1


# In[122]:


p_value = B_cell_result[1]
cor_matrix = (p_value<0.005)*1
print(cor_matrix)


# In[123]:


exp_matrix.to_csv(f"./lung_global/scrna_lung_{identify}"+"_rna_expression.csv")
cor_matrix = pd.DataFrame(cor_matrix, index = gene_list, columns = gene_list)
cor_matrix.to_csv(f"./lung_global/scrna_lung_{identify}"+"_pvalue.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Kidney

# In[1]:


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


# In[2]:


import seaborn as sns


# In[3]:


rcb.logger.setLevel(logging.ERROR)
ro.pandas2ri.activate()
anndata2ri.activate()
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[4]:


get_ipython().run_cell_magic('R', '', 'library(Seurat)\nlibrary(sctransform)\nlibrary(Hmisc)\n')


# In[5]:


for i in glob.glob("/ysm-gpfs/pi/zhao/tl688/GIANT/GIANT/src/analysis/lung/*.h5ad"):
    print(i)


# In[28]:


read_path = "/ysm-gpfs/pi/zhao/tl688/GIANT/GIANT/src/analysis/kidney/scrna_kidney_b2.h5ad"
identify = read_path.split('_')[2].split('.')[0]


# In[29]:


identify


# In[30]:


adata = sc.read_h5ad(read_path)


# In[31]:


adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)


# In[32]:


sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')


# In[33]:


adata = adata[adata.obs.n_genes_by_counts < 3000, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]


# In[34]:


adata = adata[:,adata.var['mt']==False]


# In[35]:


adata


# In[ ]:





# In[36]:


# del adata.obs


# In[37]:


if issparse(adata.X):
    if not adata.X.has_sorted_indices:
        adata.X.sort_indices()
ro.globalenv["adata"] = adata


# In[39]:


adata.obs['n_counts'] = np.array(np.sum(adata.X, axis= 1))


# In[40]:


get_ipython().run_cell_magic('R', '', 'seurat_obj = as.Seurat(adata, counts="X", data = NULL)\nseurat_obj = RenameAssays(seurat_obj, originalexp = "RNA")\nres = SCTransform(object=seurat_obj, vst.flavor = "v2", variable.features.n = 1000 , method = "glmGamPoi", verbose = FALSE)\n')


# In[41]:


adata.var_names


# In[42]:


gene_list = list(ro.r("rownames(res@assays$SCT@scale.data)"))
norm_x = ro.r("res@assays$SCT@scale.data")
exp_matrix = pd.DataFrame(norm_x, index=gene_list)


# In[43]:


exp_matrix


# In[44]:


adata_new = adata[:,gene_list]


# In[45]:


adata_new


# In[46]:


counts = adata_new.X.todense()
seq_depth = adata_new.obs['n_counts'].values


# In[47]:


B_cell_result = CSCORE_IRLS(np.array(counts), seq_depth)


# In[48]:


p_value = B_cell_result[1]
cor_matrix = (p_value<0.005)*1
print(cor_matrix)


# In[49]:


exp_matrix.to_csv(f"./kidney_atlas/scrna_kidney_{identify}"+"_rna_expression.csv")
cor_matrix = pd.DataFrame(cor_matrix, index = gene_list, columns = gene_list)
cor_matrix.to_csv(f"./kidney_atlas/scrna_kidney_{identify}"+"_pvalue.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Thymus

# In[1]:


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


# In[2]:


import seaborn as sns


# In[3]:


rcb.logger.setLevel(logging.ERROR)
ro.pandas2ri.activate()
anndata2ri.activate()
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[4]:


get_ipython().run_cell_magic('R', '', 'library(Seurat)\nlibrary(sctransform)\nlibrary(Hmisc)\n')


# In[5]:


for i in glob.glob("/ysm-gpfs/pi/zhao/tl688/GIANT/GIANT/src/analysis/thymus/*.h5ad"):
    print(i)


# In[ ]:





# In[6]:


read_path = "/ysm-gpfs/pi/zhao/tl688/GIANT/GIANT/src/analysis/thymus/scrna_thymus_A31.h5ad"
identify = read_path.split('_')[2].split('.')[0]


# In[ ]:





# In[7]:


identify


# In[8]:


adata = sc.read_h5ad(read_path)


# In[9]:


adata


# In[10]:


# sc.pp.filter_cells(adata, min_genes=200)
# sc.pp.filter_genes(adata, min_cells=3)


# In[11]:


adata


# In[12]:


adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)


# In[13]:


sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')


# In[14]:


adata = adata[adata.obs.n_genes_by_counts < 3500, :]
adata = adata[adata.obs.pct_counts_mt < 50, :]


# In[15]:


adata = adata[:,adata.var['mt']==False]


# In[16]:


adata


# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


gene_name_filter = pd.read_csv("/ysm-gpfs/pi/zhao/tl688/GIANT/GIANT/src/analysis/ensemble_to_gene_all.txt", sep="\t")


# In[11]:


gene_name_filter


# In[12]:


adata.var_names


# In[13]:


trans_list = []
gene_list = []
for i in adata.var_names:
    if i in gene_name_filter['Gene stable ID version'].values:
        gene_index = gene_name_filter[gene_name_filter['Gene stable ID version'] == i]['Gene name']
        if gene_index.isna().values[0] == False:
            trans_list.append(i)
            gene_list.append(gene_index.values[0])


# In[14]:


gene_list


# In[15]:


trans_list


# In[16]:


gene_list


# In[17]:


adata = adata[:,trans_list]
adata.var_names = np.array(gene_list)


# In[35]:


adata


# In[36]:


adata.var_names_make_unique()


# In[19]:


# del adata.obs


# In[17]:


if issparse(adata.X):
    if not adata.X.has_sorted_indices:
        adata.X.sort_indices()
ro.globalenv["adata"] = adata


# In[18]:


adata.obs['n_counts'] = np.array(np.sum(adata.X, axis= 1))


# In[19]:


get_ipython().run_cell_magic('R', '', 'seurat_obj = as.Seurat(adata, counts="X", data = NULL)\nseurat_obj = RenameAssays(seurat_obj, originalexp = "RNA")\nres = SCTransform(object=seurat_obj, vst.flavor = "v2", variable.features.n = 1000 , method = "glmGamPoi", verbose = FALSE)\n')


# In[ ]:





# In[20]:


gene_list = list(ro.r("rownames(res@assays$SCT@scale.data)"))
norm_x = ro.r("res@assays$SCT@scale.data")
exp_matrix = pd.DataFrame(norm_x, index=gene_list)


# In[ ]:





# In[21]:


len(set(adata.var_names))


# In[39]:


adata.var_names


# In[22]:


adata_new = adata[:,gene_list]


# In[23]:


adata_new


# In[ ]:





# In[26]:


counts = adata_new.X
seq_depth = adata_new.obs['n_counts'].values


# In[27]:


B_cell_result = CSCORE_IRLS(np.array(counts), seq_depth)


# In[28]:


# %%R
# correct_matrix = res@assays$SCT@scale.data

# correct_matrix = t(correct_matrix)

# correlaton_result = rcorr(correct_matrix, type="pearson")

# p_mat = correlaton_result$P

# p_mat[is.na(p_mat)] <- 0


# In[29]:


# cor_matrix = ro.r("p_mat")
# cor_matrix = (cor_matrix<0.005)*1


# In[30]:


p_value = B_cell_result[1]
cor_matrix = (p_value<0.005)*1
print(cor_matrix)


# In[32]:


exp_matrix.to_csv(f"./thymus_atlas/scrna_thymus_{identify}"+"_rna_expression.csv")
cor_matrix = pd.DataFrame(cor_matrix, index = gene_list, columns = gene_list)
cor_matrix.to_csv(f"./thymus_atlas/scrna_thymus_{identify}"+"_pvalue.csv")


# In[ ]:





# In[ ]:





# # Spleen

# In[5]:


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


# In[6]:


import seaborn as sns


# In[7]:


rcb.logger.setLevel(logging.ERROR)
ro.pandas2ri.activate()
anndata2ri.activate()
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[8]:


get_ipython().run_cell_magic('R', '', 'library(Seurat)\nlibrary(sctransform)\nlibrary(Hmisc)\n')


# In[33]:


tissue_id = "spleen"


# In[34]:


for i in glob.glob("/ysm-gpfs/pi/zhao/tl688/GIANT/GIANT/src/analysis/spleen/*.h5ad"):
    print(i)


# In[ ]:





# In[62]:


read_path = f"/ysm-gpfs/pi/zhao/tl688/GIANT/GIANT/src/analysis/{tissue_id}/scrna_{tissue_id}_640C.h5ad"
identify = read_path.split('_')[2].split('.')[0]


# In[63]:


identify


# In[64]:


adata = sc.read_h5ad(read_path)


# In[65]:


adata


# In[66]:


adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)


# In[67]:


sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')


# In[68]:


adata = adata[adata.obs.n_genes_by_counts < 5000, :]
adata = adata[adata.obs.pct_counts_mt < 20, :]


# In[69]:


adata = adata[:,adata.var['mt']==False]


# In[70]:


adata


# In[ ]:





# In[ ]:





# In[ ]:





# In[71]:


# del adata.obs


# In[72]:


if issparse(adata.X):
    if not adata.X.has_sorted_indices:
        adata.X.sort_indices()
ro.globalenv["adata"] = adata


# In[73]:


adata.obs['n_counts'] = np.array(np.sum(adata.X, axis= 1))


# In[ ]:





# In[74]:


get_ipython().run_cell_magic('R', '', 'seurat_obj = as.Seurat(adata, counts="X", data = NULL)\nseurat_obj = RenameAssays(seurat_obj, originalexp = "RNA")\nres = SCTransform(object=seurat_obj, vst.flavor = "v2", variable.features.n = 1000 , method = "glmGamPoi", verbose = FALSE)\n')


# In[75]:


adata.var_names


# In[76]:


gene_list = list(ro.r("rownames(res@assays$SCT@scale.data)"))
norm_x = ro.r("res@assays$SCT@scale.data")
exp_matrix = pd.DataFrame(norm_x, index=gene_list)


# In[77]:


exp_matrix


# In[78]:


adata_new = adata[:,gene_list]


# In[79]:


adata_new


# In[80]:


counts = adata_new.X
seq_depth = adata_new.obs['n_counts'].values


# In[81]:


B_cell_result = CSCORE_IRLS(np.array(counts), seq_depth)


# In[82]:


p_value = B_cell_result[1]
cor_matrix = (p_value<0.005)*1
print(cor_matrix)


# In[83]:


exp_matrix.to_csv(f"./{tissue_id}_atlas/scrna_{tissue_id}_{identify}"+"_rna_expression.csv")
cor_matrix = pd.DataFrame(cor_matrix, index = gene_list, columns = gene_list)
cor_matrix.to_csv(f"./{tissue_id}_atlas/scrna_{tissue_id}_{identify}"+"_pvalue.csv")


# In[ ]:





# # Liver

# In[1]:


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


# In[2]:


import seaborn as sns


# In[3]:


rcb.logger.setLevel(logging.ERROR)
ro.pandas2ri.activate()
anndata2ri.activate()
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[4]:


get_ipython().run_cell_magic('R', '', 'library(Seurat)\nlibrary(sctransform)\nlibrary(Hmisc)\n')


# In[5]:


tissue_id = "liver"


# In[6]:


tissue_id


# In[7]:


for i in glob.glob(f"/ysm-gpfs/pi/zhao/tl688/GIANT/GIANT/src/analysis/{tissue_id}/*.h5ad"):
    print(i)


# In[ ]:





# In[8]:


read_path = f"/ysm-gpfs/pi/zhao/tl688/GIANT/GIANT/src/analysis/{tissue_id}/scrna_{tissue_id}_A29.h5ad"
identify = read_path.split('_')[2].split('.')[0]


# In[9]:


identify


# In[17]:


adata = sc.read_h5ad(read_path)


# In[11]:


# del adata.obs


# In[18]:


adata


# In[19]:


adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)


# In[20]:


sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')


# In[21]:


adata = adata[adata.obs.n_genes_by_counts < 3000, :]
adata = adata[adata.obs.pct_counts_mt < 30, :]


# In[22]:


adata = adata[:,adata.var['mt']==False]


# In[23]:


adata


# In[24]:


if issparse(adata.X):
    if not adata.X.has_sorted_indices:
        adata.X.sort_indices()
ro.globalenv["adata"] = adata


# In[25]:


adata.obs['n_counts'] = np.array(np.sum(adata.X, axis= 1))


# In[26]:


get_ipython().run_cell_magic('R', '', 'seurat_obj = as.Seurat(adata, counts="X", data = NULL)\nseurat_obj = RenameAssays(seurat_obj, originalexp = "RNA")\nres = SCTransform(object=seurat_obj, vst.flavor = "v2", variable.features.n = 1000 , method = "glmGamPoi", verbose = FALSE)\n')


# In[27]:


adata.var_names


# In[28]:


gene_list = list(ro.r("rownames(res@assays$SCT@scale.data)"))
norm_x = ro.r("res@assays$SCT@scale.data")
exp_matrix = pd.DataFrame(norm_x, index=gene_list)


# In[29]:


exp_matrix


# In[30]:


adata_new = adata[:,gene_list]


# In[31]:


adata_new


# In[32]:


counts = adata_new.X
seq_depth = adata_new.obs['n_counts'].values


# In[33]:


B_cell_result = CSCORE_IRLS(np.array(counts), seq_depth)


# In[34]:


p_value = B_cell_result[1]
cor_matrix = (p_value<0.005)*1
print(cor_matrix)


# In[35]:


tissue_id


# In[36]:


exp_matrix.to_csv(f"./{tissue_id}_atlas/scrna_{tissue_id}_{identify}"+"_rna_expression.csv")
cor_matrix = pd.DataFrame(cor_matrix, index = gene_list, columns = gene_list)
cor_matrix.to_csv(f"./{tissue_id}_atlas/scrna_{tissue_id}_{identify}"+"_pvalue.csv")


# In[ ]:





# # Bonemarrow

# In[163]:


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


# In[164]:


import seaborn as sns


# In[165]:


rcb.logger.setLevel(logging.ERROR)
ro.pandas2ri.activate()
anndata2ri.activate()
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[166]:


get_ipython().run_cell_magic('R', '', 'library(Seurat)\nlibrary(sctransform)\nlibrary(Hmisc)\n')


# In[207]:


tissue_id = "bonemarrow"


# In[209]:


tissue_id


# In[210]:


for i in glob.glob(f"/ysm-gpfs/pi/zhao/tl688/GIANT/GIANT/src/analysis/{tissue_id}/*.h5ad"):
    print(i)


# In[ ]:





# In[211]:


read_path = f"/ysm-gpfs/pi/zhao/tl688/GIANT/GIANT/src/analysis/{tissue_id}/scrna_{tissue_id}_A35.h5ad"
identify = read_path.split('_')[2].split('.')[0]


# In[212]:


identify


# In[213]:


adata = sc.read_h5ad(read_path)


# In[214]:


# del adata.obs


# In[215]:


if issparse(adata.X):
    if not adata.X.has_sorted_indices:
        adata.X.sort_indices()
ro.globalenv["adata"] = adata


# In[216]:


adata.obs['n_counts'] = np.array(np.sum(adata.X, axis= 1))


# In[217]:


get_ipython().run_cell_magic('R', '', 'seurat_obj = as.Seurat(adata, counts="X", data = NULL)\nseurat_obj = RenameAssays(seurat_obj, originalexp = "RNA")\nres = SCTransform(object=seurat_obj, vst.flavor = "v2", variable.features.n = 1000 , method = "glmGamPoi", verbose = FALSE)\n')


# In[218]:


adata.var_names


# In[219]:


gene_list = list(ro.r("rownames(res@assays$SCT@scale.data)"))
norm_x = ro.r("res@assays$SCT@scale.data")
exp_matrix = pd.DataFrame(norm_x, index=gene_list)


# In[220]:


exp_matrix


# In[221]:


adata_new = adata[:,gene_list]


# In[222]:


adata_new


# In[223]:


counts = adata_new.X
seq_depth = adata_new.obs['n_counts'].values


# In[224]:


B_cell_result = CSCORE_IRLS(np.array(counts), seq_depth)


# In[225]:


p_value = B_cell_result[1]
cor_matrix = (p_value<0.005)*1
print(cor_matrix)


# In[226]:


tissue_id


# In[227]:


exp_matrix.to_csv(f"./{tissue_id}_atlas/scrna_{tissue_id}_{identify}"+"_rna_expression.csv")
cor_matrix = pd.DataFrame(cor_matrix, index = gene_list, columns = gene_list)
cor_matrix.to_csv(f"./{tissue_id}_atlas/scrna_{tissue_id}_{identify}"+"_pvalue.csv")


# In[ ]:




