# Codes for paper: MuSe-GNN: Learning Unified Gene Representation From Multimodal Biological Graph Data


In this repository, we have four folders with different functions. The sequence of code running for our model should follow this order: 

Graph construction -> model ->benchmark

## Datasets

This folder contains the download links of our used datasets. Before running our codes, please download the datasets based on these links.

## Graph construction

This folder contrains the codes we used to generate graph structure data, which contain gene expression profiles and co-expression network. For scRNA-seq data, please directly run scRNA_preprocessing.ipynb. For scATAC-seq data, please run scATAC_preprocessing.ipynb based on the gene activity matrix of scATAC-seq data. For spatial transcriptomics data, please run spatial_sparkx.ipynb and spatial_preprocessing.ipynb to generate spatial co-expression network and augumented data for the training process.


## Model

This folder contains three .py files. MuSeGNN_heart.py represents the gene embeddings generation process for human heart scRNA-seq data. MuSeGNN_covid.py represents the gene embeddings generation process for human pancreas dataset with/without COVID. MuSeGNN_all_tissues.py represents the gene embeddings generation process for multimodal biological data.

To run the example codes, after finishing the preprocessing and preparation steps, please use:

```
python MuSeGNN_heart.py
```

## Benchmark

This folder includes the benchmark metrics and baseline models we used for our this paper. Please check benchmark metrics.py as a simple tutorial.

## Analysis

This folder includes codes for performing biological analysis, including enrichment pathway discovery, ploting tissue specific enrichment bubble plots and ploting the number of shared TFs in each combination.