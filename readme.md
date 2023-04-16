# Codes for paper: Learning Unified Gene Representation From Multimodal Biological Graph Data


In this repository, we have four folders with different functions. The sequence of code running for our model should follow this order: 

Graph construction -> model ->benchmark

## Datasets

This folder contains the download links of our used datasets. Before running our codes, please download the datasets based on these links.

## Graph construction

This folder contrains the codes we used to generate graph structure data, which contain gene expression profiles and co-expression network. For scRNA-seq data, please directly run Global_process.py. For scATAC-seq data, please run Global_process.py based on the gene activity matrix of scATAC-seq data. For spatial transcriptomics data, please run  Global_process.py and spatial_neighbor_construction.py to generate spatial co-expression network and augumented data for the training process.


## Model

This folder contains three .py files. WSLGNN_heart.py represents the gene embeddings generation process for human heart scRNA-seq data. WSLGNN_covid.py represents the gene embeddings generation process for human pancreas dataset with/without COVID. WSLGNN_all_tissue.py represents the gene embeddings generation process for multimodal biological data.

To run the example codes, after finishing the preprocessing and preparation steps, please use:

```
python TRIANGLE_heart.py
```

## Benchmark

This folder includes the benchmark functions we used for our model. Please check benchmark heart datasets.ipynb as a simple tutorial.

## Contact

If you have any questions, please post them in the issues of this repo or email: tianyu.liu@yale.edu
