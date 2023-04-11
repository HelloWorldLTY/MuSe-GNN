import numpy as np
import networkx as nx
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import cpu_count



eps = 1e-8
def find_neighbor_overlap(G1,G2, gene1, gene2):
    G1_neg = set(G1.neighbors(gene1))
    G2_neg = set(G2.neighbors(gene2))
    overlap_score = len(G1_neg & G2_neg) / (len(G1_neg | G2_neg) + eps)
    return overlap_score

def process_pair(i, j, graph_list, graph_networkx_list):
    graph = graph_list[i]
    G1 = graph_networkx_list[i]
    graph_new = graph_list[j]
    G2 = graph_networkx_list[j]

    genes_common = list(set(graph.gene_list).intersection(set(graph_new.gene_list)))
    index_i = graph.gene_list.get_indexer(genes_common)
    index_j = graph_new.gene_list.get_indexer(genes_common)

    common_gene_set = [index_i, index_j]
    common_gene_overlap = [find_neighbor_overlap(G1, G2, item, item) for item in genes_common]

    s1_gene = set(graph.gene_list) - set(graph_new.gene_list)
    s2_gene = set(graph_new.gene_list) - set(graph.gene_list)

    index_i = graph.gene_list.get_indexer(s1_gene)
    index_j = graph_new.gene_list.get_indexer(s2_gene)
    diff_gene_set = [index_i, index_j]

    s1_nei_gene = np.array([np.random.choice(list(G1.neighbors(gene1))) if len(list(G1.neighbors(gene1))) !=0 else gene1 for gene1 in s1_gene])
    s2_nei_gene = np.array([np.random.choice(list(G2.neighbors(gene1))) if len(list(G2.neighbors(gene1))) !=0 else gene1 for gene1 in s2_gene])

    index_i = graph.gene_list.get_indexer(s1_nei_gene)
    index_j = graph_new.gene_list.get_indexer(s2_nei_gene)
    diff_gene_neighbor_set = [index_i, index_j]

    return i, j, common_gene_set, common_gene_overlap, diff_gene_set, diff_gene_neighbor_set