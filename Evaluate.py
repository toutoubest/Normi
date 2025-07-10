# -*- coding:utf-8 -*-
import pandas as pd
from sklearn import metrics
from sklearn.covariance import GraphicalLassoCV
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os


def concat_ref(df, df_r):
    '''
    Concat outputFile and referenceNetwork to get the ground truth of the edges

    Input: 
    df: dataFrame of the outputFile
    df_r: dataFrame of the reference network

    Output: dataFrame of the outputFile with ground truth
    '''
    df.columns = ['Gene1', 'Gene2', 'score']
    #df_r.columns = ['Gene1', 'Gene2']
    df_r['real_score'] = 1
    df = pd.merge(df, df_r, how='left')
    df.loc[df.real_score.isna(), 'real_score'] = 0
    df = df.loc[df.Gene1!=df.Gene2].drop_duplicates(keep='first')
    return df


def cal_auc_aupr(df):
    '''
    Evaluate auroc and auprc of the algorithm on a specific dataset

    Input: 
    df: dataFrame of the outputFile with ground truth

    Output: a dict with keys ['auroc', 'auprc']
    '''
    scores = df.score.values
    true = df.real_score.astype(int).values
    fpr, tpr, thresholds = metrics.roc_curve(true, scores, pos_label=1)
    prec, recall, thresholds = metrics.precision_recall_curve(true, scores, pos_label=1)
    auroc = metrics.auc(fpr, tpr)
    auprc = metrics.auc(recall, prec)
    return {'AUROC':auroc, 'AUPRC':auprc}


def cal_EPR(df, k, sparsity):
    '''
    Evaluate EP, EPR of the algorithm on a specific dataset

    Input: 
    df: dataFrame of the inferred network with ground truth
    k: the top k edges are taken for evaluation 
    sparsity: the sparsity of the real network

    Output: a dict with keys of EP, EPR on the top k edges
    '''
    k = min(k, len(df))
    df = df.sort_values(by='score', ascending=False)
    TP = df.iloc[:k, :].real_score.sum().astype(int)
    EP = TP/k
    EPR = EP/sparsity
    return TP, EP, EPR



def add_sign_and_plot(df_edge, expression_file, top_k=None, plot=True, output_pdf=None):
    '''
    Add activation/inhibition labels to edges using partial correlation,
    based only on genes involved in MI-ranked edges.

    Parameters:
    -----------
    df_edge : pd.DataFrame
        DataFrame with ['Gene1', 'Gene2', 'score'].
    expression_file : str
        Path to expression CSV file (cells × genes).
    top_k : int or None
        Top number of edges to keep.
    plot : bool
        Whether to plot network.
    output_pdf : str or None
        Path to save PDF.

    Returns:
    --------
    df_edge : pd.DataFrame with 'sign' column.
    '''

    # Load expression data
    df_exp_full = pd.read_csv(expression_file, index_col=0)

    # Keep top_k edges
    df_edge = df_edge.sort_values(by='score', ascending=False)
    if top_k is not None:
        df_edge = df_edge.iloc[:top_k]

    # Get unique genes in selected edges
    selected_genes = sorted(set(df_edge['Gene1']).union(df_edge['Gene2']))

    # Subset expression data
    df_exp = df_exp_full[selected_genes]
    X = df_exp.values
    gene_to_idx = {g: i for i, g in enumerate(df_exp.columns)}

    # Compute partial correlation using only selected genes
    model = GraphicalLassoCV().fit(X)
    precision = model.precision_
    d = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1.0)

    # Add activation/inhibition sign
    signs = []
    for _, row in df_edge.iterrows():
        i, j = gene_to_idx[row['Gene1']], gene_to_idx[row['Gene2']]
        corr = partial_corr[i, j]
        signs.append('activation' if corr > 0 else 'inhibition')
    df_edge['sign'] = signs

    # Plot
    if plot or output_pdf:
        G = nx.DiGraph()
        for _, row in df_edge.iterrows():
            style = 'solid' if row['sign'] == 'activation' else 'dashed'
            color = 'black' if row['sign'] == 'activation' else 'red'
            G.add_edge(row['Gene1'], row['Gene2'], style=style, color=color)

        pos = nx.circular_layout(G)
        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(G, pos, node_size=600)
        nx.draw_networkx_labels(G, pos, font_size=9)

        solid_edges = [(u, v) for u, v in G.edges if G[u][v]['style'] == 'solid']
        dashed_edges = [(u, v) for u, v in G.edges if G[u][v]['style'] == 'dashed']
        nx.draw_networkx_edges(G, pos, edgelist=solid_edges, edge_color='black', arrows=True, arrowsize=15)
        nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, edge_color='red', style='dashed', arrows=True, arrowsize=15)

        plt.title("Gene Regulatory Network with Activation/Inhibition")
        if output_pdf:
            plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
            print(f"Saved network plot to {output_pdf}")
        if plot:
            plt.show()

    return df_edge
