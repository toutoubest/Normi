 # -*- coding:utf-8 -*-
import pandas as pd
from Evaluate import *
from pqdm.processes import pqdm


def MRMR(data_v, data):
    '''
    Revise the mutual information of a specific taget gene and its TFs

    Input: 
    data_v: dataFrame of a specific target
    data: dataFrame of the inferred network

    Output: dataFrame containing the revised mutual information
    '''
    data_v = data_v.sort_values(by='score', ascending=False)
    if len(data_v) == 1:
        return data_v

    length = len(data_v)-1
    df_res = pd.DataFrame(columns=data.columns)
    df_res = pd.concat([df_res, data_v.iloc[[0], :]])

    df = data_v.copy().iloc[1:, :]
    df.columns = ['Gene1', 'Gene2', 'mi']
    df['red'] = 0

    TF_selected = list(df_res.Gene1)
    TF_candidate = list(df.Gene1)

    for t in range(length):
        if len(TF_candidate)==0:
            break
        temp = data.loc[data.Gene2==TF_selected[-1]]
        temp = temp[['Gene1', 'score']]
        df = pd.merge(df, temp, on='Gene1', how='left').fillna(0)
        df['red'] += df['score']
        df['temp'] = df['mi'] - df['red']/len(TF_selected)
        df = df.sort_values(by='temp', ascending=False)

        add_edge = df.iloc[0][['Gene1', 'Gene2']]
        add_edge['score'] = df.iloc[0,:]['temp']
        df_res = pd.concat([df_res, add_edge.to_frame().T])

        df = df.iloc[1:, :][['Gene1', 'Gene2', 'mi', 'red']]
        TF_selected.append(str(add_edge.Gene1))
        TF_candidate = list(df.Gene1)
        
    return df_res


def MRMR2(data, n_jobs=1):
    print('---------- data.shape=', data.shape)

    target = list(data.Gene2.drop_duplicates())
    params = [[data.loc[data.Gene2==v], data] for v in target]
    result = pqdm(params, MRMR, n_jobs=n_jobs, argument_type='args', desc='Computations of MRMR')
    df_res = pd.DataFrame(columns=data.columns)
    for i in range(len(result)):
        if isinstance(result[i], pd.DataFrame):
            df_res = pd.concat([df_res, result[i]])
        else:
            print(type(result[i]))
    return df_res


def MRMR2_divergence(df_mi, n_jobs=1):
    """
    Modified mRMR using divergence scores instead of mutual information.
    This implements:  D*(X;Z) = D(X,Z) - (1/|S|) * sum_{Y in S} D(X,Y)
    """
    df_res = pd.DataFrame(columns=df_mi.columns)
    target = list(df_mi.Gene2.drop_duplicates())

    for gene_z in target:
        df_gene = df_mi[df_mi.Gene2 == gene_z]
        df_gene = df_gene[df_gene.Gene1 != gene_z]
        if df_gene.empty:
            continue

        S = []  # selected TFs
        U = df_gene.sort_values(by='score', ascending=False).Gene1.tolist()
        scores = df_gene.set_index('Gene1')['score'].to_dict()

        while len(U) > 0:
            best_score = -float('inf')
            best_gene = None
            for gene_x in U:
                div_xz = scores.get(gene_x, 0)
                if len(S) == 0:
                    score = div_xz
                else:
                    redun = [df_mi[(df_mi.Gene1 == gene_x) & (df_mi.Gene2 == gene_y)]['score'].values[0]
                             for gene_y in S if not df_mi[(df_mi.Gene1 == gene_x) & (df_mi.Gene2 == gene_y)].empty]
                    score = div_xz - sum(redun) / len(S) if redun else div_xz
                if score > best_score:
                    best_score = score
                    best_gene = gene_x
            if best_gene is None:
                break
            S.append(best_gene)
            df_res = pd.concat([df_res, pd.DataFrame([[best_gene, gene_z, scores[best_gene]]],
                                                     columns=['Gene1', 'Gene2', 'score'])])
            U.remove(best_gene)

    return df_res
