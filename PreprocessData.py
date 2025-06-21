# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np 
import pandas as pd
from scipy.stats import gaussian_kde, ks_2samp, entropy

def write_file(filePath, text):
    file = open(filePath, 'w')
    sample = ''
    for t in text:
        sample = sample + str(t) + '\n'
    file.write(sample)
    file.close()


def smooth(df_p, df_e, slipe, k=5):
    '''
    Applying sliding size-fixed windows to expression profiles to obtain smoothed data

    Input: 
    df_p: dataframe of pseudo time 
    df_e: dataframe of expression profile with columns as genes, indexs as cells
    slipe: sliding length
    k: window size

    Output: a list containing the num of each linear trajectory and a dataframe of 
            expression profile after preprocessing
    '''
    count = []    
    for i in range(len(df_p.columns)):
        df_p_c = df_p.iloc[:, i:(i + 1)].dropna()
        df_p_c.columns = ['pseudotime']
        df_p_c.sort_values(by=['pseudotime'], ascending=True, inplace=True)
        df_p_e = pd.merge(df_p_c, df_e, left_index=True, right_index=True)
        df_ee = df_p_e.iloc[:, 1:]
        
        start = 0
        end = 0
        j = 0

        df_e_new = pd.DataFrame()
        while end==0:
            j = j + 1
            if (start+k) < len(df_ee.index):
                df = df_ee.iloc[start:(start+k), :]
            else:
                df = df_ee.iloc[(len(df_ee.index)-k):len(df_ee.index), :]
                end = 1

            # take zeros into consideration
            res = df.apply(lambda x:x.value_counts().get(0,0), axis=0).astype(float)
            for item in range(len(df.columns)):
                res[item]=0.0 if res[item]>=k/2 else df.iloc[:, item].mean()
            df_mean = res.to_frame().T
            df_mean.index=[j]

            if j==0:
                df_e_new = df_mean
            else:
                df_e_new = pd.concat([df_e_new, df_mean])

            start = start + slipe
        
        if i == 0:
            df_e_all = df_e_new
        else:
            df_e_all = pd.concat([df_e_all, df_e_new])

        count.append(len(df_e_new))
    return count, df_e_all

#new code to add many divergences:

def compute_f_divergence(p1, p2, divergence_type="kullback-leibler"):
    kde_p1 = gaussian_kde(p1)
    kde_p2 = gaussian_kde(p2)
    
    grid = np.linspace(
        max(min(p1), min(p2)),
        min(max(p1), max(p2)),
        1000
    )
    
    density_p1 = kde_p1(grid)
    density_p2 = kde_p2(grid)
    
    epsilon = 1e-10
    density_p1 = (density_p1 + epsilon) / np.sum(density_p1 + epsilon)
    density_p2 = (density_p2 + epsilon) / np.sum(density_p2 + epsilon)
    
    if divergence_type == "kullback-leibler":
        return entropy(density_p1, density_p2)
    elif divergence_type == "jensen-shannon":
        m = 0.5 * (density_p1 + density_p2)
        return 0.5 * (entropy(density_p1, m) + entropy(density_p2, m))
    else:
        raise ValueError("Unknown divergence type")

def smooth_divergence(df_pse, df_exp, slide=1, k=5, distance=7):  # distance=7 example = symmetric KL
    df_exp_smooth = pd.DataFrame(columns=df_exp.columns)
    count = []
    
    pseudo_time = df_pse.values.flatten()
    order = np.argsort(pseudo_time)
    df_exp_sorted = df_exp.iloc[order, :].reset_index(drop=True)
    
    N = len(df_exp_sorted)
    num_windows = (N - k) // slide + 1
    
    for i in range(num_windows):
        start = i * slide
        end = start + k
        if end + 1 > N:
            break
        
        data_t = df_exp_sorted.iloc[start:end, :]
        data_t1 = df_exp_sorted.iloc[start+1:end+1, :]
        
        row_values = {}
        for gene in df_exp.columns:
            p1 = data_t[gene].values
            p2 = data_t1[gene].values
            
            if distance == 1:  # KS distance
                ks_stat = ks_2samp(p1, p2).statistic
                row_values[gene] = ks_stat
            elif distance == 5:  # KL(P||Q) forward KL
                kl_pq = compute_f_divergence(p1, p2, "kullback-leibler")
                row_values[gene] = kl_pq
            elif distance == 6:  # KL(Q||P) reverse KL
                kl_qp = compute_f_divergence(p2, p1, "kullback-leibler")
                row_values[gene] = kl_qp
            elif distance == 7:  # Symmetric KL
                kl_pq = compute_f_divergence(p1, p2, "kullback-leibler")
                kl_qp = compute_f_divergence(p2, p1, "kullback-leibler")
                row_values[gene] = 0.5 * (kl_pq + kl_qp)
            elif distance == 8:  # Jesen-shannon divergence(JS divergence)
                js = compute_f_divergence(p1, p2, "jensen-shannon")
                row_values[gene] = js
            # Add elif distance == 9 ... 12 here if we want implement Pearson, Neyman, etc
            else:
                raise ValueError("Unsupported distance type!")
        
        df_exp_smooth = pd.concat([df_exp_smooth, pd.DataFrame(row_values, index=[0])], ignore_index=True)
        count.append(k)
    
    return count, df_exp_smooth

