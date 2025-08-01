# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import scipy.spatial as ss
from itertools import permutations, product
from pqdm.processes import pqdm
import dcor
from scipy.special import digamma
from math import log, floor
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.stats import energy_distance 


def MI_Gao(x,y,k=5):
    '''
    Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
    Using *Mixed-KSG* mutual information estimator

    Input: 
    x: 2D array of size N*d_x (or 1D list of size N if d_x = 1)
    y: 2D array of size N*d_y (or 1D list of size N if d_y = 1)
    k: k-nearest neighbor parameter

    Output: one number of I(X;Y)
    '''

    assert len(x)==len(y), "Lists should have same length"
    assert k <= len(x)-1, "Set k smaller than num. samples - 1"
    N = len(x)
    data = np.c_[x, y]

    if x.ndim == 1:
        x = x.reshape((N,1))

    if y.ndim == 1:
        y = y.reshape((N,1))

    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_dis = [tree_xy.query(point,k+1,p=float('inf'))[0][k] for point in data]
    ans = 0

    for i in range(N):
        kp, nx, ny = k, k, k
        if knn_dis[i] == 0:
            kp = len(tree_xy.query_ball_point(data[i],1e-15,p=float('inf')))
            nx = len(tree_x.query_ball_point(x[i],1e-15,p=float('inf')))
            ny = len(tree_y.query_ball_point(y[i],1e-15,p=float('inf')))
            ans += log(kp) + log(N) - log(nx) - log(ny)
        else:
            nx = len(tree_x.query_ball_point(x[i],knn_dis[i]-1e-15,p=float('inf')))
            ny = len(tree_y.query_ball_point(y[i],knn_dis[i]-1e-15,p=float('inf')))
            ans += digamma(kp) + digamma(N) - digamma(nx) - digamma(ny)
    return ans/N


def cal_mi(i, j, x, y, count):
    '''
    Estimate the mutual information I(X;Y) of X and Y from samples {x_i, y_i}_{i=1}^N
    Using *Mixed-KSG* mutual information estimator

    Input: 
    i: name of TF
    j: name of TG
    x: the expression values of TF i
    y: the expression values of TG j
    count: a list including the num of cells in different branches
    MAXD: int, the maximum time lag

    Output: a dict with keys ['Gene1', 'Gene2']
    '''
    '''
    maxDelay = floor(min(count)/3)
    dc = []
    
    for d in range(maxDelay):
        xraw = x.copy()
        yraw = y.copy()
        sums = 0
        for k in range(len(count)):
            sums += count[k]

            # source gene
            idx_x = list(range(sums-(k+1)*maxDelay)) + list(range(sums-k*maxDelay, len(xraw)))
            xraw = xraw[idx_x]
            
            # target gene
            idx_y = list(range(sums-count[k]-k*maxDelay)) + list(range(sums-count[k]+d-k*maxDelay, len(yraw)))
            yraw = yraw[idx_y]
            idx_y = list(range(sums-maxDelay-k*maxDelay)) + list(range(sums-k*maxDelay-d, len(yraw)))
            yraw = yraw[idx_y]

        # calculate the dc, [0,1]
        d = dcor.distance_correlation(xraw, yraw)
        dc.append(d)
    dc = np.array(dc)
    MAXD = np.argmax(dc)  
'''
    MAXD = 1  # Fixed lag = 1

    sums = 0
    # Xt_L:Xt-2, Xt-1, Xt
    # Yt_L:Yt-2, Yt-1, Yt
    for k in range(len(count)):
        xraw = x.copy()
        yraw = y.copy()
        for d in [0, MAXD]:
            idx = list(range(sums+d, sums+count[k]-MAXD+d))
            if d == 0:
                Xt_d = xraw[idx]
                if MAXD == 0:
                    Yt_d = yraw[idx]
                    break
            else:
                Yt_d = yraw[idx]
        if k == 0:
            Xt_L = Xt_d
            Yt_L = Yt_d
        else:
            Xt_L = np.r_[Xt_L, Xt_d]
            Yt_L = np.r_[Yt_L, Yt_d]
        sums += count[k]

    if Xt_L.ndim == 1:
        Xt_L = Xt_L.reshape((len(Xt_L),1))
    if Yt_L.ndim == 1:
        Yt_L = Yt_L.reshape((len(Yt_L),1))

    # I(Xt-L;Yt)
    mi = max(MI_Gao(Xt_L, Yt_L), 0)

    return {'Gene1':i, 'Gene2':j, 'score':mi}


def cal_mi2(data, count, n_jobs=1, TF_set=[]):
    
    print('---------- data.shape=', data.shape)

    if len(TF_set) == 0:
        gene_combs = list(permutations(data.columns.values, 2))
    else:
        TG_set = set(data.columns)
        gene_combs = product(TF_set, TG_set)
    gene_combs = filter(lambda x: x[0]!=x[1], gene_combs)
    params = [[v[0], v[1], data[v[0]].values, data[v[1]].values, count] for v in gene_combs]
    result = pqdm(params, cal_mi, n_jobs=n_jobs, argument_type='args', desc='Computations of MI')
    df_res = pd.DataFrame.from_dict(result)
    return df_res  
    
def cal_mi_divergence(i, j, x, y):
    """Wraps KL divergence in a dict output format for pqdm"""
    score = kl_divergence(x, y)
    return {'Gene1': i, 'Gene2': j, 'score': score}

def cal_mi2_divergence(data, n_jobs=1, TF_set=[]):

    print(f'---------- data.shape= {data.shape}')
    if len(TF_set) == 0:
        gene_combs = list(permutations(data.columns.values, 2))
    else:
        TG_set = set(data.columns)
        gene_combs = product(TF_set, TG_set)
    gene_combs = filter(lambda x: x[0] != x[1], gene_combs)
    params = [[v[0], v[1], data[v[0]].values, data[v[1]].values] for v in gene_combs]
    result = pqdm(params, cal_mi_divergence, n_jobs=n_jobs, argument_type='args', desc='Computations of MI (divergence)')
    df_res = pd.DataFrame.from_dict(result)
    return df_res
     

def compute_optimal_lag(x, y, max_lag=None):
    """
    Compute optimal time lag using distance correlation.
    x: divergence profile of TF
    y: divergence profile of TG
    max_lag: maximum lag to consider (default = 1/3 length)
    """
    N = len(x)
    if max_lag is None:
        max_lag = N // 3
    dcorrs = []

    for lag in range(max_lag + 1):
        # Align series with lag
        if lag == 0:
            x_lagged = x
            y_lagged = y
        else:
            x_lagged = x[:-lag]
            y_lagged = y[lag:]
        
        # compute distance correlation
        d = dcor.distance_correlation(x_lagged, y_lagged)
        dcorrs.append(d)
    
    # find lag with max distance correlation
    optimal_lag = np.argmax(dcorrs)
    return optimal_lag

########################################### 
# forward KL based score
def kl_divergence(p, q):
    """Compute KL(P||Q) between empirical distributions of two genes."""
    # Smooth to avoid zeros (critical for scRNA-seq)
    p_smoothed = (p + 1e-10) / (np.sum(p) + 1e-10)
    q_smoothed = (q + 1e-10) / (np.sum(q) + 1e-10)
    return entropy(p_smoothed, q_smoothed)

def cal_mi_kl(i, j, x, y, count):
    """Compute KL(X_{t-1} || Y_t) for fixed lag=1."""
    x_aligned = x[:-1]  # X at t-1
    y_aligned = y[1:]   # Y at t
    kl = kl_divergence(x_aligned, y_aligned)
    return {'Gene1': i, 'Gene2': j, 'score': kl}

# symmetric KL based score:
def symmetric_kl_divergence(p, q):
    """Compute symmetric KL divergence: D_KL(P||Q) + D_KL(Q||P)"""
    p_smoothed = (p + 1e-10) / (np.sum(p) + 1e-10)
    q_smoothed = (q + 1e-10) / (np.sum(q) + 1e-10)

    kl_pq = entropy(p_smoothed, q_smoothed)
    kl_qp = entropy(q_smoothed, p_smoothed)
    
    return kl_pq + kl_qp

def cal_kl_symmetric(i, j, x, y, count):
    """Compute symmetric KL divergence between X_{t-1} and Y_t."""
    x_aligned = x[:-1]
    y_aligned = y[1:]
    skl = symmetric_kl_divergence(x_aligned, y_aligned)
    return {'Gene1': i, 'Gene2': j, 'score': skl}

def cal_kl2(data, n_jobs=1, TF_set=[]):
    print(f'---------- data.shape= {data.shape}')
    if len(TF_set) == 0:
        gene_combs = list(permutations(data.columns.values, 2))
    else:
        TG_set = set(data.columns)
        gene_combs = product(TF_set, TG_set)
    gene_combs = filter(lambda x: x[0] != x[1], gene_combs)
    params = [[v[0], v[1], data[v[0]].values, data[v[1]].values, []] for v in gene_combs]
    result = pqdm(params, cal_mi_kl, n_jobs=n_jobs, argument_type='args', desc='Computations of KL')
    df_res = pd.DataFrame.from_dict(result)
    return df_res


def cal_kl2_symmetric(data, n_jobs=1, TF_set=[]):
    print(f'---------- data.shape= {data.shape}')
    if len(TF_set) == 0:
        gene_combs = list(permutations(data.columns.values, 2))
    else:
        TG_set = set(data.columns)
        gene_combs = product(TF_set, TG_set)
    gene_combs = filter(lambda x: x[0] != x[1], gene_combs)
    params = [[v[0], v[1], data[v[0]].values, data[v[1]].values, []] for v in gene_combs]
    result = pqdm(params, cal_kl_symmetric, n_jobs=n_jobs, argument_type='args', desc='Computations of Symmetric KL')
    df_res = pd.DataFrame.from_dict(result)
    return df_res

#JS-based score:
def js_divergence(p, q):
    """Compute Jensen-Shannon divergence between p and q"""
    p_smoothed = (p + 1e-10) / (np.sum(p) + 1e-10)
    q_smoothed = (q + 1e-10) / (np.sum(q) + 1e-10)
    m = 0.5 * (p_smoothed + q_smoothed)
    return 0.5 * entropy(p_smoothed, m) + 0.5 * entropy(q_smoothed, m)

def cal_js(i, j, x, y, count):
    """Compute JS divergence between X_{t-1} and Y_t."""
    x_aligned = x[:-1]
    y_aligned = y[1:]
    js = js_divergence(x_aligned, y_aligned)
    return {'Gene1': i, 'Gene2': j, 'score': js}

def cal_js2(data, n_jobs=1, TF_set=[]):
    print(f'---------- data.shape= {data.shape}')
    if len(TF_set) == 0:
        gene_combs = list(permutations(data.columns.values, 2))
    else:
        TG_set = set(data.columns)
        gene_combs = product(TF_set, TG_set)
    gene_combs = filter(lambda x: x[0] != x[1], gene_combs)
    params = [[v[0], v[1], data[v[0]].values, data[v[1]].values, []] for v in gene_combs]
    result = pqdm(params, cal_js, n_jobs=n_jobs, argument_type='args', desc='Computations of JS divergence')
    df_res = pd.DataFrame.from_dict(result)
    return df_res
###backward KL:
def cal_backward(i, j, x, y, count):
    """Compute backward KL: KL(Y_t || X_{t-1})"""
    x_aligned = x[:-1]
    y_aligned = y[1:]
    return {'Gene1': i, 'Gene2': j, 'score': entropy(y_aligned, x_aligned)}  # assume already smoothed

def cal_backward_kl(data, n_jobs=1, TF_set=[]):
    print(f"---------- data.shape= {data.shape}")
    if len(TF_set) == 0:
        gene_combs = list(permutations(data.columns.values, 2))
    else:
        TG_set = set(data.columns)
        gene_combs = product(TF_set, TG_set)

    gene_combs = filter(lambda x: x[0] != x[1], gene_combs)
    params = [[i, j, data[i].values, data[j].values, []] for (i, j) in gene_combs]

    result = pqdm(params, cal_backward, n_jobs=n_jobs, argument_type='args', desc='Backward KL Computation')
    df_res = pd.DataFrame.from_dict(result)
    return df_res

def cal_wasserstein(i, j, x, y, count):
    return {'Gene1': i, 'Gene2': j, 'score': wasserstein_distance(x[:-1], y[1:])}

def cal_wasserstein2(data, n_jobs=1, TF_set=[]):
    if len(TF_set) == 0:
        gene_combs = list(permutations(data.columns.values, 2))
    else:
        TG_set = set(data.columns)
        gene_combs = product(TF_set, TG_set)

    gene_combs = filter(lambda x: x[0] != x[1], gene_combs)
    params = [[i, j, data[i].values, data[j].values, []] for (i, j) in gene_combs]

    result = pqdm(params, cal_wasserstein, n_jobs=n_jobs, argument_type='args', desc='Wasserstein Computation')
    return pd.DataFrame(result)


def cal_energy(i, j, x, y, count):
    try:
        score = energy_distance(x[:-1], y[1:])
        return {'Gene1': i, 'Gene2': j, 'score': score}
    except Exception as e:
        print(f"Error in cal_energy({i}, {j}): {e}")
        return {'Gene1': i, 'Gene2': j, 'score': np.nan}

def cal_energy2(data, n_jobs=1, TF_set=[]):
    # 1. Set up gene combinations
    if len(TF_set) == 0:
        gene_combs = list(permutations(data.columns.values, 2))
    else:
        TG_set = set(data.columns)
        gene_combs = list(product(TF_set, TG_set))

    gene_combs = list(filter(lambda x: x[0] != x[1], gene_combs))
    print(f"Total gene pairs to process: {len(gene_combs)}")

    # 2. Build parameter list
    params = [[i, j, data[i].values, data[j].values, []] for (i, j) in gene_combs]
    print(f"Example param: {params[0][:2]}")

    # 3. Run parallel processing
    result = pqdm(params, cal_energy, n_jobs=n_jobs, argument_type='args', desc='Energy Distance Computation')

    # 4. Debug: check result structure
    print("First 3 results returned:")
    for item in result[:3]:
        print(item)

    # 5. Convert to DataFrame
    if isinstance(result, list) and isinstance(result[0], dict) and "Gene1" in result[0]:
        df = pd.DataFrame(result)
        print("DataFrame columns:", df.columns.tolist())
        return df
    else:
        raise ValueError("Unexpected result structure from pqdm.")



def cal_cramer(i, j, x, y, count):
    try:
        score = dcor.distance_correlation(x[:-1], y[1:])
    except Exception:
        score = np.nan
    return {'Gene1': i, 'Gene2': j, 'score': score}


def cal_cramer2(data, n_jobs=1, TF_set=[]):
    from itertools import permutations, product
    from pqdm.processes import pqdm

    if len(TF_set) == 0:
        gene_combs = list(permutations(data.columns.values, 2))
    else:
        TG_set = set(data.columns)
        gene_combs = product(TF_set, TG_set)

    gene_combs = filter(lambda x: x[0] != x[1], gene_combs)
    params = [[i, j, data[i].values, data[j].values, []] for (i, j) in gene_combs]

    result = pqdm(params, cal_cramer, n_jobs=n_jobs, argument_type='args', desc='Cramér Distance Computation')

    # Debug print to confirm structure
    print("Cramér result preview:", result[:3])
    
    return pd.DataFrame(result)



#  Unified Divergence Runner 
def run_divergence(df, score_func, n_jobs=4, TF_set=[]):
    if len(TF_set) == 0:
        gene_combs = list(permutations(df.columns, 2))
    else:
        TG_set = set(df.columns)
        gene_combs = product(TF_set, TG_set)

    gene_combs = filter(lambda x: x[0] != x[1], gene_combs)
    params = [[i, j, df[i].values, df[j].values, []] for (i, j) in gene_combs]
    result = pqdm(params, score_func, n_jobs=n_jobs, argument_type='args', desc=f"Score: {score_func.__name__}")
    df_res = pd.DataFrame.from_dict(result)
    return df_res

