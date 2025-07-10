#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 14:28:04 2025

@author: linglingzhang
"""
# Replace the Slingshot
import pandas as pd
import scanpy as sc
import phate
import matplotlib.pyplot as plt
from EstimateMI import compute_divergence_average_3lags
from Evaluate import add_sign_and_plot
import os
os.makedirs("outputtimelag=1", exist_ok=True)



# Load Data 
df_exp = pd.read_csv("input/ExpressionData.csv", index_col=0)
adata = sc.AnnData(df_exp)
adata.var_names_make_unique()

#  PHATE 
print("=== Running PHATE ===")
phate_operator = phate.PHATE(n_components=2)
adata.obsm["X_phate"] = phate_operator.fit_transform(adata.X)

# Compute pseudotime as distance from first cell in PHATE space
from sklearn.metrics import pairwise_distances
root_cell_index = 0
distances = pairwise_distances([adata.obsm["X_phate"][root_cell_index]], adata.obsm["X_phate"])[0]
adata.obs["phate_pseudotime"] = distances

# Save PHATE pseudotime
df_phate = pd.DataFrame({
    "cell_id": adata.obs_names,
    "Pseudotime": adata.obs["phate_pseudotime"].values
})
df_phate.to_csv("phate_pseudotime_timlag=1.csv", index=False)

#  Diffusion Maps 
print("=== Running Diffusion Maps ===")
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.diffmap(adata)
adata.uns['iroot'] = adata.obsm["X_diffmap"][:, 0].argmin()
sc.tl.dpt(adata)

# Save diffusion pseudotime
df_diff = pd.DataFrame({
    "cell_id": adata.obs_names,
    "Pseudotime": adata.obs["dpt_pseudotime"].values
})
df_diff.to_csv("diffmap_pseudotime_timlag=1.csv", index=False)
print("Saved: diffmap_pseudotime.csv")

#  PCA-based Pseudotime 
print("=== Running PCA-based Pseudotime ===")
pc1 = adata.obsm["X_pca"][:, 0]
pc1_minmax = (pc1 - pc1.min()) / (pc1.max() - pc1.min())
adata.obs["pca_pseudotime"] = pc1_minmax

# Save PCA pseudotime
df_pca = pd.DataFrame({
    "cell_id": adata.obs_names,
    "Pseudotime": adata.obs["pca_pseudotime"].values
})
df_pca.to_csv("pca_pseudotime_timlag=1.csv", index=False)
print("Saved: pca_pseudotime.csv")

# PAGA 
print("=== Running PAGA ===")
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata, resolution=1.0)
sc.tl.paga(adata, groups='leiden')
sc.pl.paga(adata, show=False)  # <-- Add this line!
sc.tl.draw_graph(adata, init_pos='paga')
sc.tl.dpt(adata)

# Save PAGA pseudotime
df_paga = pd.DataFrame({
    "cell_id": adata.obs_names,
    "Pseudotime": adata.obs["dpt_pseudotime"].values
})
df_paga.to_csv("paga_pseudotime_timlag=1.csv", index=False)
print("Saved: paga_pseudotime.csv")

###############################################
# Using above 4 methods get auc and roc:
import pandas as pd
from PreprocessData import smooth_divergence
from EstimateMI import cal_mi2_divergence as cal_mutual_information
from mRMR import MRMR2_divergence
from Evaluate import concat_ref, cal_auc_aupr

# Load expression and reference network
df_exp = pd.read_csv("input/ExpressionData.csv", index_col=0)
df_ref = pd.read_csv("input/refNetwork.csv", usecols=['Gene1', 'Gene2'])

# Define pseudotime CSVs (from 4 methods)
pseudotime_files = {
    "phate": "phate_pseudotime_timlag=1.csv",
    "diffmap": "diffmap_pseudotime_timlag=1.csv",
    "pca": "pca_pseudotime_timlag=1.csv",
    "paga": "paga_pseudotime_timlag=1.csv"
}


# Distances to test
distance_list = [1, 5, 6, 7, 8, 9, 10, 11]

# Save results
all_results = []

for method_name, pseudo_file in pseudotime_files.items():
    print(f"\n Evaluating method: {method_name} ")
    df_pse = pd.read_csv(pseudo_file, index_col=0)

    method_results = []
    for distance in distance_list:
        print(f"--- Normi timelag=1 with {method_name} pseudotime and divergence distance={distance} ---")
        num_windows, df_exp_smooth = smooth_divergence(df_pse, df_exp, slide=1, k=5, distance=distance)
        df_mi = cal_mutual_information(df_exp_smooth, n_jobs=1)
        
            
        df_mi = df_mi[df_mi.score > 0].sort_values(by="score", ascending=False)
        df_mrmr = MRMR2_divergence(df_mi, n_jobs=1)
        df_mrmr = df_mrmr[df_mrmr.score > 0].sort_values(by="score", ascending=False)

        out_file = f"outputtimelag=1/rankedEdges_{method_name}_dist{distance}.csv"

        df_mrmr.to_csv(out_file, index=False)
        #add network plot pdfs:
        pdf_out = f"outputtimelag=1/GRN_{method_name}_dist{distance}.pdf"


        add_sign_and_plot(
            df_edge=df_mrmr,
            expression_file="input/ExpressionData.csv",
            top_k=100,
            plot=False,
            output_pdf=pdf_out
        )
        df_eval = concat_ref(df_mrmr, df_ref)
        res = cal_auc_aupr(df_eval)
        print(f"→ AUROC={res['AUROC']:.4f}, AUPRC={res['AUPRC']:.4f}")

        method_results.append({
            'Method': method_name, 'Distance': distance,
            'AUROC': res['AUROC'], 'AUPRC': res['AUPRC']
        })
    all_results.extend(method_results)

# Save final summary
df_all = pd.DataFrame(all_results)
df_all.to_csv("outputtimelag=1/Divergence_AllMethods_Summary_timlag=1.csv", index=False)
print("\n=== Finished! Summary saved to 'Divergence_AllMethods_Summary_timlag=1.csv' ===")

