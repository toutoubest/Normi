#important packages:
#pip install pandas numpy scipy matplotlib tqdm dcor pqdm scikit-learn(terminal)
import pandas as pd
from PreprocessData import smooth_divergence
from EstimateMI import cal_mi2_divergence as cal_mutual_information
#from mRMR import MRMR2
from mRMR import MRMR2_divergence
from Evaluate import concat_ref, cal_auc_aupr

# Load data
df_exp = pd.read_csv("input/ExpressionData.csv", index_col=0)
df_pse = pd.read_csv("input/PseudoTime.csv", index_col=0)
df_ref = pd.read_csv("input/refNetwork.csv", usecols=['Gene1', 'Gene2'])

# Distances to test
distance_list = [1, 5, 6, 7, 8]

# Store results
results = []

for distance in distance_list:
    print(f"\n=== Running Normi with divergence distance={distance} ===")

    # STEP 1: Preprocessing (divergence only)
    num_windows, df_exp_smooth = smooth_divergence(df_pse, df_exp, slide=1, k=5, distance=distance)

    # STEP 2: Estimate MI using divergence-smoothed input
    df_mi = cal_mutual_information(df_exp_smooth, n_jobs=1)
    df_mi = df_mi[df_mi.score > 0]
    df_mi.sort_values(by="score", ascending=False, inplace=True)

    # STEP 3: Filter redundant edges using mRMR
    #df_mrmr = MRMR2(df_mi, n_jobs=1)
    df_mrmr = MRMR2_divergence(df_mi, n_jobs=1)
    df_mrmr = df_mrmr[df_mrmr.score > 0]
    df_mrmr.sort_values(by="score", ascending=False, inplace=True)

    # Save edge list
    out_file = f"rankedEdges_TimeLag_dist{distance}.csv"
    df_mrmr.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

    # STEP 4: Evaluation
    df_eval = concat_ref(df_mrmr, df_ref)
    res = cal_auc_aupr(df_eval)
    print(f"Distance {distance} → AUROC={res['AUROC']:.4f}, AUPRC={res['AUPRC']:.4f}")

    # Save result
    results.append({'Distance': distance, 'AUROC': res['AUROC'], 'AUPRC': res['AUPRC']})

# Save summary table
df_summary = pd.DataFrame(results)
df_summary.to_csv("Divergence_Results_TimeLag=1_mrmrD_Summary.csv", index=False)
print("\n=== All done! Summary saved to Divergence_Results_TimeLag=1_mrmrD_Summary.csv ===")



##############################################
#replace the Eq8 and 9 with some divergences, and still use slingshot, 07/13:
import pandas as pd
import os
from EstimateMI import cal_kl2, cal_kl2_symmetric, cal_js2
from mRMR import MRMR2_kl, MRMR2_divergence
from Evaluate import concat_ref, cal_auc_aupr, add_sign_and_plot

# Load data
df_exp = pd.read_csv("input/ExpressionData.csv", index_col=0)
df_pse = pd.read_csv("input/PseudoTime.csv", index_col=0).squeeze("columns")
df_ref = pd.read_csv("input/refNetwork.csv", usecols=['Gene1', 'Gene2'])

# Output folder
output_dir = "outputtimelag=1divergence"
os.makedirs(output_dir, exist_ok=True)

# Distances to test
distance_list = [5, 6, 7, 8, 9, 10, 11]

summary_rows = []

for dist in distance_list:
    print(f"\n=== Running for Distance = {dist} ===")

    # Step 1: Smooth expression based on current distance divergence 
    count, df_exp_sorted = smooth_divergence(df_pse, df_exp, distance=dist)

    # Step 2: Forward KL score
    df_kl = cal_kl2(df_exp_sorted, n_jobs=4)
    df_kl.to_csv(f"{output_dir}/forward_kl_score_d{dist}.csv", index=False)

    df_mrmr_kl = MRMR2_kl(df_kl, n_jobs=4)
    df_mrmr_kl.to_csv(f"{output_dir}/forward_kl_mrmr_d{dist}.csv", index=False)

    df_eval_kl = concat_ref(df_mrmr_kl, df_ref)
    df_eval_kl.to_csv(f"{output_dir}/forward_kl_eval_d{dist}.csv", index=False)

    res_kl = cal_auc_aupr(df_eval_kl)
    pd.DataFrame([res_kl]).to_csv(f"{output_dir}/forward_kl_auc_d{dist}.csv", index=False)

    add_sign_and_plot(df_mrmr_kl.copy(), "input/ExpressionData.csv", top_k=50,
                      output_pdf=f"{output_dir}/network_forward_kl_d{dist}.pdf", plot=False)

    # ===== Step 3: Symmetric KL score =====
    df_skl = cal_kl2_symmetric(df_exp_sorted, n_jobs=4)
    df_skl.to_csv(f"{output_dir}/symmetric_kl_score_d{dist}.csv", index=False)

    df_mrmr_skl = MRMR2_divergence(df_skl, n_jobs=4)
    df_mrmr_skl.to_csv(f"{output_dir}/symmetric_kl_mrmr_d{dist}.csv", index=False)

    df_eval_skl = concat_ref(df_mrmr_skl, df_ref)
    df_eval_skl.to_csv(f"{output_dir}/symmetric_kl_eval_d{dist}.csv", index=False)

    res_skl = cal_auc_aupr(df_eval_skl)
    pd.DataFrame([res_skl]).to_csv(f"{output_dir}/symmetric_kl_auc_d{dist}.csv", index=False)

    add_sign_and_plot(df_mrmr_skl.copy(), "input/ExpressionData.csv", top_k=50,
                      output_pdf=f"{output_dir}/network_symmetric_kl_d{dist}.pdf", plot=False)

    # Step 4: JS Divergence score 
    df_js = cal_js2(df_exp_sorted, n_jobs=4)
    df_js.to_csv(f"{output_dir}/js_score_d{dist}.csv", index=False)

    df_mrmr_js = MRMR2_divergence(df_js, n_jobs=4)
    df_mrmr_js.to_csv(f"{output_dir}/js_mrmr_d{dist}.csv", index=False)

    df_eval_js = concat_ref(df_mrmr_js, df_ref)
    df_eval_js.to_csv(f"{output_dir}/js_eval_d{dist}.csv", index=False)

    res_js = cal_auc_aupr(df_eval_js)
    pd.DataFrame([res_js]).to_csv(f"{output_dir}/js_auc_d{dist}.csv", index=False)

    add_sign_and_plot(df_mrmr_js.copy(), "input/ExpressionData.csv", top_k=50,
                      output_pdf=f"{output_dir}/network_js_d{dist}.pdf", plot=False)

    #  Step 5: Add to Summary Table
    summary_rows.append({
        'Distance': dist,
        'AUROC_forward_kl': res_kl['AUROC'],
        'AUPRC_forward_kl': res_kl['AUPRC'],
        'AUROC_symmetric_kl': res_skl['AUROC'],
        'AUPRC_symmetric_kl': res_skl['AUPRC'],
        'AUROC_js': res_js['AUROC'],
        'AUPRC_js': res_js['AUPRC'],
    })

# ===== Final Summary CSV =====
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(f"{output_dir}/Divergence_Summary_Slingshot.csv", index=False)

print("\n=== Final Summary Table ===")
print(df_summary)

print("\n All done. Outputs saved in:", output_dir)
