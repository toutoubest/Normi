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
