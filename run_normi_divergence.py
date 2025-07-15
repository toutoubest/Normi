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
distance_list = [1, 5, 6, 7, 8, 9, 10, 11]

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

#############################
# this is 5 runs using the forward KL, symmetric Kl, JS based score:
import os
import numpy as np
import pandas as pd
from EstimateMI import cal_kl2, cal_kl2_symmetric, cal_js2
from EstimateMI import cal_backward_kl2, cal_wasserstein2, cal_energy2, cal_cramer2
from mRMR import MRMR2_kl, MRMR2_divergence
from Evaluate import concat_ref, cal_auc_aupr, add_sign_and_plot
from PreprocessData import smooth


def add_noise(df_expr, noise_level=0.05):
    """Add Gaussian noise based on expression range."""
    ranges = (df_expr.max() - df_expr.min()).values  # shape: (genes,)
    noise = np.random.randn(*df_expr.shape) * ranges  # shape: (cells, genes)
    df_noisy = df_expr + noise * noise_level
    df_noisy[df_noisy < 0] = 0  # ensure non-negative
    return df_noisy


#path
df_exp_orig = pd.read_csv("input/ExpressionData.csv", index_col=0)
df_pse = pd.read_csv("input/PseudoTime.csv", index_col=0)
df_ref = pd.read_csv("input/refNetwork.csv", usecols=["Gene1", "Gene2"])
output_dir = "output_normi_divergence_timelag1"
os.makedirs(output_dir, exist_ok=True)

#  Divergence setups
divergences = {
    "forward_kl": (cal_kl2, MRMR2_kl),
    "symmetric_kl": (cal_kl2_symmetric, MRMR2_divergence),
    "js": (cal_js2, MRMR2_divergence),
}

summary_all = []

for run in range(1, 6):
    print(f"\n=== RUN {run} ===")

    # Step 1: Add noise
    df_exp_noisy = add_noise(df_exp_orig)
    noise_path = f"{output_dir}/ExpressionData_run{run}.csv"
    df_exp_noisy.to_csv(noise_path)

    # Step 2: Average smoothing (Normi style)
    count, df_smoothed = smooth(df_pse, df_exp_noisy, slipe=1, k=5)

    # Collect results for each divergence
    run_summary = {"Run": run}
    for name, (score_func, mrmr_func) in divergences.items():
        print(f"  >> {name}")
        df_score = score_func(df_smoothed, n_jobs=4)
        df_mrmr = mrmr_func(df_score, n_jobs=4)
        df_eval = concat_ref(df_mrmr, df_ref)

        # Clean invalid scores
        df_eval = df_eval[np.isfinite(df_eval["score"])]

        result = cal_auc_aupr(df_eval)
        run_summary[f"{name}_AUROC"] = result["AUROC"]
        run_summary[f"{name}_AUPRC"] = result["AUPRC"]

        # Plot network
        net_plot_path = f"{output_dir}/network_{name}_run{run}.pdf"
        add_sign_and_plot(df_mrmr.copy(), "input/ExpressionData.csv", top_k=50,
                          output_pdf=net_plot_path, plot=False)

    summary_all.append(run_summary)

# Save summary
df_summary = pd.DataFrame(summary_all)
df_summary.to_csv(f"{output_dir}/summary_divergence_5runs.csv", index=False)
print("Saved summary to:", f"{output_dir}/summary_divergence_5runs.csv")

###############################
#use some other distance-based score, like the backward KL, wass, energy,cramer:
import os
import numpy as np
import pandas as pd
from EstimateMI import cal_backward_kl, cal_wasserstein, cal_wasserstein2, cal_energy, cal_energy2, cal_cramer, cal_cramer2
from mRMR import MRMR2_divergence
from Evaluate import concat_ref, cal_auc_aupr, add_sign_and_plot
from PreprocessData import smooth

def add_noise(df_expr, noise_level=0.05):
    """Add Gaussian noise based on expression range."""
    ranges = (df_expr.max() - df_expr.min()).values
    noise = np.random.randn(*df_expr.shape) * ranges
    df_noisy = df_expr + noise * noise_level
    df_noisy[df_noisy < 0] = 0
    return df_noisy

#  Paths
df_exp_orig = pd.read_csv("input/ExpressionData.csv", index_col=0)
df_pse = pd.read_csv("input/PseudoTime.csv", index_col=0)
df_ref = pd.read_csv("input/refNetwork.csv", usecols=["Gene1", "Gene2"])
output_dir = "outputtimelag=1_4divergence"
os.makedirs(output_dir, exist_ok=True)

#  Define 4 divergence methods 
from EstimateMI import cal_backward_kl, cal_wasserstein, cal_energy, cal_cramer

divergences = {
    "backward_kl": cal_backward_kl,
    "wasserstein": cal_wasserstein2,  # use wrapper, not raw func
    "energy": cal_energy2,
    "cramer": cal_cramer2,
}


summary_all = []

for run in range(1, 6):
    print(f"\n=== RUN {run} ===")
    df_exp_noisy = add_noise(df_exp_orig)
    df_exp_noisy.to_csv(f"{output_dir}/ExpressionData_run{run}.csv")

    # Step 1: Average smoothing
    count, df_smoothed = smooth(df_pse, df_exp_noisy, slipe=1, k=5)

    run_summary = {"Run": run}
    for name, score_func in divergences.items():
        print(f"  >> {name}")
        df_score = score_func(df_smoothed, n_jobs=4)
        if df_score.empty:
            print(f"    Skipped {name} due to empty score matrix")
            run_summary[f"{name}_AUROC"] = np.nan
            run_summary[f"{name}_AUPRC"] = np.nan
            continue

        df_mrmr = MRMR2_divergence(df_score, n_jobs=4)
        df_eval = concat_ref(df_mrmr, df_ref)
        df_eval = df_eval[np.isfinite(df_eval["score"])]
        result = cal_auc_aupr(df_eval)
        run_summary[f"{name}_AUROC"] = result["AUROC"]
        run_summary[f"{name}_AUPRC"] = result["AUPRC"]

        # Save network plot
        plot_path = f"{output_dir}/network_{name}_run{run}.pdf"
        add_sign_and_plot(df_mrmr.copy(), "input/ExpressionData.csv", top_k=50,
                          output_pdf=plot_path, plot=False)

    summary_all.append(run_summary)

# Save summary table
df_summary = pd.DataFrame(summary_all)
df_summary.to_csv(f"{output_dir}/summary_4divergence_5runs.csv", index=False)
print("Saved summary to:", f"{output_dir}/summary_4divergence_5runs.csv")
# === Calculate mean and standard deviation ===
df_mean = df_summary.mean(numeric_only=True).to_frame(name='Mean').T
df_std = df_summary.std(numeric_only=True).to_frame(name='Std').T

# Combine and save
df_stats = pd.concat([df_mean, df_std], axis=0)
df_stats.to_csv(f"{output_dir}/summary_4divergence_mean_std.csv", index=True)
print("Saved mean/std summary to:", f"{output_dir}/summary_4divergence_mean_std.csv")
