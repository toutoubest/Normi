
#############################
# this is 5 runs using the forward KL, symmetric Kl, JS based score:
import os
import numpy as np
import pandas as pd
from psedoScore import cal_kl2, cal_kl2_symmetric, cal_js2
from psedoScor import cal_backward_kl2, cal_wasserstein2, cal_energy2, cal_cramer2
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
from psedoScore import cal_backward_kl, cal_wasserstein, cal_wasserstein2, cal_energy, cal_energy2, cal_cramer, cal_cramer2
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
####################add some pearson divergence, JS like, neyman, symmetric pearson based score:
import os
import numpy as np
import pandas as pd
from psedoScore import (
    cal_pearson2, cal_symmetric_pearson2,
    cal_js_pearson2, cal_neyman2
)
from mRMR import MRMR2_divergence
from Evaluate import concat_ref, cal_auc_aupr, add_sign_and_plot
from PreprocessData import smooth

def add_noise(df_expr, noise_level=0.05):
    """Add Gaussian noise to expression matrix."""
    ranges = (df_expr.max() - df_expr.min()).values
    noise = np.random.randn(*df_expr.shape) * ranges
    df_noisy = df_expr + noise * noise_level
    df_noisy[df_noisy < 0] = 0
    return df_noisy

#  Input paths 
df_exp_orig = pd.read_csv("input/ExpressionData.csv", index_col=0)
df_pse = pd.read_csv("input/PseudoTime.csv", index_col=0)
df_ref = pd.read_csv("input/refNetwork.csv", usecols=["Gene1", "Gene2"])
output_dir = "outputslingshotpearsondivergence"
os.makedirs(output_dir, exist_ok=True)

#  Pearson-based divergence methods
divergences = {
    "pearson": cal_pearson2,
    "symmetric_pearson": cal_symmetric_pearson2,
    "js_pearson": cal_js_pearson2,
    "neyman": cal_neyman2,
}

summary_all = []

for run in range(1, 6):
    print(f"\n=== RUN {run} ===")
    df_exp_noisy = add_noise(df_exp_orig)
    noisy_path = f"{output_dir}/ExpressionData_run{run}.csv"
    df_exp_noisy.to_csv(noisy_path)

    # Step 1: smooth expression by pseudotime
    count, df_smoothed = smooth(df_pse, df_exp_noisy, slipe=1, k=5)

    run_summary = {"Run": run}

    for name, score_func in divergences.items():
        print(f"  >> {name}")
        df_score = score_func(df_smoothed, n_jobs=4)
        if df_score.empty:
            print(f"    [Skipped] {name} is empty")
            run_summary[f"{name}_AUROC"] = np.nan
            run_summary[f"{name}_AUPRC"] = np.nan
            continue

        # Step 2: Apply MRMR
        df_score = score_func(df_smoothed, n_jobs=4)  
        df_score = df_score.stack().reset_index()
        df_score.columns = ['Gene1', 'Gene2', 'score']  
        df_mrmr = MRMR2_divergence(df_score, n_jobs=4)

        # Step 3: Evaluate with reference
        df_eval = concat_ref(df_mrmr, df_ref)
        df_eval = df_eval[np.isfinite(df_eval["score"])]
        result = cal_auc_aupr(df_eval)
        run_summary[f"{name}_AUROC"] = result["AUROC"]
        run_summary[f"{name}_AUPRC"] = result["AUPRC"]

        # Step 4: Plot top-50 network
        plot_path = f"{output_dir}/network_{name}_run{run}.pdf"
        add_sign_and_plot(df_mrmr.copy(), "input/ExpressionData.csv", top_k=50,
                          output_pdf=plot_path, plot=False)

    summary_all.append(run_summary)

# Save summary 
df_summary = pd.DataFrame(summary_all)
df_summary.to_csv(f"{output_dir}/summary_pearson_divergence_5runs.csv", index=False)

# Calculate statistics (mean and std) for numeric columns only
numeric_cols = df_summary.select_dtypes(include=[np.number]).columns

# Create stats DataFrame
stats_data = {
    'Run': ['Mean', 'Std'],
    **{
        col: [
            df_summary[col].mean(),
            df_summary[col].std()
        ]
        for col in numeric_cols
    }
}

df_stats = pd.DataFrame(stats_data)

# Combine original data with statistics
df_combined = pd.concat([
    df_summary,
    df_stats
], ignore_index=True)

# Save combined results
df_combined.to_csv(
    f"{output_dir}/summary_pearson_divergence_5runs_with_stats.csv", 
    index=False
)

print("Combined summary with mean/std saved to:", 
      f"{output_dir}/summary_pearson_divergence_5runs_with_stats.csv")

########################using different lambda for all distances:
import os
import numpy as np
import pandas as pd
from PreprocessData import smooth
from psedoScore import (
    cal_kl2, cal_kl2_symmetric, cal_js2,
    cal_backward_kl, cal_wasserstein2, cal_energy2, cal_cramer2,
    cal_pearson2, cal_symmetric_pearson2, cal_js_pearson2, cal_neyman2
)
from mRMR import MRMR2_divergence
from Evaluate import concat_ref, cal_auc_aupr, add_sign_and_plot


def add_noise(df_expr, noise_level=0.05):
    """Add Gaussian noise to expression matrix."""
    ranges = (df_expr.max() - df_expr.min()).values
    noise = np.random.randn(*df_expr.shape) * ranges
    df_noisy = df_expr + noise * noise_level
    df_noisy[df_noisy < 0] = 0
    return df_noisy

#  Input Paths 
df_exp_orig = pd.read_csv("input/ExpressionData.csv", index_col=0)
df_pse = pd.read_csv("input/PseudoTime.csv", index_col=0)
df_ref = pd.read_csv("input/refNetwork.csv", usecols=["Gene1", "Gene2"])
output_dir = "output_slingshot_divergence10"
os.makedirs(output_dir, exist_ok=True)

#  Divergence Methods 
divergences = {
    "forwardKL": cal_kl2,
    "symmetricKL": cal_kl2_symmetric,
    "JS": cal_js2,
    "backwardKL": cal_backward_kl,
    "wasserstein": cal_wasserstein2,
    "energy": cal_energy2,
    "cramer": cal_cramer2,
    "pearson": cal_pearson2,
    "symmetric_pearson": cal_symmetric_pearson2,
    "js_pearson": cal_js_pearson2,
    "neyman": cal_neyman2,
}

#  Main Loop 
summary_all = []

lambda_vals = [0.1, 1, 1.5, 2, 2.5, 3, 5]  

for run in range(1, 6):
    print(f"\n=== RUN {run} ===")
    df_exp_noisy = add_noise(df_exp_orig)
    count, df_smoothed = smooth(df_pse, df_exp_noisy, slipe=1, k=5)
    run_summary = {"Run": run}

    for name, score_func in divergences.items():
        print(f"  >> {name}")
        df_score = score_func(df_smoothed, n_jobs=4)

        # Reshape matrix to long form
        if set(df_score.columns) == set(df_smoothed.columns):
            df_score = df_score.stack().reset_index()
            df_score.columns = ['Gene1', 'Gene2', 'score']

        df_score['score'] = pd.to_numeric(df_score['score'], errors='coerce')

        for lambda_val in lambda_vals:
            print(f"     -- λ = {lambda_val}")
            # Filter top-score edges (simulate sparsity)
            #df_score_lambda = df_score.copy()
            #threshold = df_score_lambda['score'].quantile(1 - lambda_val)
            #df_score_lambda = df_score_lambda[df_score_lambda['score'] >= threshold]
            df_mrmr = MRMR2_divergence(df_score, n_jobs=4, lambda_val=lambda_val)

            #df_mrmr = MRMR2_divergence(df_score_lambda, n_jobs=4)
            df_eval = concat_ref(df_mrmr, df_ref)
            df_eval = df_eval[np.isfinite(df_eval["score"])]

            result = cal_auc_aupr(df_eval)
            run_summary[f"{name}_λ{lambda_val}_AUROC"] = result["AUROC"]
            run_summary[f"{name}_λ{lambda_val}_AUPRC"] = result["AUPRC"]

            plot_path = f"{output_dir}/network_{name}_lambda{lambda_val}_run{run}.pdf"
            try:
                add_sign_and_plot(df_mrmr.copy(), "input/ExpressionData.csv", top_k=50,
                                  output_pdf=plot_path, plot=False)
                print(f"       [Plot Saved] {plot_path}")
            except Exception as e:
                print(f"       [Plot Error] {e}")

    summary_all.append(run_summary)

# Save Summary in Wide Format with All Runs
def save_results_wide_format_fixed(df_summary, output_dir):
    import os
    import pandas as pd

    # Convert long-style column names into parts
    df_long = pd.melt(
        df_summary,
        id_vars=['Run'],
        var_name='metric_lambda',
        value_name='score'
    )

    # Extract: distance, lambda, metric (AUROC/AUPRC)
    components = df_long['metric_lambda'].str.extract(
        r'(?P<distance>[a-zA-Z0-9]+)_λ(?P<lambda>[0-9.]+)_(?P<metric>AUROC|AUPRC)'
    )
    df_long = pd.concat([df_long, components], axis=1).dropna()

    # Convert lambda to float for sorting
    df_long['lambda'] = df_long['lambda'].astype(float)

    # Pivot to put AUROC and AUPRC as columns, with runs spread across rows
    df_pivot = df_long.pivot_table(
        index=['distance', 'lambda', 'Run'],
        columns='metric',
        values='score'
    ).reset_index()

    # Group by distance and lambda, then collect 5 runs per metric
    grouped = df_pivot.groupby(['distance', 'lambda'])

    output_rows = []
    for (dist, lam), group in grouped:
        row = {
            'pseudotime_method': 'Slingshot',
            'distance': dist,
            'lambda': lam
        }

        for i, r in enumerate(group.itertuples(), 1):
            row[f'AUROC_run{i}'] = r.AUROC
            row[f'AUPRC_run{i}'] = r.AUPRC

        # Add mean values
        row['AUROC_mean'] = group['AUROC'].mean()
        row['AUPRC_mean'] = group['AUPRC'].mean()
        output_rows.append(row)

    df_out = pd.DataFrame(output_rows)

    # Column order
    column_order = ['pseudotime_method', 'distance', 'lambda']
    for metric in ['AUROC', 'AUPRC']:
        column_order += [f"{metric}_run{i}" for i in range(1, 6)]
        column_order += [f"{metric}_mean"]

    df_out = df_out[column_order]

    # Save
    output_path = os.path.join(output_dir, "results_wide_format.csv")
    df_out.to_csv(output_path, index=False)
    print(f"Saved corrected wide format to:\n{output_path}")
    return df_out

################################## 07/18
############### we can use cross validation to find the best lambda for the auc of using cramer and plot the curve:

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from PreprocessData import smooth
from psedoScore import cal_cramer2
from mRMR import MRMR2_divergence
from Evaluate import concat_ref, cal_auc_aupr


output_dir = "output_cramer"
os.makedirs(output_dir, exist_ok=True)


df_exp = pd.read_csv("input/ExpressionData.csv", index_col=0)
df_pse = pd.read_csv("input/PseudoTime.csv", index_col=0)
df_ref = pd.read_csv("input/refNetwork.csv", usecols=["Gene1", "Gene2"])


_, df_smoothed = smooth(df_pse, df_exp, slipe=1, k=5)


lambda_vals = [0.1, 0.5, 1, 1.5, 2, 3, 5]
kf = KFold(n_splits=5, shuffle=True, random_state=0)
results = []


for lam in lambda_vals:
    aucs = []
    for train_idx, test_idx in kf.split(df_ref):
        ref_train = df_ref.iloc[train_idx]
        ref_test = df_ref.iloc[test_idx]

        df_score = cal_cramer2(df_smoothed, n_jobs=4)
        if set(df_score.columns) == set(df_smoothed.columns):
            df_score = df_score.stack().reset_index()
            df_score.columns = ['Gene1', 'Gene2', 'score']
        df_score['score'] = pd.to_numeric(df_score['score'], errors='coerce')

        df_mrmr = MRMR2_divergence(df_score, n_jobs=4, lambda_val=lam)
        df_eval = concat_ref(df_mrmr, ref_test)
        df_eval = df_eval[np.isfinite(df_eval["score"])]
        auc = cal_auc_aupr(df_eval)["AUROC"]
        aucs.append(auc)

    mean_auc = np.mean(aucs)
    results.append((lam, mean_auc))


df_results = pd.DataFrame(results, columns=["lambda", "mean_AUROC"])
df_results.to_csv(f"{output_dir}/cv_results_cramer.csv", index=False)


plt.figure()
plt.plot(df_results["lambda"], df_results["mean_AUROC"], marker="o")
plt.xlabel("Lambda")
plt.ylabel("Mean AUROC")
plt.title("5-fold CV: AUROC vs Lambda (Cramér)")
plt.grid(True)
plt.savefig(f"{output_dir}/cv_auc_vs_lambda_cramer.pdf")
plt.close()
