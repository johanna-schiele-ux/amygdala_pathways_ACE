"""
Group difference analysis of FD and log FC 
- use demeaned age and t1_volume, to make intercept interpretable (= adjusted mean of group 1, ptsd)

- Assumptions checks for ANCOVA
   - homogeneity of slopes
   - homogeneity of variances
   - homoscedasticity of residuals
   - normality of residuals
   - linearity

- perform ancova
   - fd: age as covariate
   - fc: age and brain volume as covariate
   - option to include total CTQ score as covariate in both models
"""

from pathlib import Path
import pandas as pd
from scipy.stats import shapiro, levene
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# paths
# ==============
# folder
csv_folder = Path("path/to/folder")

# files in above folder containing fd/fc values of all tracts and covariates
fd_file = Path(csv_folder / "FD_group_covariates_demeaned.csv") # columns participant_id	group	CB_fd_mean	UF_fd_mean	ST_fd_mean	mri_age_demeaned	CTQ_total_score_demeaned
fc_file = Path(csv_folder / "FC_group_covariates_demeaned.csv") # columns participant_id	group	CB_fc_mean	UF_fc_mean	ST_fc_mean	mri_age_demeaned	t1_volume_demeaned	CTQ_total_score_demeaned

# tracts
TRACTS_FD = [("UF", "UF_fd_mean"), ("CB", "CB_fd_mean"), ("ST", "ST_fd_mean"),]
TRACTS_FC = [("UF", "UF_fc_mean"), ("CB", "CB_fc_mean"), ("ST", "ST_fc_mean"),]

# --------------------------------------------------------------------------------------------------------------------------------------------------------------

# ====================================================
# choose below whether to include CTQ as covariate
# ====================================================

include_CTQ = True

if include_CTQ:
    FD_COVARIATES = ["mri_age_demeaned", "CTQ_total_score_demeaned"]
    FC_COVARIATES = ["mri_age_demeaned", "t1_volume_demeaned", "CTQ_total_score_demeaned"]
    print("=" * 120 + "\n" + "=" * 120)
    print("Group difference of FD and log FC -- With total CTQ as additional covariate")
    print("=" * 120 + "\n" + "=" * 120)
else:
    FD_COVARIATES = ["mri_age_demeaned"]
    FC_COVARIATES = ["mri_age_demeaned", "t1_volume_demeaned"]
    print("=" * 120 + "\n" + "=" * 120)
    print("Group difference of FD and log FC -- Without total CTQ as covariate")
    print("=" * 120 + "\n" + "=" * 120)

# ==================================================================================
# HELPER FUNCTIONS for asumption checks and ANCOVA
# ==================================================================================


def _formula(outcome_col, covariates):
    """Build formula string from outcome and list of covariates."""
    return f"{outcome_col} ~ C(group) + {' + '.join(covariates)}"

  
def check_homogeneity_of_slopes(df, outcome_col, covariates):
    """1) Homogeneity of regression slopes.
    Tests C(group)*covariate interaction for each covariate separately.
    Formula expands to: outcome ~ C(group) * (cov1 + cov2 + ...)
    which is equivalent to: ... + C(group)*cov1 + C(group)*cov2
    """
    covar_str = " + ".join(covariates)
    # use parenthesised sum so statsmodels expands all interactions at once
    model = smf.ols(f"{outcome_col} ~ C(group) * ({covar_str})", data=df).fit()
 
    # check each interaction term individually and report separately
    for cov in covariates:
        term = f"C(group)[T.control]:{cov}"
        p = model.pvalues.get(term, None)
        if p is None:
            print(f"  ⚠️  Interaction term '{term}' not found — check group coding.")
        elif p < 0.05:
            print(f"  ❌ WARNING: group*{cov} interaction significant (p={p:.4f}). Slopes differ!")
        else:
            print(f"  ✅ group*{cov} interaction not significant (p={p:.4f}). Homogeneity of slopes assumed.")
 

def check_levene(df, outcome_col):
    """2) Homogeneity of variances (Levene test). Covariates not needed here."""
    groups = [grp[outcome_col].dropna() for _, grp in df.groupby("group")]
    stat, p = levene(*groups)
    if p < 0.05:
        print(f"  ❌ WARNING: Levene W={stat:.3f}, p={p:.3f} — variances differ between groups!")
    else:
        print(f"  ✅ Levene W={stat:.3f}, p={p:.3f} — homogeneity of variances assumed.")


def check_homoscedasticity(df, outcome_col, covariates):
    """3) Homoscedasticity of residuals (Breusch-Pagan test)."""
    model = smf.ols(_formula(outcome_col, covariates), data=df).fit()
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    bp_p = bp_test[3]
    if bp_p < 0.05:
        print(f"  ❌ WARNING: Heteroscedasticity detected (Breusch-Pagan p={bp_p:.4f})")
    else:
        print(f"  ✅ Breusch-Pagan p={bp_p:.4f} — homoscedasticity assumed.")

 
def check_normality(df, outcome_col, covariates, label=""):
    """4) Normality of residuals (Shapiro-Wilk + QQ-plot)."""
    model = smf.ols(_formula(outcome_col, covariates), data=df).fit()
    stat, p = shapiro(model.resid)
    if p < 0.05:
        print(f"  ❌ WARNING: Shapiro-Wilk W={stat:.3f}, p={p:.3f} — residuals not normally distributed!")
    else:
        print(f"  ✅ Shapiro-Wilk W={stat:.3f}, p={p:.3f} — normality of residuals assumed.")
 
    sm.qqplot(model.resid, line='s')
    plt.title(f"QQ-plot residuals — {label}")
    plt.tight_layout()
    # plt.show()


def check_linearity(df, outcome_col, covariates, label=""):
    """5) Linearity: one scatter+regression plot per covariate.
    loops over covariates to create one plot per covariate.
    """
    for cov in covariates:
        plt.figure(figsize=(7, 5))
        for g in df["group"].unique():
            subset = df[df["group"] == g]
            sns.scatterplot(x=cov, y=outcome_col, data=subset, alpha=0.5, label=f"group {g}")
            sns.regplot(x=cov, y=outcome_col, data=subset, scatter=False)
        sns.regplot(x=cov, y=outcome_col, data=df, scatter=False, color="black", label="Pooled")
        plt.legend()
        plt.title(f"{label} ~ {cov} — group-specific + pooled regression")
        plt.tight_layout()
        #plt.show()


def run_ancova(df, outcome_col, covariates, label=""):

    covar_str = " + ".join(covariates)

    # HC3-robust type II ANCOVA table: SS, F, p, np2 for group + all covariates
    # uses OLS as basis, effect size (np2) needs to be computed manually
    model = smf.ols(f"{outcome_col} ~ C(group) + {covar_str}", data=df).fit()
    
    aov_robust = sm.stats.anova_lm(model, typ=2, robust='hc3')
    # add np2
    ss_resid_robust = aov_robust.loc['Residual', 'sum_sq']
    aov_robust['np2'] = aov_robust['sum_sq'] / (aov_robust['sum_sq'] + ss_resid_robust)

    print(f"\n\nANCOVA results (HC3-robust, Type II) — {label}")
    print("-" * 62)
    print(aov_robust)

    p_val = aov_robust.loc['C(group)', 'PR(>F)']
    if p_val < 0.05:
        print(f"  ✅ group effect significant (p={p_val:.4f})")
    else:
        print(f"  ❌ group effect not significant (p={p_val:.4f})")

    cov_pvals = {cov: aov_robust.loc[cov, 'PR(>F)'] for cov in covariates}

    # OLS refit with HC3 SEs for coefficient table and adjusted means to supplement ANCOVA outputs
    model = smf.ols(f"{outcome_col} ~ C(group) + {covar_str}", data=df).fit(cov_type="HC3", use_t=True)

    print("\n\nOLS coefficient table (HC3 SEs):")
    print(model.summary().tables[1])

    # adjusted means: all covariates at 0 (= their demeaned means)
    grid = pd.DataFrame({
        "group": pd.Categorical(
            df["group"].unique(),
            categories=df["group"].astype("category").cat.categories
        ),
        **{cov: 0 for cov in covariates}
    })
    pred = model.get_prediction(grid).summary_frame()
    grid["adjusted_mean"] = model.predict(grid)
    print("\nAdjusted group means (all covariates at their mean):")
    print("-" * 62)
    print(pd.concat([grid[["group", "adjusted_mean"]].reset_index(drop=True),
                     pred.reset_index(drop=True)], axis=1))

    return p_val, cov_pvals



# ==================================================================================
# MAIN: run FD and FC analysis
# ==================================================================================
 
def run_metric(df_raw, tracts, covariates, metric_label):
    p_values = []
    covariate_results = []
 
    for tract, col in tracts:
        label = f"{tract}_{metric_label}"
        print(f"\n\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
 
        df = df_raw[["group"] + covariates + [col]].dropna().copy()
        df = df.rename(columns={col: "outcome"})

        # explicitly set PTSD as reference group via the category order
        df["group"] = pd.Categorical(df["group"], categories=["PTSD", "control"])
 
        stats = df.groupby("group")["outcome"].agg(["mean", "std", "count"])
        print(f"\nDescriptive statistics (raw):")
        print("-" * 62)
        print(stats)

        print("\n Assumption checks")
        print("-" * 62)
        check_homogeneity_of_slopes(df, "outcome", covariates)
        check_levene(df, "outcome")
        check_homoscedasticity(df, "outcome", covariates)
        check_normality(df, "outcome", covariates, label=label)
        check_linearity(df, "outcome", covariates, label=label)
 
        p_val, cov_pvals = run_ancova(df, "outcome", covariates, label=label)
        p_values.append({"tract": label, "p_unc": p_val})
        covariate_results.append({"tract": label, **cov_pvals})
 
    return pd.DataFrame(p_values), pd.DataFrame(covariate_results)
 

# --- FD ---
print("\nLoading FD data...")
df_fd = pd.read_csv(fd_file)
results_fd, cov_fd = run_metric(df_fd, TRACTS_FD, FD_COVARIATES, "fd")
 
# --- FC ---
print("\n\n" + "=" * 120 + "\n" + "=" * 120)
print("\n\nLoading FC data...")
df_fc = pd.read_csv(fc_file)
results_fc, cov_fc = run_metric(df_fc, TRACTS_FC, FC_COVARIATES, "fc")
 
# --- combined results ---
results_all = pd.concat([results_fd, results_fc], ignore_index=True)
print("\n\nAll uncorrected p-values of the tracts:")
print(results_all)
print("\nUncorrected p-values for FD Covariates")
print(cov_fd)
print("\nUncorrected p-values for FC Covariates")
print(cov_fc)
