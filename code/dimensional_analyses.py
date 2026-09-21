"""
Multiple Linear Regression: FD & FC tract metrics →  clinical variable
- Predictors: 3 tracts (UF, CB, ST) + AGE as covariate (all variables and targets z-scored)
- FC predictors are residualized on brain volume beforehand
- use Huber regression to handle violation of normality of residuals
- options to choose group and clinical variable to be analyzed (PTSD vs No PTSD and PID5 vs CTQ vs PCL5(only for PTSD))

- Assumption checks: 
    - homoscedasticity of residuals (not as critical for Huber),
    - normality of residuals (not required for Huber, but check nevertheless to spot extreme violations),
    - multicollinearity,
    - linearity plot
"""


from pathlib import Path
import pandas as pd
from scipy.stats import shapiro
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.robust.robust_linear_model import RLM

# choose below which group and clinical variable to analyze
# ----------------------------------------------------------
# choose "PTSD" or "NoPTSD"
group = "PTSD"

# choose "PCL5", "PID5", or "CTQ"
# PCL5 is only available for the PTSD group
target = "CTQ"
# ----------------------------------------------------------

# tracts
TRACTS_FC = ["UF_fc_mean_res_z", "CB_fc_mean_res_z", "ST_fc_mean_res_z"]
TRACTS_FD = ["UF_fd_mean_z",     "CB_fd_mean_z",     "ST_fd_mean_z"    ]

# covariate
COVARIATE = "mri_age_z"   # same for both metrics

pid5_outcomes = {
    "Negative Affectivity":  "Negative_Affectivity_z",
    "Detachment":  "Detachment_z",
    "Antagonism":  "Antagonism_z",
    "Disinhibition": "Disinhibition_z",
    "Psychoticism": "Psychoticism_z"
}

pcl5_outcomes = {
    "Total PCL":          "PCL_SUM_z",
    "Cluster Intrusion":  "PCL_INTRU_z",
    "Cluster Avoidance":  "PCL_AVOID_z",
    "Cluster Cogn/Mood":  "PCL_COMO_z",
    "Cluster Hyperarous": "PCL_HYPE_z",
}

ctq_outcomes = {
    "Total CTQ":         "CTQ_total_score_z",
    "Emotional Abuse":   "CTQemotional_abuse_z",
    "Physical Abuse":    "CTQphysical_abuse_z",
    "Sexual Abuse":      "CTQsexual_abuse_z",
    "Emotional Neglect": "CTQemotional_neglect_z",
    "Physical Neglect":  "CTQphys_neglect_z",
}


# select respective paths based on group and target choice above and set coutcome columns accordingly
if group == "PTSD" and target == "PID5":
    fc_file = Path("path/to/PID5/file/fc") # columns participant_id	group	SEX	mri_age_z	CB_fc_mean_res_z	ST_fc_mean_res_z	UF_fc_mean_res_z	CB_fc_L_res_z	ST_fc_L_res_z	UF_fc_L_res_z	CB_fc_R_res_z	ST_fc_R_res_z	UF_fc_R_res_z	PCL_INTRU_z	PCL_AVOID_z	PCL_COMO_z	PCL_HYPE_z	PCL_SUM_z	Negative_Affectivity_z	Detachment_z	Antagonism_z	Disinhibition_z	Psychoticism_z	CTQemotional_abuse_z	CTQphysical_abuse_z	CTQsexual_abuse_z	CTQemotional_neglect_z	CTQphys_neglect_z	CTQminimization_z	CTQinconsistency_z	CTQ_total score_z
    fd_file = Path("path/to/PID5/file/fd") # columns participant_id	group	SEX	mri_age_z	CB_fd_mean_z	ST_fd_mean_z	UF_fd_mean_z	CB_fd_L_z	ST_fd_L_z	UF_fd_L_z	CB_fd_R_z	ST_fd_R_z	UF_fd_R_z	PCL_INTRU_z	PCL_AVOID_z	PCL_COMO_z	PCL_HYPE_z	PCL_SUM_z	Negative_Affectivity_z	Detachment_z	Antagonism_z	Disinhibition_z	Psychoticism_z	CTQemotional_abuse_z	CTQphysical_abuse_z	CTQsexual_abuse_z	CTQemotional_neglect_z	CTQphys_neglect_z	CTQminimization_z	CTQinconsistency_z	CTQ_total score_z

    OUTCOMES = pid5_outcomes
    print("=" * 120 + "\n" + "=" * 120 + "\nPredicting PID5 scores from tract metrics in the group with PTSD\n" + "=" * 120 + "\n" + "=" * 120 + "\n")
    
elif group == "PTSD" and target in ["PCL5", "CTQ"]:
    fc_file = Path("path/to/PCL5_CTQ/file/fc") 
    fd_file = Path("path/to/PCL5_CTQ/file/fd")

    if target == "PCL5":
        OUTCOMES = pcl5_outcomes
        print("=" * 120 + "\n" + "=" * 120 + "\nPredicting PCL5 scores from tract metrics in the group with PTSD\n" + "=" * 120 + "\n" + "=" * 120 + "\n")
    else:
        OUTCOMES = ctq_outcomes
        print("=" * 120 + "\n" + "=" * 120 + "\nPredicting CTQ scores from tract metrics in the  group with PTSD\n" + "=" * 120 + "\n" + "=" * 120 + "\n")

elif group == "NoPTSD" and target in ["PID5", "CTQ"]:
    fc_file = Path("path/to/PID5_CTQ/file_fc")
    fd_file = Path("path/to/PID5_CTQ/file_fd")

    if target == "PID5":
        OUTCOMES = pid5_outcomes
        print("=" * 120 + "\n" + "=" * 120 + "\nPredicting PID5 scores from tract metrics in the group without PTSD\n" + "=" * 120 + "\n" + "=" * 120 + "\n")
    else:
        OUTCOMES = ctq_outcomes
        print("=" * 120 + "\n" + "=" * 120 + "\nPredicting CTQ scores from tract metrics in the group without PTSD\n" + "=" * 120 + "\n" + "=" * 120 + "\n")

else:
    raise SystemExit("Group or target choice invalid. Please check your input.")
    



# ============================================================
# HELPER FUNCTIONS for assumption checks and MLR
# ============================================================
 
def _formula(outcome_col, predictors, covariate):
    """Build regression formula string."""
    all_terms = predictors + [covariate]
    return f"{outcome_col} ~ {' + '.join(all_terms)}"
 
 
def check_homoscedasticity(df, outcome_col, predictors, covariate):
    """1) Homoscedasticity of residuals (Breusch-Pagan test)."""
    model = RLM.from_formula(_formula(outcome_col, predictors, covariate), data=df, M=sm.robust.norms.HuberT()).fit()
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    bp_p = bp_test[3]
    if bp_p < 0.05:
        print(f"  ❌ WARNING: Heteroscedasticity detected (Breusch-Pagan p={bp_p:.4f}) — homoscedasticity not assumed. Not as critical for Huber regression")
    else:
        print(f"  ✅ Breusch-Pagan p={bp_p:.4f} — homoscedasticity assumed.")
 

def check_normality(df, outcome_col, predictors, covariate, label=""):
    """2) Normality of residuals (Shapiro-Wilk + QQ-plot)."""
    model = RLM.from_formula(_formula(outcome_col, predictors, covariate), data=df, M=sm.robust.norms.HuberT()).fit()
    stat, p = shapiro(model.resid)
    if p < 0.05:
        print(f"  ❌ WARNING: Shapiro-Wilk W={stat:.3f}, p={p:.3f} — residuals not normally distributed! (Not a formal requirement for Huber regression)")
    else:
        print(f"  ✅ Shapiro-Wilk W={stat:.3f}, p={p:.3f} — normality of residuals assumed.")


    sm.qqplot(model.resid, line='s')
    plt.title(f"QQ-plot residuals — {label}")
    plt.tight_layout()
    #plt.show()


def check_vif(df, predictors, covariate):
    """3) No multicollinearity (VIF)."""
    all_terms = predictors + [covariate]
    X = df[all_terms].copy()
    X["Intercept"] = 1
 
    vif_df = pd.DataFrame()
    vif_df["Variable"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
 
    max_vif = vif_df[vif_df["Variable"] != "Intercept"]["VIF"].max()
    if max_vif > 5:
        print(f"  ❌ WARNING: Potential multicollinearity detected (VIF > 5).")
    else:
        print(f"  ✅ No problematic multicollinearity detected.")
    print(vif_df.to_string(index=False))


def check_linearity(df, outcome_col, predictors, covariate, label=""):
    """4) Linearity: fitted values vs residuals — should be a random cloud."""
    model = RLM.from_formula(_formula(outcome_col, predictors, covariate), data=df, M=sm.robust.norms.HuberT()).fit()
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=model.fittedvalues, y=model.resid)
    plt.axhline(0, color="red", linewidth=0.8, linestyle="--")
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title(f"Fitted vs Residuals — {label}")
    plt.tight_layout()
    #plt.show()
    print(f"  ➡️  Inspect plot: residuals should form a random cloud around zero.")
 


def run_mlr(df, outcome_col, predictors, covariate, label=""):
    """Fit Huber regression, print summary."""
    formula = _formula(outcome_col, predictors, covariate)
    
    model = RLM.from_formula(formula, data=df, M=sm.robust.norms.HuberT()).fit()

    print(f"\nMLR results: {label}")
    print("-" * 52)
    print(f"formula: {formula}\n")
    print(model.summary())
 
    return model


# ============================================================
# MAIN: run assumption checks + MLR for one metric
# ============================================================
 
def run_metric(df_raw, tracts, covariate, metric_label):
 
    for outcome_label, outcome_col in OUTCOMES.items():
        label = f"{metric_label} → {outcome_label}"
 
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}\n")
 
        df = df_raw[["participant_id"] + tracts + [covariate, outcome_col]].dropna().copy()

        print("Assumption checks")
        print("-" * 52)
        print(f"\n[1] Homoscedasticity")
        check_homoscedasticity(df, outcome_col, tracts, covariate)
 
        print(f"\n[2] Normality of residuals")
        check_normality(df, outcome_col, tracts, covariate, label=label)
 
        print(f"\n[3] Multicollinearity (VIF)")
        check_vif(df, tracts, covariate)
 
        print(f"\n[4] Linearity")
        check_linearity(df, outcome_col, tracts, covariate, label=label)
 
        run_mlr(df, outcome_col, tracts, covariate, label=label)
 


# ============================================================
# --- FD ---
# ============================================================
print("Loading FD data...")
df_fd = pd.read_csv(fd_file)
run_metric(df_fd, TRACTS_FD, COVARIATE, "FD")
 
# ============================================================
# --- FC ---
# ============================================================
print("\n\n" + "=" * 120 + "\n" + "=" * 120)
print("\n\nLoading FC data...")
df_fc = pd.read_csv(fc_file)
run_metric(df_fc, TRACTS_FC, COVARIATE, "FC")
 
print("\n\nAll models complete.")

