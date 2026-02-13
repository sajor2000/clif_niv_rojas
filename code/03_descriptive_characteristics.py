import pandas as pd
import numpy as np
import os
import json
from scipy import stats

# =====================================================
# Load Data & Config
# =====================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Load site config
config_path = os.path.join(ROOT_DIR, 'config.json')
if not os.path.exists(config_path):
    config_path = os.path.join(ROOT_DIR, 'clif_config.json')
with open(config_path) as f:
    config = json.load(f)
SITE = config.get('site', 'unknown')

df = pd.read_csv(os.path.join(ROOT_DIR, 'output', 'NIPPV_analytic_dataset.csv'))

# Ensure NIPPV failure is coded as Yes/No
df['failure'] = df['failure'].map({1: "Yes", 0: "No"})

# =====================================================
# Variable Definitions
# =====================================================

continuous_vars = {
    "age_at_admission": "Age at Admission, years",
    "map_after_NIPPV": "Mean Arterial Pressure, mmHg",
    "pco2_after_NIPPV": "pCO2, mmHg",
    "ph_after_NIPPV": "pH",
    "peep_set_after_NIPPV": "PEEP, cmH2O",
    "tidal_volume_obs_after_NIPPV": "Tidal Volume, mL",
    "heart_rate_after_NIPPV": "Heart Rate, bpm",
    "respiratory_rate_after_NIPPV": "Respiratory Rate, bpm",
    "fio2_after_NIPPV": "FiO2"
}

categorical_vars = {
    "sex_category": "Sex"
}

# =====================================================
# Helper Functions (SAME AS YOUR ORIGINAL)
# =====================================================

def summarize_continuous(subdf, var):
    if subdf[var].notna().sum() == 0:
        return "NA"
    return f"{subdf[var].mean():.1f} ± {subdf[var].std():.1f}"

def summarize_categorical(subdf, var):
    counts = subdf[var].value_counts(dropna=False)
    pct = subdf[var].value_counts(normalize=True, dropna=False) * 100
    return {lvl: f"{counts[lvl]} ({pct[lvl]:.1f}%)" for lvl in counts.index}

# =====================================================
# NEW: P-VALUE FUNCTIONS (Parametric Assumption)
# =====================================================

def format_pvalue(p):
    """Format p-value: show <0.001 for very small values."""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def pvalue_continuous_ttest(df, var):
    x = df[df['failure'] == "Yes"][var].dropna()
    y = df[df['failure'] == "No"][var].dropna()

    if len(x) < 2 or len(y) < 2:
        return "NA"

    p = stats.ttest_ind(x, y, equal_var=False)[1]
    return format_pvalue(p)

def pvalue_categorical_chisq(df, var):
    tab = pd.crosstab(df[var], df['failure'])

    if tab.shape[0] < 2:
        return "NA"

    # Use Fisher's exact test for 2x2 tables with any expected cell count < 5
    if tab.shape == (2, 2):
        expected = stats.chi2_contingency(tab)[3]
        if (expected < 5).any():
            p = stats.fisher_exact(tab)[1]
            return format_pvalue(p)

    p = stats.chi2_contingency(tab)[1]
    return format_pvalue(p)

# =====================================================
# BUILD TABLE 1 (ORIGINAL + P-VALUES)
# =====================================================

rows = []
rows.append(["Variable", "NIPPV Failure: Yes", "NIPPV Failure: No", "Total", "p-value"])

# ----- Continuous -----
for var, label in continuous_vars.items():
    rows.append([
        label,
        summarize_continuous(df[df['failure'] == "Yes"], var),
        summarize_continuous(df[df['failure'] == "No"], var),
        summarize_continuous(df, var),
        pvalue_continuous_ttest(df, var)
    ])

# ----- Categorical -----
for var, label in categorical_vars.items():
    rows.append([label, "", "", "", ""])
   
    levels = df[var].dropna().unique()
    for lvl in levels:
        yes_stats = summarize_categorical(df[df['failure'] == "Yes"], var).get(lvl, "0 (0%)")
        no_stats  = summarize_categorical(df[df['failure'] == "No"], var).get(lvl, "0 (0%)")
        total_stats = summarize_categorical(df, var).get(lvl, "0 (0%)")

        rows.append([
            f"    {lvl}",
            yes_stats,
            no_stats,
            total_stats,
            pvalue_categorical_chisq(df, var)
        ])

# =====================================================
# Add sample size row at top
# =====================================================

n_yes = len(df[df['failure'] == "Yes"])
n_no = len(df[df['failure'] == "No"])
n_total = len(df)
rows.insert(1, ["Patients, n (%)", f"{n_yes} ({100*n_yes/n_total:.1f}%)",
                 f"{n_no} ({100*n_no/n_total:.1f}%)", str(n_total), ""])

# =====================================================
# Display & Export
# =====================================================

table1 = pd.DataFrame(rows[1:], columns=rows[0])

print(f"\nTable 1. Descriptive Characteristics — Site: {SITE} (N={n_total})\n")
print(table1.to_string(index=False))

SHARE_DIR = os.path.join(ROOT_DIR, 'output_to_share')
os.makedirs(SHARE_DIR, exist_ok=True)

# Add site metadata
table1['site'] = SITE
table1['N'] = n_total
table1['N_events'] = n_yes

table1.to_csv(os.path.join(SHARE_DIR, 'descriptive_characteristics.csv'), index=False)
print(f"\nDescriptive characteristics exported for site: {SITE}")