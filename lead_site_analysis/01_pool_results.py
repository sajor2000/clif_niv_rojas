"""
01_pool_results.py — Fixed-Effects Meta-Analysis Pooling

Reads site-level CSVs from all_site_results/<site_name>/ and pools
regression results using inverse-variance fixed-effects meta-analysis.

Outputs pooled results to lead_site_analysis/output/
"""

import pandas as pd
import numpy as np
import os
import glob
from statsmodels.stats.meta_analysis import combine_effects
from scipy.stats import chi2

# =====================================================
# PATHS
# =====================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
SITES_DIR = os.path.join(ROOT_DIR, 'all_site_results')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# DISCOVER SITES
# =====================================================

site_dirs = sorted([
    d for d in glob.glob(os.path.join(SITES_DIR, '*'))
    if os.path.isdir(d) and not os.path.basename(d).startswith('.')
])

if not site_dirs:
    raise FileNotFoundError(
        f"No site directories found in {SITES_DIR}/\n"
        "Each site should have a subdirectory (e.g., all_site_results/rush/, "
        "all_site_results/bidmc/) containing their 14 CSV files."
    )

site_names = [os.path.basename(d) for d in site_dirs]
print(f"Found {len(site_dirs)} sites: {', '.join(site_names)}")


# =====================================================
# HELPER: LOAD CSV FROM ALL SITES
# =====================================================

def load_from_all_sites(filename):
    """Load a CSV from each site directory and concatenate."""
    frames = []
    for site_dir in site_dirs:
        path = os.path.join(site_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Ensure site column is populated
            if 'site' not in df.columns:
                df['site'] = os.path.basename(site_dir)
            frames.append(df)
        else:
            print(f"  WARNING: {filename} not found in {os.path.basename(site_dir)}/")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# =====================================================
# INVERSE-VARIANCE FIXED-EFFECTS META-ANALYSIS
# =====================================================

def pool_regression_results(df, model_label):
    """
    Pool regression results across sites using inverse-variance
    fixed-effects meta-analysis via statsmodels.combine_effects.

    Expects columns: Variable, log_OR, SE_log_OR, site, N, N_events
    Returns a DataFrame with pooled OR, CI, p-value, heterogeneity stats.
    """
    if df.empty:
        print(f"  WARNING: No data for {model_label}")
        return pd.DataFrame()

    # Exclude Intercept from pooling (not clinically meaningful as pooled OR)
    df_pool = df[df['Variable'] != 'Intercept'].copy()

    variables = df_pool['Variable'].unique()
    pooled_rows = []

    for var in variables:
        var_data = df_pool[df_pool['Variable'] == var].copy()

        if len(var_data) < 1:
            continue

        effects = var_data['log_OR'].values
        variances = (var_data['SE_log_OR'].values) ** 2

        # Skip if any variance is zero or negative (degenerate)
        if np.any(variances <= 0) or np.any(np.isnan(variances)):
            print(f"  WARNING: Invalid variance for {var}, skipping")
            continue

        site_labels = var_data['site'].tolist()

        if len(effects) == 1:
            # Single site — no pooling needed, pass through
            pooled_rows.append({
                'Variable': var,
                'n_sites': 1,
                'pooled_log_OR': effects[0],
                'pooled_SE': np.sqrt(variances[0]),
                'pooled_OR': np.exp(effects[0]),
                'pooled_CI_lower': np.exp(effects[0] - 1.96 * np.sqrt(variances[0])),
                'pooled_CI_upper': np.exp(effects[0] + 1.96 * np.sqrt(variances[0])),
                'pooled_p': var_data['P-Value'].values[0],
                'Q': np.nan,
                'Q_p': np.nan,
                'I2': np.nan,
                'tau2': np.nan,
                'total_N': int(var_data['N'].sum()),
                'total_events': int(var_data['N_events'].sum()),
                'model': model_label,
            })
            continue

        # Use statsmodels combine_effects for fixed-effects pooling
        res = combine_effects(effects, variances, method_re="dl",
                              use_t=False, row_names=site_labels)

        # Extract fixed-effects results
        fe = res.summary_frame().loc['fixed effect']
        re = res.summary_frame().loc['random effect']

        # Heterogeneity: Cochran's Q
        weights = 1.0 / variances
        weighted_mean = np.sum(weights * effects) / np.sum(weights)
        Q = np.sum(weights * (effects - weighted_mean) ** 2)
        Q_df = len(effects) - 1
        Q_p = chi2.sf(Q, Q_df) if Q_df > 0 else np.nan

        # I-squared
        I2 = max(0, (Q - Q_df) / Q) * 100 if Q > 0 else 0.0

        pooled_rows.append({
            'Variable': var,
            'n_sites': len(effects),
            'pooled_log_OR': fe['eff'],
            'pooled_SE': fe['sd_eff'],
            'pooled_OR': np.exp(fe['eff']),
            'pooled_CI_lower': np.exp(fe['ci_low']),
            'pooled_CI_upper': np.exp(fe['ci_upp']),
            'pooled_p': fe['pvalue'] if 'pvalue' in fe.index else np.nan,
            'Q': round(Q, 4),
            'Q_p': round(Q_p, 4) if not np.isnan(Q_p) else np.nan,
            'I2': round(I2, 1),
            'tau2': round(res.tau2, 6) if hasattr(res, 'tau2') else np.nan,
            'total_N': int(var_data['N'].sum()),
            'total_events': int(var_data['N_events'].sum()),
            'model': model_label,
        })

    return pd.DataFrame(pooled_rows)


# =====================================================
# 1. POOL UNIVARIATE RESULTS
# =====================================================

print("\n=== Pooling Univariate Results ===")
uni_all = load_from_all_sites('univariate_logistic_results.csv')
pooled_uni = pool_regression_results(uni_all, 'univariate')
if not pooled_uni.empty:
    pooled_uni.to_csv(os.path.join(OUTPUT_DIR, 'pooled_univariate.csv'), index=False)
    print(f"  Exported: pooled_univariate.csv ({len(pooled_uni)} variables)")

# =====================================================
# 2. POOL MULTIVARIABLE NO-INTERACTION RESULTS
# =====================================================

print("\n=== Pooling Multivariable (No Interaction) Results ===")
multi_no_all = load_from_all_sites('multivariable_logistic_results_NoInteraction.csv')
pooled_multi_no = pool_regression_results(multi_no_all, 'multivariable_no_interaction')
if not pooled_multi_no.empty:
    pooled_multi_no.to_csv(os.path.join(OUTPUT_DIR, 'pooled_multivariable_NoInteraction.csv'), index=False)
    print(f"  Exported: pooled_multivariable_NoInteraction.csv ({len(pooled_multi_no)} variables)")

# =====================================================
# 3. POOL MULTIVARIABLE INTERACTION RESULTS
# =====================================================

print("\n=== Pooling Multivariable (Interaction) Results ===")
multi_int_all = load_from_all_sites('multivariable_logistic_results_Interaction.csv')
pooled_multi_int = pool_regression_results(multi_int_all, 'multivariable_interaction')
if not pooled_multi_int.empty:
    pooled_multi_int.to_csv(os.path.join(OUTPUT_DIR, 'pooled_multivariable_Interaction.csv'), index=False)
    print(f"  Exported: pooled_multivariable_Interaction.csv ({len(pooled_multi_int)} variables)")

# =====================================================
# 4. POOL FIRTH SENSITIVITY RESULTS
# =====================================================

print("\n=== Pooling Firth Sensitivity Results ===")
firth_no_all = load_from_all_sites('firth_multivariable_results_NoInteraction.csv')
pooled_firth_no = pool_regression_results(firth_no_all, 'firth_no_interaction')
if not pooled_firth_no.empty:
    pooled_firth_no.to_csv(os.path.join(OUTPUT_DIR, 'pooled_firth_NoInteraction.csv'), index=False)
    print(f"  Exported: pooled_firth_NoInteraction.csv")

firth_int_all = load_from_all_sites('firth_multivariable_results_Interaction.csv')
pooled_firth_int = pool_regression_results(firth_int_all, 'firth_interaction')
if not pooled_firth_int.empty:
    pooled_firth_int.to_csv(os.path.join(OUTPUT_DIR, 'pooled_firth_Interaction.csv'), index=False)
    print(f"  Exported: pooled_firth_Interaction.csv")

# =====================================================
# 5. AGGREGATE CONSORT DIAGRAMS
# =====================================================

print("\n=== Aggregating CONSORT Data ===")
consort_all = load_from_all_sites('consort.csv')
if not consort_all.empty:
    consort_all.to_csv(os.path.join(OUTPUT_DIR, 'all_sites_consort.csv'), index=False)
    # Compute pooled totals
    numeric_cols = consort_all.select_dtypes(include=[np.number]).columns
    consort_pooled = consort_all[numeric_cols].sum().to_frame().T
    consort_pooled['site'] = 'pooled'
    consort_pooled = pd.concat([consort_all, consort_pooled], ignore_index=True)
    consort_pooled.to_csv(os.path.join(OUTPUT_DIR, 'consort_pooled.csv'), index=False)
    print(f"  Exported: consort_pooled.csv")

consort_flow_all = load_from_all_sites('consort_flow.csv')
if not consort_flow_all.empty:
    consort_flow_all.to_csv(os.path.join(OUTPUT_DIR, 'all_sites_consort_flow.csv'), index=False)
    print(f"  Exported: all_sites_consort_flow.csv")

# =====================================================
# 6. AGGREGATE DESCRIPTIVE CHARACTERISTICS
# =====================================================

print("\n=== Aggregating Descriptive Characteristics ===")
desc_all = load_from_all_sites('descriptive_characteristics.csv')
if not desc_all.empty:
    desc_all.to_csv(os.path.join(OUTPUT_DIR, 'all_sites_descriptive.csv'), index=False)
    print(f"  Exported: all_sites_descriptive.csv")

# =====================================================
# 7. AGGREGATE MODEL DIAGNOSTICS
# =====================================================

print("\n=== Aggregating Model Diagnostics ===")
diag_no = load_from_all_sites('model_diagnostics_NoInteraction.csv')
diag_int = load_from_all_sites('model_diagnostics_Interaction.csv')
diag_all = pd.concat([diag_no, diag_int], ignore_index=True)
if not diag_all.empty:
    diag_all.to_csv(os.path.join(OUTPUT_DIR, 'all_sites_diagnostics.csv'), index=False)
    print(f"  Exported: all_sites_diagnostics.csv")

# =====================================================
# 8. AGGREGATE MISSINGNESS
# =====================================================

print("\n=== Aggregating Missingness Data ===")
miss_all = load_from_all_sites('missingness_table.csv')
if not miss_all.empty:
    miss_all.to_csv(os.path.join(OUTPUT_DIR, 'all_sites_missingness.csv'), index=False)
    print(f"  Exported: all_sites_missingness.csv")

# =====================================================
# 9. AGGREGATE VIF
# =====================================================

print("\n=== Aggregating VIF Data ===")
vif_all = load_from_all_sites('vif_NoInteraction.csv')
if not vif_all.empty:
    vif_all.to_csv(os.path.join(OUTPUT_DIR, 'all_sites_vif.csv'), index=False)
    print(f"  Exported: all_sites_vif.csv")

# =====================================================
# 10. STORE SITE-LEVEL REGRESSION FOR FIGURES
# =====================================================

# Save concatenated site-level data for use by figures script
for name, df in [('univariate', uni_all),
                 ('multi_no_interaction', multi_no_all),
                 ('multi_interaction', multi_int_all),
                 ('firth_no_interaction', firth_no_all),
                 ('firth_interaction', firth_int_all)]:
    if not df.empty:
        df.to_csv(os.path.join(OUTPUT_DIR, f'site_level_{name}.csv'), index=False)

# =====================================================
# SUMMARY
# =====================================================

print("\n" + "=" * 60)
print("POOLING COMPLETE")
print("=" * 60)
total_n = consort_all['patients_post_missing'].sum() if not consort_all.empty and 'patients_post_missing' in consort_all.columns else 'N/A'
total_events = consort_all['failures_post_missing'].sum() if not consort_all.empty and 'failures_post_missing' in consort_all.columns else 'N/A'
print(f"Sites:        {len(site_dirs)}")
print(f"Total N:      {total_n}")
print(f"Total events: {total_events}")
print(f"Output:       {OUTPUT_DIR}/")
