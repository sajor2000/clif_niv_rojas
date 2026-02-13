"""
03_tables.py — Publication-Ready Tables for Meta-Analysis

Reads pooled results from lead_site_analysis/output/ and generates:
  - Table 1: Pooled descriptive characteristics
  - Table 2: Univariate + multivariable results (publication format)
  - Table 3: Sensitivity analysis (MLE vs Firth)
  - Table 4: Model diagnostics summary
  - Table 5: Heterogeneity summary

All tables saved to lead_site_analysis/tables/ as CSV.
"""

import pandas as pd
import numpy as np
import os

# =====================================================
# PATHS
# =====================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
TABLE_DIR = os.path.join(SCRIPT_DIR, 'tables')
os.makedirs(TABLE_DIR, exist_ok=True)

# =====================================================
# PREDICTOR DISPLAY LABELS
# =====================================================

PREDICTOR_LABELS = {
    'age_scale': 'Age (per 10 yr)',
    'female': 'Female sex',
    'pco2_scale': 'pCO2 (per 10 mmHg)',
    'ph_scale': 'pH (per 0.1 unit)',
    'map_scale': 'MAP (per 10 mmHg)',
    'rr_scale': 'RR (per 5 bpm)',
    'hr_scale': 'HR (per 10 bpm)',
    'fio2_high': 'FiO2 > 0.4',
    'peep_scale': 'PEEP (per 2 cmH2O)',
    'tidal_volume_scale': 'Tidal Volume (per 100 mL)',
    'age_scale:ph_scale': 'Age x pH',
    'pco2_scale:rr_scale': 'pCO2 x RR',
}

PREDICTOR_ORDER = [
    'age_scale', 'female', 'pco2_scale', 'ph_scale', 'map_scale',
    'rr_scale', 'hr_scale', 'fio2_high', 'peep_scale', 'tidal_volume_scale',
    'age_scale:ph_scale', 'pco2_scale:rr_scale',
]


def get_label(var):
    return PREDICTOR_LABELS.get(var, var)


def sort_by_order(df, col='Variable'):
    order_map = {v: i for i, v in enumerate(PREDICTOR_ORDER)}
    df = df.copy()
    df['_sort'] = df[col].map(order_map).fillna(99)
    df = df.sort_values('_sort').drop(columns='_sort')
    return df


def fmt_or(row, or_col='pooled_OR', lo_col='pooled_CI_lower', hi_col='pooled_CI_upper'):
    """Format OR (95% CI) as string."""
    return f"{row[or_col]:.2f} ({row[lo_col]:.2f}-{row[hi_col]:.2f})"


def fmt_p(p):
    """Format p-value."""
    if pd.isna(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def load(name):
    path = os.path.join(OUTPUT_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


# =====================================================
# TABLE 1: POOLED DESCRIPTIVE CHARACTERISTICS
# =====================================================

def table1_pooled():
    """
    Combine site-level descriptive characteristics.
    Since we only have summary stats (mean +/- SD), we list each site's
    Table 1 side by side with the site label.
    """
    desc = load('all_sites_descriptive.csv')
    if desc.empty:
        print("  Skipping Table 1: no descriptive data")
        return

    # Pivot to have one column per site for each group
    sites = desc['site'].unique()

    if len(sites) == 1:
        # Single site — just reformat
        out = desc[['Variable', 'NIPPV Failure: Yes', 'NIPPV Failure: No',
                     'Total', 'p-value']].copy()
    else:
        # Multi-site: reshape
        frames = []
        for site in sites:
            s = desc[desc['site'] == site][['Variable', 'NIPPV Failure: Yes',
                                             'NIPPV Failure: No', 'Total', 'p-value']].copy()
            s.columns = ['Variable'] + [f'{c} ({site})' for c in s.columns[1:]]
            frames.append(s)

        out = frames[0]
        for f in frames[1:]:
            out = out.merge(f, on='Variable', how='outer')

    out.to_csv(os.path.join(TABLE_DIR, 'table1_descriptive.csv'), index=False)
    print(f"  Saved: table1_descriptive.csv ({len(out)} rows)")


# =====================================================
# TABLE 2: REGRESSION RESULTS (PUBLICATION FORMAT)
# =====================================================

def table2_regression():
    """
    Combined table: univariate OR | multivariable OR (no interaction)
    For each predictor: Crude OR (95% CI) | p | Adjusted OR (95% CI) | p
    """
    uni = load('pooled_univariate.csv')
    multi = load('pooled_multivariable_NoInteraction.csv')

    if uni.empty and multi.empty:
        print("  Skipping Table 2: no regression data")
        return

    rows = []
    all_vars = PREDICTOR_ORDER[:10]  # Main predictors only (no interactions)

    for var in all_vars:
        row = {'Predictor': get_label(var)}

        # Univariate
        uni_row = uni[uni['Variable'] == var]
        if not uni_row.empty:
            r = uni_row.iloc[0]
            row['Crude OR (95% CI)'] = fmt_or(r)
            row['Crude p'] = fmt_p(r['pooled_p'])
        else:
            row['Crude OR (95% CI)'] = '—'
            row['Crude p'] = '—'

        # Multivariable
        multi_row = multi[multi['Variable'] == var]
        if not multi_row.empty:
            r = multi_row.iloc[0]
            row['Adjusted OR (95% CI)'] = fmt_or(r)
            row['Adjusted p'] = fmt_p(r['pooled_p'])
        else:
            row['Adjusted OR (95% CI)'] = '—'
            row['Adjusted p'] = '—'

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLE_DIR, 'table2_regression_results.csv'), index=False)
    print(f"  Saved: table2_regression_results.csv ({len(out)} predictors)")


# =====================================================
# TABLE 2b: INTERACTION MODEL RESULTS
# =====================================================

def table2b_interaction():
    """Multivariable model with interaction terms."""
    multi_int = load('pooled_multivariable_Interaction.csv')
    if multi_int.empty:
        print("  Skipping Table 2b: no interaction data")
        return

    multi_int = sort_by_order(multi_int)
    rows = []
    for _, r in multi_int.iterrows():
        rows.append({
            'Predictor': get_label(r['Variable']),
            'Adjusted OR (95% CI)': fmt_or(r),
            'p-value': fmt_p(r['pooled_p']),
            'I²': f"{r['I2']:.0f}%" if pd.notna(r.get('I2')) else '—',
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLE_DIR, 'table2b_interaction_results.csv'), index=False)
    print(f"  Saved: table2b_interaction_results.csv ({len(out)} predictors)")


# =====================================================
# TABLE 3: SENSITIVITY ANALYSIS (MLE vs FIRTH)
# =====================================================

def table3_sensitivity():
    """Compare MLE and Firth pooled ORs side by side."""
    mle = load('pooled_multivariable_NoInteraction.csv')
    firth = load('pooled_firth_NoInteraction.csv')

    if mle.empty and firth.empty:
        print("  Skipping Table 3: no sensitivity data")
        return

    rows = []
    all_vars = PREDICTOR_ORDER[:10]

    for var in all_vars:
        row = {'Predictor': get_label(var)}

        mle_row = mle[mle['Variable'] == var]
        if not mle_row.empty:
            r = mle_row.iloc[0]
            row['MLE OR (95% CI)'] = fmt_or(r)
            row['MLE p'] = fmt_p(r['pooled_p'])
        else:
            row['MLE OR (95% CI)'] = '—'
            row['MLE p'] = '—'

        firth_row = firth[firth['Variable'] == var]
        if not firth_row.empty:
            r = firth_row.iloc[0]
            row['Firth OR (95% CI)'] = fmt_or(r)
            row['Firth p'] = fmt_p(r['pooled_p'])
        else:
            row['Firth OR (95% CI)'] = '—'
            row['Firth p'] = '—'

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLE_DIR, 'table3_sensitivity.csv'), index=False)
    print(f"  Saved: table3_sensitivity.csv ({len(out)} predictors)")


# =====================================================
# TABLE 4: MODEL DIAGNOSTICS SUMMARY
# =====================================================

def table4_diagnostics():
    """Summary of model diagnostics across sites."""
    diag = load('all_sites_diagnostics.csv')
    if diag.empty:
        print("  Skipping Table 4: no diagnostics data")
        return

    cols = ['site', 'model', 'N', 'N_events', 'AUC', 'AUC_SE',
            'cal_intercept', 'cal_slope', 'brier_score', 'EPV']
    available = [c for c in cols if c in diag.columns]
    out = diag[available].copy()

    # Round numeric columns
    for c in ['AUC', 'AUC_SE', 'cal_intercept', 'cal_slope', 'brier_score', 'EPV']:
        if c in out.columns:
            out[c] = out[c].round(4)

    out.to_csv(os.path.join(TABLE_DIR, 'table4_diagnostics.csv'), index=False)
    print(f"  Saved: table4_diagnostics.csv ({len(out)} rows)")


# =====================================================
# TABLE 5: HETEROGENEITY SUMMARY
# =====================================================

def table5_heterogeneity():
    """Heterogeneity statistics for each pooled predictor."""
    multi = load('pooled_multivariable_NoInteraction.csv')
    if multi.empty:
        print("  Skipping Table 5: no data")
        return

    multi = sort_by_order(multi)
    rows = []
    for _, r in multi.iterrows():
        rows.append({
            'Predictor': get_label(r['Variable']),
            'Pooled OR': f"{r['pooled_OR']:.2f}",
            'n sites': int(r['n_sites']),
            'Q statistic': f"{r['Q']:.2f}" if pd.notna(r.get('Q')) else '—',
            'Q p-value': fmt_p(r.get('Q_p')),
            'I² (%)': f"{r['I2']:.1f}" if pd.notna(r.get('I2')) else '—',
            'tau²': f"{r['tau2']:.4f}" if pd.notna(r.get('tau2')) else '—',
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(TABLE_DIR, 'table5_heterogeneity.csv'), index=False)
    print(f"  Saved: table5_heterogeneity.csv ({len(out)} predictors)")


# =====================================================
# TABLE 6: CONSORT SUMMARY
# =====================================================

def table6_consort():
    """CONSORT diagram data — per site and pooled."""
    consort = load('consort_pooled.csv')
    if consort.empty:
        print("  Skipping Table 6: no consort data")
        return

    consort.to_csv(os.path.join(TABLE_DIR, 'table6_consort.csv'), index=False)
    print(f"  Saved: table6_consort.csv ({len(consort)} rows)")


# =====================================================
# TABLE 7: MISSINGNESS
# =====================================================

def table7_missingness():
    """Missingness patterns across all sites."""
    miss = load('all_sites_missingness.csv')
    if miss.empty:
        print("  Skipping Table 7: no missingness data")
        return

    miss.to_csv(os.path.join(TABLE_DIR, 'table7_missingness.csv'), index=False)
    print(f"  Saved: table7_missingness.csv ({len(miss)} rows)")


# =====================================================
# MAIN
# =====================================================

if __name__ == '__main__':
    print("=" * 60)
    print("GENERATING TABLES")
    print("=" * 60)

    table1_pooled()
    table2_regression()
    table2b_interaction()
    table3_sensitivity()
    table4_diagnostics()
    table5_heterogeneity()
    table6_consort()
    table7_missingness()

    print(f"\nAll tables saved to: {TABLE_DIR}/")
