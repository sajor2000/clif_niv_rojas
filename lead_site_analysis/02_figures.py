"""
02_figures.py — Publication-Quality Figures for Meta-Analysis

Reads pooled results from lead_site_analysis/output/ and generates:
  - Forest plots (univariate, multivariable, sensitivity)
  - Funnel plot (publication bias assessment)
  - CONSORT flow diagram
  - Model diagnostics comparison across sites

All figures saved to lead_site_analysis/figures/
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from scipy.stats import norm

# =====================================================
# PATHS
# =====================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

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
    'age_scale:ph_scale': 'Age x pH interaction',
    'pco2_scale:rr_scale': 'pCO2 x RR interaction',
}

# Preferred display order
PREDICTOR_ORDER = [
    'age_scale', 'female', 'pco2_scale', 'ph_scale', 'map_scale',
    'rr_scale', 'hr_scale', 'fio2_high', 'peep_scale', 'tidal_volume_scale',
    'age_scale:ph_scale', 'pco2_scale:rr_scale',
]


def get_label(var):
    return PREDICTOR_LABELS.get(var, var)


def sort_by_order(df, col='Variable'):
    """Sort DataFrame rows by PREDICTOR_ORDER."""
    order_map = {v: i for i, v in enumerate(PREDICTOR_ORDER)}
    df = df.copy()
    df['_sort'] = df[col].map(order_map).fillna(99)
    df = df.sort_values('_sort').drop(columns='_sort')
    return df


# =====================================================
# FOREST PLOT — CUSTOM PUBLICATION QUALITY
# =====================================================

def forest_plot(pooled_df, site_df, title, filename, show_heterogeneity=True):
    """
    Publication-quality forest plot.

    For each predictor:
      - Site-specific estimates shown as squares (sized by weight)
      - Pooled estimate shown as a diamond
      - 95% CIs as horizontal lines
      - Annotations on the right: OR (95% CI)
    """
    if pooled_df.empty:
        print(f"  Skipping {filename}: no data")
        return

    pooled_df = sort_by_order(pooled_df)
    variables = pooled_df['Variable'].tolist()
    n_vars = len(variables)

    # Determine sites
    sites = sorted(site_df['site'].unique()) if not site_df.empty else []
    n_sites = len(sites)

    # Layout: for each variable, we need rows for each site + pooled + gap
    rows_per_var = n_sites + 1  # sites + pooled
    total_rows = n_vars * (rows_per_var + 1) - 1  # +1 gap between variables

    fig_height = max(6, total_rows * 0.3 + 2)
    fig, ax = plt.subplots(1, 1, figsize=(10, fig_height))

    y_positions = []
    y_labels = []
    y_pos = total_rows

    # Color scheme
    site_colors = plt.cm.Set2(np.linspace(0, 1, max(n_sites, 1)))

    for var_idx, var in enumerate(variables):
        var_pooled = pooled_df[pooled_df['Variable'] == var].iloc[0]

        # Site-specific estimates
        for site_idx, site in enumerate(sites):
            site_data = site_df[(site_df['Variable'] == var) & (site_df['site'] == site)]
            if site_data.empty:
                y_pos -= 1
                continue

            row = site_data.iloc[0]
            log_or = row['log_OR']
            se = row['SE_log_OR']
            or_val = np.exp(log_or)
            ci_lo = np.exp(log_or - 1.96 * se)
            ci_hi = np.exp(log_or + 1.96 * se)

            # Weight = 1/variance (for sizing the square)
            weight = 1.0 / (se ** 2) if se > 0 else 1.0

            # Plot CI line
            ax.plot([ci_lo, ci_hi], [y_pos, y_pos], color=site_colors[site_idx],
                    linewidth=1.0, zorder=2)
            # Plot square (sized by weight)
            marker_size = max(4, min(12, weight * 0.5))
            ax.plot(or_val, y_pos, 's', color=site_colors[site_idx],
                    markersize=marker_size, zorder=3)

            # Annotation
            ann = f"{or_val:.2f} ({ci_lo:.2f}-{ci_hi:.2f})"
            y_labels.append((y_pos, f"  {site}", ann, 'site'))
            y_pos -= 1

        # Pooled estimate (diamond)
        p_or = var_pooled['pooled_OR']
        p_lo = var_pooled['pooled_CI_lower']
        p_hi = var_pooled['pooled_CI_upper']

        # Draw diamond
        diamond_h = 0.3
        diamond = plt.Polygon([
            [p_lo, y_pos],
            [p_or, y_pos + diamond_h],
            [p_hi, y_pos],
            [p_or, y_pos - diamond_h]
        ], closed=True, facecolor='black', edgecolor='black', zorder=4)
        ax.add_patch(diamond)

        # Format p-value
        p_val = var_pooled['pooled_p']
        if pd.notna(p_val):
            p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
        else:
            p_str = "—"

        # Heterogeneity annotation
        het_str = ""
        if show_heterogeneity and n_sites > 1:
            I2 = var_pooled.get('I2', np.nan)
            Q_p = var_pooled.get('Q_p', np.nan)
            if pd.notna(I2):
                het_str = f" I²={I2:.0f}%"

        ann = f"{p_or:.2f} ({p_lo:.2f}-{p_hi:.2f}) p={p_str}{het_str}"
        y_labels.append((y_pos, f"{get_label(var)}", ann, 'pooled'))

        y_pos -= 2  # gap between variables

    # Reference line at OR = 1
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8, zorder=1)

    # Y-axis labels
    for yp, label, ann, ltype in y_labels:
        fontweight = 'bold' if ltype == 'pooled' else 'normal'
        fontsize = 9 if ltype == 'pooled' else 8
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else 0.01, yp,
                f" {label}", va='center', ha='left', fontsize=fontsize,
                fontweight=fontweight, transform=ax.get_yaxis_transform())

    ax.set_yticks([])
    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)

    # Use log scale for x-axis
    ax.set_xscale('log')

    # Set reasonable x limits
    all_ci_lo = pooled_df['pooled_CI_lower'].min()
    all_ci_hi = pooled_df['pooled_CI_upper'].max()
    x_min = max(0.05, all_ci_lo * 0.5)
    x_max = min(500, all_ci_hi * 2.0)
    ax.set_xlim(x_min, x_max)

    # Re-draw labels now that xlim is set
    ax.cla()
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8, zorder=1)
    ax.set_xscale('log')
    ax.set_xlim(x_min, x_max)

    # Redraw all elements with correct positioning
    y_pos = total_rows
    right_texts = []

    for var_idx, var in enumerate(variables):
        var_pooled = pooled_df[pooled_df['Variable'] == var].iloc[0]

        for site_idx, site in enumerate(sites):
            site_data = site_df[(site_df['Variable'] == var) & (site_df['site'] == site)]
            if site_data.empty:
                y_pos -= 1
                continue

            row = site_data.iloc[0]
            log_or = row['log_OR']
            se = row['SE_log_OR']
            or_val = np.exp(log_or)
            ci_lo = np.exp(log_or - 1.96 * se)
            ci_hi = np.exp(log_or + 1.96 * se)
            weight = 1.0 / (se ** 2) if se > 0 else 1.0

            ax.plot([ci_lo, ci_hi], [y_pos, y_pos], color=site_colors[site_idx],
                    linewidth=1.0, zorder=2)
            marker_size = max(4, min(12, weight * 0.3))
            ax.plot(or_val, y_pos, 's', color=site_colors[site_idx],
                    markersize=marker_size, zorder=3)

            ann = f"{or_val:.2f} ({ci_lo:.2f}, {ci_hi:.2f})"
            right_texts.append((y_pos, f"    {site}", ann, False))
            y_pos -= 1

        p_or = var_pooled['pooled_OR']
        p_lo = var_pooled['pooled_CI_lower']
        p_hi = var_pooled['pooled_CI_upper']

        diamond_h = 0.3
        diamond = plt.Polygon([
            [p_lo, y_pos], [p_or, y_pos + diamond_h],
            [p_hi, y_pos], [p_or, y_pos - diamond_h]
        ], closed=True, facecolor='black', edgecolor='black', zorder=4)
        ax.add_patch(diamond)

        p_val = var_pooled['pooled_p']
        p_str = "<0.001" if (pd.notna(p_val) and p_val < 0.001) else (f"{p_val:.3f}" if pd.notna(p_val) else "—")

        het_str = ""
        if show_heterogeneity and n_sites > 1:
            I2 = var_pooled.get('I2', np.nan)
            if pd.notna(I2):
                het_str = f"  I²={I2:.0f}%"

        ann = f"{p_or:.2f} ({p_lo:.2f}, {p_hi:.2f})  p={p_str}{het_str}"
        right_texts.append((y_pos, get_label(var), ann, True))
        y_pos -= 2

    # Draw labels
    for yp, label, ann, is_pooled in right_texts:
        fw = 'bold' if is_pooled else 'normal'
        fs = 9 if is_pooled else 7.5
        # Left label
        ax.text(0.0, yp, label, va='center', ha='right', fontsize=fs,
                fontweight=fw, transform=ax.get_yaxis_transform(), clip_on=False)
        # Right annotation
        ax.text(1.01, yp, ann, va='center', ha='left', fontsize=7,
                fontweight=fw, transform=ax.get_yaxis_transform(), clip_on=False)

    ax.set_yticks([])
    ax.set_ylim(y_pos - 1, total_rows + 1)
    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Legend for sites
    if n_sites > 1:
        handles = [mpatches.Patch(color=site_colors[i], label=s) for i, s in enumerate(sites)]
        handles.append(mpatches.Patch(color='black', label='Pooled (fixed-effect)'))
        ax.legend(handles=handles, loc='lower right', fontsize=7, framealpha=0.9)

    plt.tight_layout()
    plt.subplots_adjust(left=0.25, right=0.72)
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


# =====================================================
# FUNNEL PLOT
# =====================================================

def funnel_plot(site_df, pooled_df, title, filename):
    """Funnel plot: SE vs log(OR) for each predictor across sites."""
    if site_df.empty or pooled_df.empty:
        print(f"  Skipping {filename}: no data")
        return

    site_df = site_df[site_df['Variable'] != 'Intercept'].copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(site_df['Variable'].unique())))
    var_colors = {v: colors[i] for i, v in enumerate(site_df['Variable'].unique())}

    for _, row in site_df.iterrows():
        var = row['Variable']
        ax.scatter(row['log_OR'], row['SE_log_OR'],
                   color=var_colors.get(var, 'gray'), s=30, alpha=0.7,
                   edgecolors='black', linewidth=0.5)

    # Inverted y-axis (higher precision at top)
    ax.invert_yaxis()
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)

    ax.set_xlabel('log(OR)', fontsize=10)
    ax.set_ylabel('Standard Error', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Legend with variable names
    handles = [mpatches.Patch(color=var_colors[v], label=get_label(v))
               for v in var_colors]
    ax.legend(handles=handles, fontsize=6, loc='lower right',
              ncol=2, framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


# =====================================================
# CONSORT FLOW DIAGRAM
# =====================================================

def consort_diagram(consort_df, filename):
    """CONSORT-style flow diagram showing cohort selection across sites."""
    if consort_df.empty:
        print(f"  Skipping {filename}: no data")
        return

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Sum across sites for pooled counts
    numeric = consort_df.select_dtypes(include=[np.number])
    totals = numeric.sum()

    boxes = [
        (5, 13, f"Total ICU Admissions\nn = {int(totals.get('total_admissions', 0)):,}"),
        (5, 11, f"NIPPV within 6h\nn = {int(totals.get('total_nippv_6h', 0)):,}"),
        (5, 9, f"FiO2 < 0.6\nn = {int(totals.get('total_fio2_60', 0)):,}"),
        (5, 7, f"pCO2 >= 45 mmHg\nn = {int(totals.get('total_pco2_45', 0)):,}"),
        (5, 5, f"pH <= 7.35\nn = {int(totals.get('total_ph_7.35', 0)):,}"),
        (5, 3, f"Complete Cases (Analyzable)\nn = {int(totals.get('patients_post_missing', 0)):,}"),
    ]

    # Exclusions (right side)
    excl_data = [
        (int(totals.get('total_admissions', 0)) - int(totals.get('total_nippv_6h', 0)),
         "No NIPPV within 6h"),
        (int(totals.get('total_nippv_6h', 0)) - int(totals.get('total_fio2_60', 0)),
         "FiO2 >= 0.6"),
        (int(totals.get('total_fio2_60', 0)) - int(totals.get('total_pco2_45', 0)),
         "pCO2 < 45 mmHg"),
        (int(totals.get('total_pco2_45', 0)) - int(totals.get('total_ph_7.35', 0)),
         "pH > 7.35"),
        (int(totals.get('patients_pre_missing', 0)) - int(totals.get('patients_post_missing', 0)),
         "Missing data"),
    ]

    for i, (x, y, text) in enumerate(boxes):
        box = FancyBboxPatch((x - 1.8, y - 0.6), 3.6, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor='lightsteelblue', edgecolor='navy',
                              linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

        # Arrow down
        if i < len(boxes) - 1:
            ax.annotate('', xy=(x, boxes[i + 1][1] + 0.6),
                        xytext=(x, y - 0.6),
                        arrowprops=dict(arrowstyle='->', color='navy', lw=1.5))

        # Exclusion box (right side)
        if i < len(excl_data):
            excl_n, excl_reason = excl_data[i]
            if excl_n > 0:
                ex = x + 3.5
                ey = y - 1.0
                ebox = FancyBboxPatch((ex - 1.5, ey - 0.4), 3.0, 0.8,
                                       boxstyle="round,pad=0.1",
                                       facecolor='mistyrose', edgecolor='darkred',
                                       linewidth=1.0)
                ax.add_patch(ebox)
                ax.text(ex, ey, f"Excluded: {excl_reason}\nn = {excl_n:,}",
                        ha='center', va='center', fontsize=7, color='darkred')
                ax.annotate('', xy=(ex - 1.5, ey),
                            xytext=(x + 1.8, y - 1.0),
                            arrowprops=dict(arrowstyle='->', color='darkred',
                                            lw=1.0, ls='--'))

    # Outcome box at bottom
    n_fail = int(totals.get('failures_post_missing', 0))
    n_success = int(totals.get('patients_post_missing', 0)) - n_fail
    outcome_text = (f"NIPPV Failure: n = {n_fail:,}\n"
                    f"NIPPV Success: n = {n_success:,}")
    obox = FancyBboxPatch((3.2, 0.8), 3.6, 1.2,
                           boxstyle="round,pad=0.1",
                           facecolor='lightyellow', edgecolor='goldenrod',
                           linewidth=1.5)
    ax.add_patch(obox)
    ax.text(5, 1.4, outcome_text, ha='center', va='center',
            fontsize=9, fontweight='bold')
    ax.annotate('', xy=(5, 2.0), xytext=(5, 2.4),
                arrowprops=dict(arrowstyle='->', color='navy', lw=1.5))

    # Site header
    site_str = ', '.join(consort_df['site'].unique()) if 'site' in consort_df.columns else ''
    ax.text(5, 13.8, f"CONSORT Flow Diagram — Pooled ({site_str})",
            ha='center', va='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


# =====================================================
# DIAGNOSTICS COMPARISON
# =====================================================

def diagnostics_comparison(diag_df, filename):
    """Bar chart comparing AUC, Brier score, calibration across sites."""
    if diag_df.empty:
        print(f"  Skipping {filename}: no data")
        return

    models = diag_df['model'].unique()
    sites = diag_df['site'].unique()
    n_models = len(models)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics = [
        ('AUC', 'AUC (c-statistic)', 'AUC_SE'),
        ('brier_score', 'Brier Score', None),
        ('cal_slope', 'Calibration Slope', None),
    ]

    x = np.arange(len(sites))
    width = 0.35

    for ax_idx, (metric, label, se_col) in enumerate(metrics):
        ax = axes[ax_idx]

        for m_idx, model in enumerate(models):
            model_data = diag_df[diag_df['model'] == model]
            vals = []
            errs = []
            for site in sites:
                site_data = model_data[model_data['site'] == site]
                if not site_data.empty:
                    vals.append(site_data[metric].values[0])
                    if se_col and se_col in site_data.columns:
                        errs.append(site_data[se_col].values[0] * 1.96)
                    else:
                        errs.append(0)
                else:
                    vals.append(0)
                    errs.append(0)

            offset = (m_idx - n_models / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=model.replace('_', ' '),
                          yerr=errs if any(e > 0 for e in errs) else None,
                          capsize=3)

        ax.set_xlabel('Site')
        ax.set_ylabel(label)
        ax.set_title(label, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sites, rotation=45, ha='right')

        if metric == 'AUC':
            ax.set_ylim(0.5, 1.0)
            ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.5)
        elif metric == 'cal_slope':
            ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.5,
                        label='Ideal')

        ax.legend(fontsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Model Diagnostics by Site', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


# =====================================================
# SENSITIVITY COMPARISON: MLE vs FIRTH
# =====================================================

def sensitivity_plot(pooled_mle, pooled_firth, title, filename):
    """Side-by-side comparison of MLE vs Firth pooled ORs."""
    if pooled_mle.empty or pooled_firth.empty:
        print(f"  Skipping {filename}: no data")
        return

    # Merge on Variable
    merged = pooled_mle.merge(pooled_firth, on='Variable', suffixes=('_mle', '_firth'))
    merged = sort_by_order(merged)

    fig, ax = plt.subplots(figsize=(10, max(4, len(merged) * 0.6)))

    y = np.arange(len(merged))
    offset = 0.15

    # MLE
    ax.errorbar(merged['pooled_OR_mle'], y + offset,
                xerr=[merged['pooled_OR_mle'] - merged['pooled_CI_lower_mle'],
                      merged['pooled_CI_upper_mle'] - merged['pooled_OR_mle']],
                fmt='s', color='steelblue', markersize=6, capsize=3,
                label='MLE', linewidth=1.2)

    # Firth
    ax.errorbar(merged['pooled_OR_firth'], y - offset,
                xerr=[merged['pooled_OR_firth'] - merged['pooled_CI_lower_firth'],
                      merged['pooled_CI_upper_firth'] - merged['pooled_OR_firth']],
                fmt='D', color='darkorange', markersize=5, capsize=3,
                label='Firth', linewidth=1.2)

    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xscale('log')
    ax.set_yticks(y)
    ax.set_yticklabels([get_label(v) for v in merged['Variable']], fontsize=9)
    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


# =====================================================
# MAIN
# =====================================================

if __name__ == '__main__':
    print("=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    # Load pooled results
    def load(name):
        path = os.path.join(OUTPUT_DIR, name)
        return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

    pooled_uni = load('pooled_univariate.csv')
    pooled_multi_no = load('pooled_multivariable_NoInteraction.csv')
    pooled_multi_int = load('pooled_multivariable_Interaction.csv')
    pooled_firth_no = load('pooled_firth_NoInteraction.csv')
    pooled_firth_int = load('pooled_firth_Interaction.csv')

    site_uni = load('site_level_univariate.csv')
    site_multi_no = load('site_level_multi_no_interaction.csv')
    site_multi_int = load('site_level_multi_interaction.csv')

    consort_all = load('all_sites_consort.csv')
    diag_all = load('all_sites_diagnostics.csv')

    # --- Forest plots ---
    print("\n--- Forest Plots ---")
    forest_plot(pooled_uni, site_uni,
                'Univariate Logistic Regression — Pooled Odds Ratios',
                'forest_univariate.png')

    forest_plot(pooled_multi_no, site_multi_no,
                'Multivariable Model (No Interaction) — Pooled Odds Ratios',
                'forest_multivariable_no_interaction.png')

    forest_plot(pooled_multi_int, site_multi_int,
                'Multivariable Model (With Interactions) — Pooled Odds Ratios',
                'forest_multivariable_interaction.png')

    # --- Funnel plots ---
    print("\n--- Funnel Plots ---")
    funnel_plot(site_multi_no, pooled_multi_no,
                'Funnel Plot — Multivariable Model (No Interaction)',
                'funnel_no_interaction.png')

    # --- CONSORT ---
    print("\n--- CONSORT Diagram ---")
    consort_diagram(consort_all, 'consort_flow_diagram.png')

    # --- Diagnostics ---
    print("\n--- Diagnostics Comparison ---")
    diagnostics_comparison(diag_all, 'diagnostics_comparison.png')

    # --- Sensitivity: MLE vs Firth ---
    print("\n--- Sensitivity Analysis ---")
    sensitivity_plot(pooled_multi_no, pooled_firth_no,
                     'MLE vs Firth — No Interaction Model',
                     'sensitivity_mle_vs_firth_no_interaction.png')
    sensitivity_plot(pooled_multi_int, pooled_firth_int,
                     'MLE vs Firth — Interaction Model',
                     'sensitivity_mle_vs_firth_interaction.png')

    print(f"\nAll figures saved to: {FIG_DIR}/")
