# Lead-Site Meta-Analysis — Pooling Pipeline

Step-by-step instructions for the Rush lead team to pool site-level results and generate publication figures and tables.

**This pipeline is run ONLY by the lead site. Participating CLIF sites do NOT run anything in this directory.**

---

## Prerequisites

- All participating sites have completed their site-level pipeline (`./run_all.sh`)
- Each site has uploaded their **14 CSV files** to the shared Box folder
- You have downloaded all site results to your local machine
- Python >= 3.12 and [uv](https://docs.astral.sh/uv/) are installed
- Dependencies are installed (`uv sync` from the repo root)

---

## Step 1: Organize Site Results

Create a subdirectory under `all_site_results/` for each site. The directory name should match the site identifier used in their CSVs.

```bash
cd nippv-pred

# Create a directory for each site
mkdir -p all_site_results/rush
mkdir -p all_site_results/bidmc
# mkdir -p all_site_results/<additional_site>
```

## Step 2: Download and Place Site CSVs

Copy each site's 14 CSV files into their corresponding directory:

```bash
# Example: copy Rush results from Box download
cp ~/Downloads/rush_results/*.csv all_site_results/rush/

# Example: copy BIDMC results from Box download
cp ~/Downloads/bidmc_results/*.csv all_site_results/bidmc/
```

Each site directory should contain exactly these 14 files:

```
all_site_results/<site_name>/
    consort.csv
    consort_flow.csv
    descriptive_characteristics.csv
    missingness_table.csv
    univariate_logistic_results.csv
    multivariable_logistic_results_NoInteraction.csv
    multivariable_logistic_results_Interaction.csv
    firth_multivariable_results_NoInteraction.csv
    firth_multivariable_results_Interaction.csv
    model_diagnostics_NoInteraction.csv
    model_diagnostics_Interaction.csv
    vif_NoInteraction.csv
    vcov_matrix_NoInteraction.csv
    vcov_matrix_Interaction.csv
```

## Step 3: Verify Site Data

Before running the pooling pipeline, check that all files are present:

```bash
# List all files per site
for site in all_site_results/*/; do
    echo "=== $(basename $site) ==="
    ls "$site" | wc -l
    echo "files:"
    ls "$site"
    echo ""
done
```

Each site should show **14 files**. If any are missing, contact the site to re-run and re-upload.

## Step 4: Run the Pooling Pipeline

```bash
./lead_site_analysis/run_pooling.sh
```

This runs three scripts in sequence:

### Step 4a: Pool Results (`01_pool_results.py`)

- Reads all site CSVs from `all_site_results/*/`
- Runs **inverse-variance fixed-effects meta-analysis** for each predictor using `statsmodels.stats.meta_analysis.combine_effects`
- Computes heterogeneity statistics: Cochran's Q, I-squared, tau-squared
- Aggregates CONSORT diagrams, descriptive characteristics, diagnostics, missingness, and VIF across all sites
- Exports pooled results to `lead_site_analysis/output/`

**Pooled output files:**

| File | Description |
|------|-------------|
| `pooled_univariate.csv` | Pooled crude odds ratios for each predictor |
| `pooled_multivariable_NoInteraction.csv` | Pooled adjusted ORs (no interaction) |
| `pooled_multivariable_Interaction.csv` | Pooled adjusted ORs (with interactions) |
| `pooled_firth_NoInteraction.csv` | Pooled Firth sensitivity (no interaction) |
| `pooled_firth_Interaction.csv` | Pooled Firth sensitivity (with interactions) |
| `consort_pooled.csv` | Per-site + pooled CONSORT counts |
| `all_sites_consort_flow.csv` | CONSORT flow data from all sites |
| `all_sites_descriptive.csv` | Descriptive characteristics from all sites |
| `all_sites_diagnostics.csv` | Model diagnostics from all sites |
| `all_sites_missingness.csv` | Missingness patterns from all sites |
| `all_sites_vif.csv` | VIF from all sites |
| `site_level_*.csv` | Concatenated site-level data (used by figures script) |

### Step 4b: Generate Figures (`02_figures.py`)

- Generates publication-quality PNG figures at 300 DPI
- Saves to `lead_site_analysis/figures/`

**Figures generated:**

| File | Description |
|------|-------------|
| `forest_univariate.png` | Forest plot: univariate ORs (site squares + pooled diamond) |
| `forest_multivariable_no_interaction.png` | Forest plot: adjusted ORs, no interaction model |
| `forest_multivariable_interaction.png` | Forest plot: adjusted ORs, interaction model |
| `funnel_no_interaction.png` | Funnel plot: SE vs log(OR) for publication bias |
| `consort_flow_diagram.png` | CONSORT flow diagram with pooled counts |
| `diagnostics_comparison.png` | AUC, Brier score, calibration slope by site |
| `sensitivity_mle_vs_firth_no_interaction.png` | MLE vs Firth comparison, no interaction |
| `sensitivity_mle_vs_firth_interaction.png` | MLE vs Firth comparison, interaction model |

### Step 4c: Generate Tables (`03_tables.py`)

- Generates publication-ready CSV tables
- Saves to `lead_site_analysis/tables/`

**Tables generated:**

| File | Description |
|------|-------------|
| `table1_descriptive.csv` | Pooled baseline characteristics (Table 1) |
| `table2_regression_results.csv` | Crude + adjusted ORs side by side (Table 2) |
| `table2b_interaction_results.csv` | Interaction model results |
| `table3_sensitivity.csv` | MLE vs Firth comparison |
| `table4_diagnostics.csv` | Model diagnostics across sites |
| `table5_heterogeneity.csv` | Q, I-squared, tau-squared per predictor |
| `table6_consort.csv` | CONSORT data per site + pooled |
| `table7_missingness.csv` | Missingness across sites |

## Step 5: Review Results

After the pipeline completes:

1. **Check the console output** for any warnings (missing files, degenerate variances, etc.)

2. **Review forest plots** in `lead_site_analysis/figures/`:
   - Verify all sites appear in each plot
   - Check that pooled diamonds are positioned correctly
   - Confirm OR scale is reasonable (no extreme values from data errors)

3. **Review heterogeneity** in `lead_site_analysis/tables/table5_heterogeneity.csv`:
   - I-squared < 25% = low heterogeneity
   - I-squared 25-75% = moderate
   - I-squared > 75% = high (may need random-effects model)

4. **Compare MLE vs Firth** in `lead_site_analysis/tables/table3_sensitivity.csv`:
   - Large differences suggest small-sample bias in MLE estimates
   - Firth results should be reported as sensitivity analysis

5. **Check diagnostics** in `lead_site_analysis/tables/table4_diagnostics.csv`:
   - AUC > 0.7 = acceptable discrimination
   - Calibration slope near 1.0 = good calibration
   - EPV >= 10 recommended (if below, note in manuscript)

## Step 6: Export for Manuscript

Copy the files you need for the manuscript:

```bash
# Figures for the paper
cp lead_site_analysis/figures/forest_multivariable_no_interaction.png ~/manuscript/figures/
cp lead_site_analysis/figures/consort_flow_diagram.png ~/manuscript/figures/

# Tables for the paper
cp lead_site_analysis/tables/table1_descriptive.csv ~/manuscript/tables/
cp lead_site_analysis/tables/table2_regression_results.csv ~/manuscript/tables/
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No site directories found` | Create subdirectories under `all_site_results/` and place CSVs inside |
| `WARNING: <file> not found in <site>/` | That site is missing a CSV; contact them to re-run and upload |
| `WARNING: Invalid variance for <var>` | SE_log_OR is zero or NaN for that predictor at some site; check their regression output |
| Forest plot looks wrong | Check that `log_OR` and `SE_log_OR` columns are present in site CSVs (old-format CSVs without these columns cannot be pooled) |
| `ModuleNotFoundError` | Run `uv sync` from the repo root to install dependencies |

---

## Directory Structure

```
lead_site_analysis/
    README.md               # This file
    run_pooling.sh          # Run the full pipeline (Steps 4a-4c)
    01_pool_results.py      # Meta-analysis pooling
    02_figures.py           # Forest plots, funnel plot, CONSORT, diagnostics
    03_tables.py            # Publication tables
    output/                 # Pooled CSVs (gitignored)
    figures/                # PNG figures at 300 DPI (gitignored)
    tables/                 # CSV tables (gitignored)
```
