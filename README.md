# NIPPV-Pred

Predictors of Noninvasive Positive Pressure Ventilation (NIPPV) failure in patients with acute hypercapnic respiratory failure — a federated analysis using the [Common Longitudinal ICU data Format (CLIF)](https://clif-icu.com).

## Prerequisites

- **Python** >= 3.12
- **[uv](https://docs.astral.sh/uv/)** — Python package manager
- **Local CLIF parquet files** — your site's CLIF tables in Parquet format

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Common-Longitudinal-ICU-data-Format/nippv-pred.git
cd nippv-pred

# 2. Configure your site settings
cp config_template.json config.json
# Edit config.json — set your site name, data directory path, filetype, and timezone

# 3. Install dependencies
uv sync

# 4. Run the full pipeline
./run_all.sh
```

## Pipeline Steps

| Step | Script | Description | Output |
|------|--------|-------------|--------|
| 1 | `code/01_wide_generator.py` | Transforms longitudinal CLIF tables to wide (hourly) format | `output/NIPPV_analytic_dataset.csv` |
| 2 | `code/02_study_cohort.ipynb` | Applies inclusion/exclusion criteria to identify the study cohort | Filtered analytic dataset |
| 3 | `code/03_descriptive_characteristics.py` | Computes baseline characteristics stratified by NIPPV failure | `output_to_share/descriptive_characteristics.csv`, `consort.csv`, `consort_flow.csv`, `missingness_table.csv` |
| 4 | `code/04_analysis_no_interaction.py` | Univariate + multivariable logistic regression (no interaction terms) | Regression results, diagnostics, VIF, VCV matrix, Firth sensitivity |
| 5 | `code/05_analysis_interaction.py` | Multivariable logistic regression with interaction terms | Regression results, diagnostics, VCV matrix, Firth sensitivity |

## Output Files

The pipeline produces **14 CSV files** in `output_to_share/`. All files must be uploaded to Box.

| File | Contents |
|------|----------|
| `consort.csv` | Cohort selection counts (flat) |
| `consort_flow.csv` | CONSORT flow diagram data |
| `descriptive_characteristics.csv` | Table 1 — baseline characteristics by NIPPV failure |
| `missingness_table.csv` | Per-variable missing data patterns |
| `univariate_logistic_results.csv` | Crude odds ratios for each predictor |
| `multivariable_logistic_results_NoInteraction.csv` | Adjusted odds ratios (no interaction) |
| `multivariable_logistic_results_Interaction.csv` | Adjusted odds ratios (with interactions) |
| `firth_multivariable_results_NoInteraction.csv` | Firth penalized regression (no interaction) |
| `firth_multivariable_results_Interaction.csv` | Firth penalized regression (with interactions) |
| `model_diagnostics_NoInteraction.csv` | AUC, calibration, Brier score (no interaction) |
| `model_diagnostics_Interaction.csv` | AUC, calibration, Brier score (with interactions) |
| `vif_NoInteraction.csv` | Variance inflation factors |
| `vcov_matrix_NoInteraction.csv` | Variance-covariance matrix (no interaction) |
| `vcov_matrix_Interaction.csv` | Variance-covariance matrix (with interactions) |

See [`output_schema.md`](output_schema.md) for detailed column descriptions and a validation checklist.

## Upload Instructions

After the pipeline completes successfully:

1. Verify all **14 CSV files** are present in `output_to_share/`
2. Check the pipeline log for "Pipeline completed successfully"
3. Review the validation checklist in [`output_schema.md`](output_schema.md)
4. Upload all 14 files from `output_to_share/` to the shared Box folder per CLIF consortium protocol

**Important:** Only aggregate statistics are shared — no patient-level data leaves your institution.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `config.json not found` | Run `cp config_template.json config.json` and edit with your site settings |
| `FileNotFoundError` for parquet files | Verify `data_directory` in `config.json` points to your CLIF parquet tables |
| `uv: command not found` | Install uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `Permission denied: ./run_all.sh` | Run `chmod +x run_all.sh` |
| Notebook execution fails | Ensure Jupyter is installed (`uv sync` should handle this) and check that Step 1 output exists in `output/` |

## License

Apache 2.0
