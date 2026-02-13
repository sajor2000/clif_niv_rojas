# NIPPV-Pred

Predictors of Noninvasive Positive Pressure Ventilation (NIPPV) failure in patients with acute hypercapnic respiratory failure — a federated analysis using the [Common Longitudinal ICU data Format (CLIF)](https://clif-icu.com).

## Study Overview

Noninvasive positive pressure ventilation (NIPPV) is a standard first-line treatment for acute hypercapnic respiratory failure (PaCO2 > 45 mmHg, pH <= 7.35). Despite its effectiveness, **5-40% of patients fail NIPPV** and require escalation to invasive mechanical ventilation. NIPPV failure is independently associated with increased ICU mortality, making early identification of at-risk patients a clinical priority.

This project uses multicenter ICU data standardized to the CLIF format to validate predictors of NIPPV failure across institutions, supporting earlier escalation decisions at the bedside.

- **Author:** Connor P. Lafeber (Rush University, M.S. thesis, 2025)
- **Participating Sites:** Rush University Medical Center, Beth Israel Deaconess Medical Center
- **Design:** Federated meta-analysis — patient-level data never leaves the originating institution; only aggregate statistical results are shared

## Objective

To identify predictors of noninvasive positive pressure ventilation (NIPPV) failure in patients with acute hypercapnic respiratory failure using the Common Longitudinal ICU data Format (CLIF).

## Outcome Definition

**NIPPV Failure** is defined as:

1. Death **or**
2. Escalation to invasive mechanical ventilation

within **48 hours** of NIPPV initiation.

## Cohort Identification

**Inclusion Criteria:**

1. NIPPV initiation within **6 hours** of the first recorded vital sign
2. Median pCO2 (arterial or venous) >= 45 mmHg prior to NIPPV initiation
3. Median pH (arterial or venous) <= 7.35 prior to NIPPV initiation

**Exclusion Criteria:**

1. Median FiO2 >= 0.6 prior to NIPPV initiation

## Predictors

Unless otherwise specified, physiologic predictors are defined as the **median value 1-12 hours following NIPPV initiation**. Continuous variables are scaled to improve interpretability of regression coefficients.

| # | Variable | Type | Scaling | Interpretation |
|---|----------|------|---------|----------------|
| 1 | `age_scale` | Continuous | (age - mean) / 10 | Per 10-year increase |
| 2 | `female` | Binary | 0/1 | Female vs. Male (reference) |
| 3 | `pco2_scale` | Continuous | (pCO2 - mean) / 10 | Median pCO2 (arterial or venous) per 10 mmHg increase |
| 4 | `ph_scale` | Continuous | (pH - mean) / 0.1 | Median pH (arterial or venous) per 0.1 unit increase |
| 5 | `map_scale` | Continuous | (MAP - mean) / 10 | Median mean arterial pressure per 10 mmHg increase |
| 6 | `rr_scale` | Continuous | (RR - mean) / 5 | Median respiratory rate per 5 bpm increase |
| 7 | `hr_scale` | Continuous | (HR - mean) / 10 | Median heart rate per 10 bpm increase |
| 8 | `fio2_high` | Binary | 0/1 | Median FiO2 > 0.4 vs. <= 0.4 (reference) |
| 9 | `peep_scale` | Continuous | (PEEP - mean) / 2 | Median PEEP per 2 cmH2O increase |
| 10 | `tidal_volume_scale` | Continuous | (TV - mean) / 100 | Median tidal volume per 100 mL increase |
| 11 | `age_scale * ph_scale` | Interaction | Age Scale x pH Scale | Age-acidosis interaction |
| 12 | `pco2_scale * rr_scale` | Interaction | pCO2 Scale x RR Scale | Hypercapnia-respiratory rate interaction |

## Prerequisites

- **Python** >= 3.12
- **[uv](https://docs.astral.sh/uv/)** -- Python package manager
- **Local CLIF parquet files** -- your site's CLIF tables in Parquet format

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
| `descriptive_characteristics.csv` | Table 1 -- baseline characteristics by NIPPV failure |
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

## Lead-Site Meta-Analysis

After all sites upload their results, the lead site (Rush) pools them into publication figures and tables. See [`lead_site_analysis/README.md`](lead_site_analysis/README.md) for detailed instructions. **Participating sites do NOT run this.**

## Statistical Methods

### Primary Analysis

- **MLE logistic regression** (statsmodels `Logit`) -- univariate and multivariable models
- Standard errors, Wald p-values, and 95% confidence intervals for odds ratios
- Two multivariable models: without interaction terms (10 predictors) and with interaction terms (10 predictors + 2 interactions)

### Sensitivity Analysis

- **Firth penalized logistic regression** ([firthlogist](https://pypi.org/project/firthlogist/)) -- addresses small-sample bias and separation in maximum likelihood estimation

### Model Diagnostics

- **Discrimination:** AUC (c-statistic) with Hanley-McNeil standard error
- **Calibration:** Calibration-in-the-large (intercept) and calibration slope via logistic recalibration
- **Overall accuracy:** Brier score
- **Events per variable (EPV):** Reported for model adequacy assessment
- **Multicollinearity:** Variance Inflation Factors (VIF) for the no-interaction model
- **Model fit:** Log-likelihood, AIC, BIC, McFadden's pseudo R-squared

### Interaction Testing

- **Likelihood ratio test** comparing the no-interaction model to the interaction model

### Meta-Analysis

- **Variance-covariance (VCV) matrices** exported for multivariate fixed-effects pooling across sites (GLORE framework; Wu et al., 2012)
- `log_OR` and `SE_log_OR` exported for standard inverse-variance meta-analysis

## Data Privacy

This project follows a federated analysis design in accordance with CLIF consortium protocols:

- Patient-level data **never leaves** the originating institution
- Each site runs the analysis pipeline locally against their own CLIF database
- Only aggregate statistics are shared (odds ratios, confidence intervals, p-values, variance-covariance matrices)
- Results are pooled at the lead site using fixed-effects meta-analysis

## CLIF Consortium

The **Common Longitudinal ICU data Format (CLIF)** is an open-source data standard for longitudinal ICU data enabling privacy-preserving multicenter critical care research.

- **Website:** [clif-icu.com](https://clif-icu.com)
- **GitHub:** [Common-Longitudinal-ICU-data-Format](https://github.com/Common-Longitudinal-ICU-data-Format)
- **Scale:** 808,000+ patients, 62 hospitals, 17 institutions
- **Python package:** [clifpy](https://pypi.org/project/clifpy/) -- CLIF schema validation, SOFA scores, wide-format transforms
- **Publication:** Rojas et al. (2025) *Intensive Care Medicine* -- [doi:10.1007/s00134-025-07848-7](https://doi.org/10.1007/s00134-025-07848-7)

## Upload Instructions

After the pipeline completes successfully:

1. Verify all **14 CSV files** are present in `output_to_share/`
2. Check the pipeline log for "Pipeline completed successfully"
3. Upload all 14 files from `output_to_share/` to the shared Box folder per CLIF consortium protocol

**Important:** Only aggregate statistics are shared -- no patient-level data leaves your institution.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `config.json not found` | Run `cp config_template.json config.json` and edit with your site settings |
| `FileNotFoundError` for parquet files | Verify `data_directory` in `config.json` points to your CLIF parquet tables |
| `uv: command not found` | Install uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `Permission denied: ./run_all.sh` | Run `chmod +x run_all.sh` |
| Notebook execution fails | Ensure Jupyter is installed (`uv sync` should handle this) and check that Step 1 output exists in `output/` |

## References

- Rojas JC, Lyons PG, Chhikara K, et al. (2025). A common longitudinal intensive care unit data format (CLIF) for critical illness research. *Intensive Care Medicine*. [doi:10.1007/s00134-025-07848-7](https://doi.org/10.1007/s00134-025-07848-7)
- Confalonieri M, et al. (2005). A chart of failure risk for noninvasive ventilation in patients with COPD exacerbation. *Eur Respir J*.
- Ko BS, et al. (2015). Early failure of noninvasive ventilation in COPD with acute hypercapnic respiratory failure. *Intern Emerg Med*.
- Wu Y, et al. (2012). Grid Binary LOgistic REgression (GLORE): building shared models without sharing data. *J Am Med Inform Assoc*.

## License

Apache 2.0
