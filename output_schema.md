# NIPPV-Pred Output Schema

Each CLIF site must share the following CSV files with the lead site after running the pipeline. All files are generated in the `output_to_share/` directory.

## Site Export Files (Stage 1)

### 1. `consort.csv` — Cohort Selection Counts (flat)
| Column | Type | Description |
|--------|------|-------------|
| site | string | Site identifier (from config.json) |
| total_rows_loaded | int | Total rows in wide dataset |
| total_admissions | int | Unique ICU admissions |
| total_nippv_6h | int | Admissions with NIPPV within 6h of first vital |
| total_fio2_60 | int | After FiO2 <= 60% filter |
| total_pco2_45 | int | After pCO2 >= 45 filter |
| total_ph_7.35 | int | After pH <= 7.35 filter |
| patients_pre_missing | int | Patients before dropping missing data |
| failures_pre_missing | int | NIPPV failures before dropping missing data |
| imv_fail_pre_missing | int | IMV-only failures (excluding both) |
| death_fail_pre_missing | int | Death-only failures (excluding both) |
| both_fail_pre_missing | int | Both IMV and death |
| patients_post_missing | int | Final analyzable N |
| failures_post_missing | int | Final NIPPV failures |
| imv_fail_post_missing | int | Final IMV-only failures |
| death_fail_post_missing | int | Final death-only failures |
| both_fail_post_missing | int | Final both failures |

### 2. `consort_flow.csv` — CONSORT Flow Diagram
| Column | Type | Description |
|--------|------|-------------|
| step | int | Step number (1-6) |
| description | string | Step description |
| n_remaining | int | Patients remaining after this step |
| n_excluded | int | Patients excluded at this step |
| exclusion_reason | string | Reason for exclusion |
| site | string | Site identifier |
| n_failure_yes | int | NIPPV failures (final step only) |
| n_failure_no | int | NIPPV successes (final step only) |

### 3. `descriptive_characteristics.csv` — Table 1
| Column | Type | Description |
|--------|------|-------------|
| Variable | string | Clinical variable name |
| NIPPV Failure: Yes | string | Mean +/- SD or n (%) for failure group |
| NIPPV Failure: No | string | Mean +/- SD or n (%) for success group |
| Total | string | Mean +/- SD or n (%) overall |
| p-value | string | t-test (continuous) or chi-square (categorical) |
| site | string | Site identifier |
| N | int | Total analyzable patients |
| N_events | int | Total NIPPV failures |

### 4. `univariate_logistic_results.csv` — Crude ORs
| Column | Type | Description |
|--------|------|-------------|
| Variable | string | Predictor name |
| log_OR | float | Log odds ratio (coefficient) |
| SE_log_OR | float | Standard error of log(OR) — **required for meta-analysis** |
| Odds Ratio | float | exp(log_OR) |
| P-Value | float | Wald p-value |
| 95% CI Lower | float | Lower bound of 95% CI for OR |
| 95% CI Upper | float | Upper bound of 95% CI for OR |
| site | string | Site identifier |
| N | int | Total analyzable patients |
| N_events | int | Total NIPPV failures |
| model_type | string | "univariate" |

### 5. `multivariable_logistic_results_NoInteraction.csv` — Adjusted ORs
Same schema as univariate, with `model_type = "multivariable_no_interaction"`.
**Includes Intercept row** — required for risk prediction pooling.

### 6. `multivariable_logistic_results_Interaction.csv` — Adjusted ORs with Interactions
Same schema as above, with `model_type = "multivariable_interaction"`.
Additional variables: `age_scale:ph_scale`, `pco2_scale:rr_scale`.

### 7. `model_diagnostics_NoInteraction.csv`
| Column | Type | Description |
|--------|------|-------------|
| site | string | Site identifier |
| model | string | Model label |
| N | int | Sample size |
| N_events | int | Number of events |
| AUC | float | Area under ROC curve (c-statistic) |
| AUC_SE | float | Standard error of AUC (Hanley-McNeil) |
| cal_intercept | float | Calibration-in-the-large (ideal=0) |
| cal_slope | float | Calibration slope (ideal=1) |
| brier_score | float | Brier score (mean squared prediction error) |
| EPV | float | Events per variable |
| n_predictors | int | Number of predictor variables |
| log_likelihood | float | Log-likelihood |
| AIC | float | Akaike Information Criterion |
| BIC | float | Bayesian Information Criterion |
| pseudo_r2 | float | McFadden's pseudo R-squared |
| converged | bool | Whether optimization converged |

### 8. `model_diagnostics_Interaction.csv`
Same as above, plus:
| Column | Type | Description |
|--------|------|-------------|
| LR_test_vs_no_interaction_chi2 | float | Likelihood ratio test statistic |
| LR_test_vs_no_interaction_p | float | LR test p-value |

### 9. `vif_NoInteraction.csv`
| Column | Type | Description |
|--------|------|-------------|
| Variable | string | Predictor name |
| VIF | float | Variance Inflation Factor (flag if >5) |
| site | string | Site identifier |

### 10. `vcov_matrix_NoInteraction.csv`
Variance-covariance matrix from the MLE logistic regression (no interaction model).
- Rows and columns are model parameter names (Intercept + predictors)
- 11x11 matrix for the no-interaction model
- Used for multivariate fixed-effects (GLS) pooling at the lead site
- **Privacy**: Safe to share — aggregate statistics, not patient data (Wu et al. 2012 "GLORE")

### 11. `vcov_matrix_Interaction.csv`
Same as above for the interaction model. 13x13 matrix (10 main + 2 interactions + intercept).

### 12. `firth_multivariable_results_NoInteraction.csv`
Firth's penalized logistic regression results — sensitivity analysis for small-sample bias.
Same schema as `multivariable_logistic_results_NoInteraction.csv` with `model_type = "firth_no_interaction"`.

### 13. `firth_multivariable_results_Interaction.csv`
Same as above for the interaction model, with `model_type = "firth_interaction"`.

### 14. `missingness_table.csv` (TRIPOD+AI + STROBE required)
Generated BEFORE complete case deletion to report missing data patterns.
| Column | Type | Description |
|--------|------|-------------|
| variable | string | Predictor variable name (raw, pre-scaling) |
| N_total | int | Total patients in cohort before dropping missing |
| N_missing | int | Number of patients with missing values |
| N_observed | int | Number of patients with observed values |
| Pct_missing | float | Percentage missing (0-100) |
| site | string | Site identifier |

## Complete Site Export Inventory

| File | Contents | Status |
|------|----------|--------|
| `univariate_logistic_results.csv` | Per-predictor crude ORs | Existing |
| `multivariable_logistic_results_NoInteraction.csv` | Adjusted ORs (incl. Intercept) | Existing |
| `multivariable_logistic_results_Interaction.csv` | Adjusted ORs with interactions | Existing |
| `firth_multivariable_results_NoInteraction.csv` | Firth sensitivity | NEW |
| `firth_multivariable_results_Interaction.csv` | Firth sensitivity | NEW |
| `vcov_matrix_NoInteraction.csv` | VCV matrix (11x11) | NEW |
| `vcov_matrix_Interaction.csv` | VCV matrix (13x13) | NEW |
| `model_diagnostics_NoInteraction.csv` | AUC, AUC_SE, calibration, EPV | UPDATED |
| `model_diagnostics_Interaction.csv` | Same + LR test | UPDATED |
| `missingness_table.csv` | Per-variable missing data | NEW |
| `vif_NoInteraction.csv` | Multicollinearity | Existing |
| `descriptive_characteristics.csv` | Table 1 with p-values | Existing |
| `consort_flow.csv` | CONSORT diagram data | Existing |
| `consort.csv` | Flat cohort counts | Existing |

## Predictor Variables

All models use these predictors with consistent names:

| Variable | Description | Scaling |
|----------|-------------|---------|
| age_scale | Age | (age - mean) / 10 — per 10-year increase |
| female | Sex | Binary: 1=Female, 0=Male (reference) |
| pco2_scale | pCO2 | (pCO2 - mean) / 10 — per 10 mmHg increase |
| ph_scale | pH | (pH - mean) / 0.1 — per 0.1 unit increase |
| map_scale | Mean Arterial Pressure | (MAP - mean) / 10 — per 10 mmHg increase |
| rr_scale | Respiratory Rate | (RR - mean) / 5 — per 5 bpm increase |
| hr_scale | Heart Rate | (HR - mean) / 10 — per 10 bpm increase |
| fio2_high | FiO2 | Binary: 1 if FiO2 > 0.4, 0 otherwise |
| peep_scale | PEEP | (PEEP - mean) / 2 — per 2 cmH2O increase |
| tidal_volume_scale | Tidal Volume | (TV - mean) / 100 — per 100 mL increase |

## Validation Checklist

Before sharing results with the lead site, verify:

- [ ] All CSV files are present in `output_to_share/`
- [ ] `site` column is populated in every file (not "unknown")
- [ ] `SE_log_OR` column is present and non-zero in regression files
- [ ] `N` and `N_events` are consistent across files
- [ ] No patient-level data is included (only aggregate statistics)
- [ ] Pipeline log shows "completed successfully" with no errors
- [ ] `vcov_matrix_*.csv` files are present and square
- [ ] `AUC_SE` is present in model diagnostics
- [ ] `cal_slope`, `cal_intercept`, `brier_score` are in diagnostics
- [ ] `missingness_table.csv` is present
- [ ] `firth_multivariable_results_*.csv` files are present
