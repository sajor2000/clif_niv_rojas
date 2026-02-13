# NIPPV-Pred

Predictors of Noninvasive Positive Pressure Ventilation (NIPPV) failure in patients with acute hypercapnic respiratory failure -- a federated analysis using the [Common Longitudinal ICU data Format (CLIF)](https://clif-icu.com).

## Study Overview

Noninvasive positive pressure ventilation (NIPPV) is a standard first-line treatment for acute hypercapnic respiratory failure (PaCO2 > 45 mmHg, pH <= 7.35). Despite its effectiveness, **5-40% of patients fail NIPPV** and require escalation to invasive mechanical ventilation. NIPPV failure is independently associated with increased ICU mortality, making early identification of at-risk patients a clinical priority.

This project uses multicenter ICU data standardized to the CLIF format to validate predictors of NIPPV failure across institutions, supporting earlier escalation decisions at the bedside.

- **Author:** Connor P. Lafeber (Rush University, M.S. thesis, 2025)
- **Design:** Federated meta-analysis -- patient-level data never leaves the originating institution; only aggregate statistical results are shared
- **Contact:** [Connor Lafeber](mailto:connor_p_lafeber@rush.edu), Rush University Medical Center

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

---

## For Participating Sites: Setup and Execution

### Prerequisites

Before you begin, ensure you have:

- **Python >= 3.12** -- [download here](https://www.python.org/downloads/) if not installed
- **[uv](https://docs.astral.sh/uv/)** -- Python package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh` on Mac/Linux, or `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` on Windows)
- **Git** -- [download here](https://git-scm.com/downloads)
- **Your site's CLIF data** in Parquet format (CLIF version **2.1+** required)

### Required CLIF Tables

Your CLIF Parquet directory must contain these tables (as individual `.parquet` files):

| CLIF Table | Used For |
|------------|----------|
| `patient` | Age, sex (demographics) |
| `hospitalization` | Encounter metadata, admission/discharge timestamps |
| `vitals` | Heart rate, MAP, respiratory rate |
| `labs` | pH, pCO2 (arterial and venous blood gases) |
| `respiratory_support` | NIPPV device identification, FiO2, PEEP, tidal volume |

Verify your directory contains at minimum: `patient.parquet`, `hospitalization.parquet`, `vitals.parquet`, `labs.parquet`, `respiratory_support.parquet`.

### Step 1: Clone the Repository

```bash
git clone https://github.com/sajor2000/clif_niv_rojas.git
cd clif_niv_rojas
```

### Step 2: Configure Your Site

```bash
cp config_template.json config.json
```

Edit `config.json` with your site-specific values:

```json
{
    "site": "rush",
    "data_directory": "/data/clif/parquet",
    "filetype": "parquet",
    "timezone": "US/Central"
}
```

| Field | Description | Examples |
|-------|-------------|---------|
| `site` | Your institution's short name, lowercase, no spaces | `rush`, `bidmc`, `ucsf`, `emory` |
| `data_directory` | **Absolute path** to the folder containing your CLIF `.parquet` files | Mac: `/Users/you/data/clif`<br>Linux: `/home/you/data/clif`<br>Windows: `C:/Users/you/data/clif` |
| `filetype` | Data format (always `parquet` for CLIF 2.1) | `parquet` |
| `timezone` | Your institution's timezone ([full list](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)) | `US/Central`, `US/Eastern`, `US/Pacific`, `US/Mountain` |

**Windows users:** Use forward slashes (`/`) or escaped backslashes (`\\`) in `data_directory`.

### Step 3: Install Dependencies

```bash
uv sync
```

This installs all required Python packages (clifpy, statsmodels, scipy, firthlogist, etc.) into an isolated virtual environment. No system-wide changes are made.

### Step 4: Run the Pipeline

**Mac / Linux:**

```bash
chmod +x run_all.sh    # first time only
./run_all.sh
```

**Windows (or any platform):**

```bash
uv run python run_all.py
```

Both runners execute the same 5 pipeline steps and produce identical output. The pipeline creates a timestamped log file (`pipeline_YYYYMMDD_HHMMSS.log`) in the repo root.

**Expected runtime:** 10-30 minutes depending on dataset size. The wide-format generation (Step 1) is typically the longest step.

### Pipeline Steps

| Step | Script | What It Does | Key Outputs |
|------|--------|-------------|-------------|
| 1 | `code/01_wide_generator.py` | Transforms longitudinal CLIF tables to wide (hourly) format using `clifpy` | `output/NIPPV_analytic_dataset.csv` |
| 2 | `code/02_study_cohort.ipynb` | Applies inclusion/exclusion criteria; builds CONSORT flow diagram | `output/NIPPV_analytic_dataset.csv` (filtered), `output_to_share/consort.csv`, `consort_flow.csv`, `missingness_table.csv` |
| 3 | `code/03_descriptive_characteristics.py` | Baseline characteristics stratified by NIPPV failure; t-tests, chi-squared, Fisher's exact | `output_to_share/descriptive_characteristics.csv` |
| 4 | `code/04_analysis_no_interaction.py` | Univariate + multivariable logistic regression (10 predictors); Firth sensitivity; model diagnostics | 8 CSV files in `output_to_share/` (see below) |
| 5 | `code/05_analysis_interaction.py` | Multivariable regression with 2 interaction terms; Firth sensitivity; LR test vs. no-interaction model | 4 CSV files in `output_to_share/` (see below) |

### Output Files (14 total)

After a successful run, `output_to_share/` will contain exactly **14 CSV files**:

| # | File | Description |
|---|------|-------------|
| 1 | `consort.csv` | Cohort selection counts (flat summary) |
| 2 | `consort_flow.csv` | Step-by-step CONSORT flow diagram data |
| 3 | `descriptive_characteristics.csv` | Table 1 -- baseline characteristics by NIPPV failure |
| 4 | `missingness_table.csv` | Per-variable missing data counts and percentages |
| 5 | `univariate_logistic_results.csv` | Crude odds ratios (log_OR, SE_log_OR, OR, 95% CI, p-value) |
| 6 | `multivariable_logistic_results_NoInteraction.csv` | Adjusted ORs, 10-predictor model |
| 7 | `multivariable_logistic_results_Interaction.csv` | Adjusted ORs, 12-predictor model (with interactions) |
| 8 | `firth_multivariable_results_NoInteraction.csv` | Firth penalized regression, 10 predictors |
| 9 | `firth_multivariable_results_Interaction.csv` | Firth penalized regression, 12 predictors |
| 10 | `model_diagnostics_NoInteraction.csv` | AUC, calibration slope/intercept, Brier score, EPV |
| 11 | `model_diagnostics_Interaction.csv` | Same diagnostics for the interaction model |
| 12 | `vif_NoInteraction.csv` | Variance inflation factors (multicollinearity check) |
| 13 | `vcov_matrix_NoInteraction.csv` | Variance-covariance matrix for multivariate pooling |
| 14 | `vcov_matrix_Interaction.csv` | Variance-covariance matrix for interaction model |

For detailed column descriptions, see [`output_schema.md`](output_schema.md).

### Step 5: Verify Your Output

Before uploading, run this quick check:

**Mac / Linux:**

```bash
echo "Files in output_to_share:" && ls -1 output_to_share/*.csv | wc -l && echo "Expected: 14"
```

**Windows (PowerShell):**

```powershell
(Get-ChildItem output_to_share\*.csv).Count
# Expected: 14
```

You should see exactly **14 files**. If any are missing, check the pipeline log for errors.

### Step 6: Upload Results to Box

1. Verify all **14 CSV files** are present in `output_to_share/` (Step 5 above)
2. Check the pipeline log for "Pipeline completed successfully" at the end
3. Open the shared Box folder link provided by the Rush lead team
4. Create a folder named after your site (e.g., `bidmc/`, `ucsf/`)
5. Upload all 14 CSV files from `output_to_share/` into your site folder
6. Notify the Rush team ([connor_p_lafeber@rush.edu](mailto:connor_p_lafeber@rush.edu)) that your upload is complete

**Important:** Only aggregate statistics are shared -- no patient-level data leaves your institution. The `output_to_share/` folder contains only odds ratios, p-values, standard errors, counts, and variance-covariance matrices. No individual patient records are included.

---

## Lead-Site Meta-Analysis

After all sites upload their results, the lead site (Rush) pools them into publication figures and tables. See [`lead_site_analysis/README.md`](lead_site_analysis/README.md) for detailed instructions. **Participating sites do NOT run this.**

---

## Statistical Methods

### Primary Analysis

- **MLE logistic regression** (statsmodels `Logit`) -- univariate and multivariable models
- Standard errors, Wald p-values, and 95% confidence intervals for odds ratios
- Two multivariable models: without interaction terms (10 predictors) and with interaction terms (10 predictors + 2 interactions)

### Sensitivity Analysis

- **Firth penalized logistic regression** ([firthlogist](https://pypi.org/project/firthlogist/)) -- addresses small-sample bias and separation in maximum likelihood estimation

### Model Diagnostics

- **Discrimination:** AUC (c-statistic) with Hanley-McNeil standard error; bootstrap optimism-corrected AUC (200 iterations)
- **Calibration:** Calibration-in-the-large (intercept) and calibration slope via logistic recalibration; bootstrap optimism-corrected calibration slope
- **Overall accuracy:** Brier score (raw and bootstrap optimism-corrected)
- **Events per variable (EPV):** Reported for model adequacy assessment (threshold: 10)
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
- **CLIF version required:** 2.1+
- **Data dictionary:** [clif-icu.com/data-dictionary](https://clif-icu.com/data-dictionary/data-dictionary-2.0.0)
- **Python package:** [clifpy](https://pypi.org/project/clifpy/) -- CLIF schema validation, SOFA scores, wide-format transforms
- **Publication:** Rojas et al. (2025) *Intensive Care Medicine* -- [doi:10.1007/s00134-025-07848-7](https://doi.org/10.1007/s00134-025-07848-7)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `config.json not found` | Run `cp config_template.json config.json` and edit with your site settings |
| `FileNotFoundError` for parquet files | Verify `data_directory` in `config.json` is an **absolute path** to your CLIF parquet directory containing `patient.parquet`, `hospitalization.parquet`, `vitals.parquet`, `labs.parquet`, `respiratory_support.parquet` |
| `uv: command not found` | Install uv: Mac/Linux: `curl -LsSf https://astral.sh/uv/install.sh \| sh`; Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| `Permission denied: ./run_all.sh` | Run `chmod +x run_all.sh`, or use the Python runner instead: `uv run python run_all.py` |
| `python: command not found` | Use `python3` instead of `python`, or use `uv run python` which finds the correct Python |
| Notebook execution fails | Ensure Step 1 completed successfully (check that `output/NIPPV_analytic_dataset.csv` exists). Try running manually: `uv run jupyter nbconvert --to notebook --execute code/02_study_cohort.ipynb` |
| Firth regression is slow | Normal for Firth penalization. Step 4 may take 5-10 minutes. The pipeline will print progress. |
| `ModuleNotFoundError` | Run `uv sync` to install dependencies. Do not use `pip install` directly. |
| Windows: `./run_all.sh` doesn't work | Use the Python runner instead: `uv run python run_all.py` |
| Wrong number of output files | Check the pipeline log (`pipeline_*.log`) for error messages. Each step must complete before the next starts. |
| `KeyError` or missing columns | Ensure your CLIF data is version 2.1+. Older CLIF versions may be missing required columns (e.g., `discharge_dttm` in hospitalization table). |

**Still stuck?** Contact [connor_p_lafeber@rush.edu](mailto:connor_p_lafeber@rush.edu) with your pipeline log file attached.

## References

- Rojas JC, Lyons PG, Chhikara K, et al. (2025). A common longitudinal intensive care unit data format (CLIF) for critical illness research. *Intensive Care Medicine*. [doi:10.1007/s00134-025-07848-7](https://doi.org/10.1007/s00134-025-07848-7)
- Confalonieri M, et al. (2005). A chart of failure risk for noninvasive ventilation in patients with COPD exacerbation. *Eur Respir J*.
- Ko BS, et al. (2015). Early failure of noninvasive ventilation in COPD with acute hypercapnic respiratory failure. *Intern Emerg Med*.
- Wu Y, et al. (2012). Grid Binary LOgistic REgression (GLORE): building shared models without sharing data. *J Am Med Inform Assoc*.

## License

Apache 2.0
