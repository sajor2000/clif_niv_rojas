# NIPPV Failure Prediction — CLIF Consortium Project

## Project Overview

This project identifies **predictors of Noninvasive Positive Pressure Ventilation (NIPPV) failure** in patients with acute hypercapnic respiratory failure using the **Common Longitudinal ICU data Format (CLIF)**. It is a federated meta-analysis conducted across CLIF consortium sites where patient-level data never leaves the institution — only aggregate statistical results are shared.

**Author**: Connor P. Lafeber (Rush University, M.S. thesis, 2025)
**Participating Sites**: Rush University Medical Center, Beth Israel Deaconess Medical Center
**Upstream Repo**: https://github.com/Common-Longitudinal-ICU-data-Format/nippv-pred

## Clinical Context

- **NIPPV** is a standard treatment for hypercapnic respiratory failure (PaCO2 > 45 mmHg, pH <= 7.35)
- 5–40% of patients fail NIPPV and require invasive mechanical ventilation
- NIPPV failure is associated with increased mortality
- This study validates predictors to support earlier escalation decisions

### Key Findings (Pooled Meta-Analysis)

Independent predictors of NIPPV failure:
| Predictor | Pooled Adjusted OR | 95% CI | p-value |
|---|---|---|---|
| Male sex (vs female) | 0.362 | 0.181–0.727 | 0.004 |
| Lower pH (per 0.1 unit) | 0.583 | 0.372–0.914 | 0.018 |
| Lower MAP (per mmHg) | 0.949 | 0.909–0.991 | 0.018 |
| Higher HR (per 10 bpm) | 1.241 | 1.006–1.531 | 0.044 |
| Higher FiO2 | 37.49 | 5.01–280.60 | <0.001 |

## CLIF Consortium

The **Common Longitudinal ICU data Format (CLIF)** is an open-source data standard for longitudinal ICU data enabling privacy-preserving multicenter research.

- **Website**: https://clif-icu.com
- **GitHub Org**: https://github.com/Common-Longitudinal-ICU-data-Format
- **License**: Apache 2.0
- **Scale**: 808,000+ patients, 62 hospitals, 17 institutions
- **Publication**: Rojas et al. (2025) *Intensive Care Medicine* — https://doi.org/10.1007/s00134-025-07848-7

### CLIF Data Tables Used in This Project

The CLIF schema is an encounter-centric relational database. Key tables relevant to this project:

- **patient** — demographics (age, sex)
- **hospitalization** — encounter-level data
- **vitals** — HR, MAP, RR (timestamped longitudinal measurements)
- **labs** — pH, pCO2 (arterial/venous blood gases)
- **respiratory_support** — NIPPV device identification, FiO2, PEEP, tidal volume
- **medication_admin_continuous / medication_admin_intermittent** — medication data

Full CLIF schema includes 20+ tables: patient, hospitalization, admission_diagnosis, provider, adt, vitals, dialysis, intake_output, procedures, therapy_details, respiratory_support, position, patient_assessment, ecmo_mcs, labs, microbiology_culture, microbiology_nonculture, sensitivity, medication_orders, medication_admin_intermittent, medication_admin_continuous, code_status.

## Technical Stack

### Languages & Tools
- **Python** >= 3.12 (primary analysis language)
- **Jupyter Notebooks** for cohort identification
- **R** for LOESS linearity assessment
- **uv** for Python dependency management

### Key Dependencies (from pyproject.toml)
- `clifpy` >= 0.3.1 — official Python package for CLIF data (validation, SOFA scores, wide-format transforms)
- `pandas` >= 2.3.3 — data manipulation
- `statsmodels` >= 0.14.6 — logistic regression
- `scipy` >= 1.17.0 — statistical tests
- `jupyter` >= 1.1.1 — notebook execution
- `nbconvert` >= 7.17.0 — notebook conversion

### clifpy Package
- **PyPI**: `pip install clifpy`
- **Docs**: https://common-longitudinal-icu-data-format.github.io/clifpy/
- Features: CLIF 2.0 schema validation, SOFA score calculation, unit conversion, encounter stitching, wide-format dataset generation
- Uses DuckDB and Polars under the hood for performance

## Project Structure

```
.
├── CLAUDE.md                          # This file
├── NIPPV_Draft_With_Feedback.docx     # Manuscript draft
├── CLIF Outputs/                      # Site-level statistical output
│   ├── consort.csv                    # Cohort selection counts (CONSORT diagram)
│   ├── descriptive_characteristics.csv
│   ├── multivariable_logistic_results_Interaction.csv
│   ├── multivariable_logistic_results_NoInteraction.csv
│   ├── univariate_logistic_results_Interaction.csv
│   └── univariate_logistic_results_NoInteraction.csv
```

### Upstream nippv-pred Repository Structure
```
nippv-pred/
├── config_template.json               # Site config (site name, data directory, filetype, timezone)
├── pyproject.toml                     # Python dependencies
├── run_all.sh                         # Execute full pipeline via uv
├── code/
│   ├── 01_wide_generator.py           # Transform CLIF tables to wide format
│   ├── 02_study_cohort.ipynb          # Cohort identification (inclusion/exclusion)
│   ├── 03_descriptive_characteristics.py
│   ├── 04_analysis_no_interaction.py  # Logistic regression without interaction terms
│   └── 05_analysis_interaction.py     # Logistic regression with interaction terms
```

## Pipeline Execution

Each site runs the same standardized pipeline against their local CLIF data:

```bash
# 1. Configure site settings
cp config_template.json config.json
# Edit config.json with: site name, data_directory, filetype (parquet), timezone

# 2. Install dependencies and run full pipeline
uv sync
./run_all.sh
```

Pipeline steps:
1. **Wide generator** — converts longitudinal CLIF tables to wide (hourly) format using clifpy
2. **Cohort selection** — applies inclusion/exclusion criteria (NIPPV within 6h, pH <= 7.35, pCO2 >= 45, FiO2 < 0.6)
3. **Descriptive characteristics** — baseline characteristics stratified by NIPPV failure, t-tests and chi-squared tests
4. **Analysis (no interaction)** — univariate + multivariable logistic regression
5. **Analysis (with interaction)** — adds Age x pH and pCO2 x RR interaction terms

## Outcome Definition

**NIPPV Failure** = death OR escalation to invasive mechanical ventilation within 48 hours of NIPPV initiation.

## Predictor Variables

| Variable | Type | Scaling | Interpretation |
|---|---|---|---|
| Age | Continuous | (age - mean) / 10 | Per 10-year increase |
| Female | Binary | 0/1 | Female vs Male (reference) |
| pCO2 | Continuous | (pCO2 - mean) / 10 | Per 10 mmHg increase |
| pH | Continuous | (pH - mean) / 5 | Per 0.1 unit increase |
| MAP | Continuous | mmHg | Median mean arterial pressure |
| RR | Continuous | (RR - mean) / 5 | Per 5 bpm increase |
| HR | Continuous | (HR - mean) / 10 | Per 10 bpm increase |
| FiO2 | Continuous | Proportion (0-1) | Oxygen requirement |
| PEEP | Continuous | cmH2O | PEEP setting |
| Tidal Volume | Continuous | mL | Observed tidal volume |
| Age x pH | Interaction | Age Scale x pH Scale | Age-acidosis interaction |
| pCO2 x RR | Interaction | pCO2 Scale x RR Scale | Hypercapnia-RR interaction |

Physiologic predictors are extracted as the **median value 1–12 hours following NIPPV initiation** (except age, sex, and MAP which uses the median of all recordings).

## Data Privacy

- Patient-level data **never leaves** the originating institution
- Each site runs the analysis code locally against their CLIF instance
- Only aggregate results (odds ratios, confidence intervals, p-values) are shared
- Results are pooled using **fixed-effects meta-analysis**

## CLIF Data Format

- Preferred file format: **Parquet**
- Data dictionary: https://clif-icu.com/data-dictionary/data-dictionary-2.0.0
- DDL available in the main CLIF repo under `ddl/` (MySQL format)
- CLIF version: 2.1.0+

## Key References

- Rojas JC, Lyons PG, Chhikara K, et al. (2025). A common longitudinal intensive care unit data format (CLIF) for critical illness research. *Intensive Care Medicine*. https://doi.org/10.1007/s00134-025-07848-7
- Confalonieri M, et al. (2005). A chart of failure risk for noninvasive ventilation in patients with COPD exacerbation. *Eur Respir J*.
- Ko BS, et al. (2015). Early failure of noninvasive ventilation in COPD with acute hypercapnic respiratory failure. *Intern Emerg Med*.

## Related CLIF Repositories

- [CLIF](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF) — Main data standard documentation
- [clifpy](https://github.com/Common-Longitudinal-ICU-data-Format/clifpy) — Python client for CLIF
- [nippv-pred](https://github.com/Common-Longitudinal-ICU-data-Format/nippv-pred) — This project's upstream analysis code
- [CLIF-Project-Template](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF-Project-Template) — Standard project structure template
- [CLIF-MIMIC](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF-MIMIC) — MIMIC-IV to CLIF converter
- [CLIF_cohort_identifier](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF_cohort_identifier) — Shiny app for cohort discovery
- [EHR-TO-CLIF](https://github.com/Common-Longitudinal-ICU-data-Format/EHR-TO-CLIF) — Institutional ETL pathways
