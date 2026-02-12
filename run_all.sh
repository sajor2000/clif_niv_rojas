#!/bin/bash
set -euo pipefail

LOG_FILE="pipeline_$(date +%Y%m%d_%H%M%S).log"

echo "=== NIPPV-Pred Pipeline ===" | tee "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "[Step 0/5] Installing dependencies..." | tee -a "$LOG_FILE"
uv sync 2>&1 | tee -a "$LOG_FILE"

echo "[Step 1/5] Generating wide dataset..." | tee -a "$LOG_FILE"
uv run python code/01_wide_generator.py 2>&1 | tee -a "$LOG_FILE"

echo "[Step 2/5] Identifying study cohort..." | tee -a "$LOG_FILE"
uv run jupyter nbconvert --to notebook --execute code/02_study_cohort.ipynb 2>&1 | tee -a "$LOG_FILE"

echo "[Step 3/5] Computing descriptive characteristics..." | tee -a "$LOG_FILE"
uv run python code/03_descriptive_characteristics.py 2>&1 | tee -a "$LOG_FILE"

echo "[Step 4/5] Running logistic regression (no interaction)..." | tee -a "$LOG_FILE"
uv run python code/04_analysis_no_interaction.py 2>&1 | tee -a "$LOG_FILE"

echo "[Step 5/5] Running logistic regression (with interaction)..." | tee -a "$LOG_FILE"
uv run python code/05_analysis_interaction.py 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Pipeline completed successfully ===" | tee -a "$LOG_FILE"
echo "Finished: $(date)" | tee -a "$LOG_FILE"
echo "Results in: output_to_share/" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE"
