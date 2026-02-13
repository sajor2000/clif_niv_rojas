#!/usr/bin/env bash
# run_pooling.sh — Execute the full lead-site meta-analysis pipeline
#
# Usage: ./lead_site_analysis/run_pooling.sh
#
# Prerequisites:
#   1. All site CSVs placed in all_site_results/<site_name>/
#   2. Dependencies installed (uv sync)
#
# This script is run ONLY by the lead site (Rush).
# Participating sites do NOT run this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "NIPPV-Pred Meta-Analysis Pipeline"
echo "============================================"
echo "Root:   $ROOT_DIR"
echo "Sites:  $ROOT_DIR/all_site_results/"
echo ""

# Check that site directories exist
SITE_COUNT=$(find "$ROOT_DIR/all_site_results" -mindepth 1 -maxdepth 1 -type d ! -name '.*' | wc -l | tr -d ' ')
if [ "$SITE_COUNT" -eq 0 ]; then
    echo "ERROR: No site directories found in all_site_results/"
    echo "       Each site should upload results to all_site_results/<site_name>/"
    exit 1
fi
echo "Found $SITE_COUNT site(s)"
echo ""

# Step 1: Pool results
echo "--- Step 1/3: Pooling results ---"
uv run python "$SCRIPT_DIR/01_pool_results.py"
echo ""

# Step 2: Generate figures
echo "--- Step 2/3: Generating figures ---"
uv run python "$SCRIPT_DIR/02_figures.py"
echo ""

# Step 3: Generate tables
echo "--- Step 3/3: Generating tables ---"
uv run python "$SCRIPT_DIR/03_tables.py"
echo ""

echo "============================================"
echo "Meta-analysis pipeline completed successfully"
echo "============================================"
echo "Pooled results:  $SCRIPT_DIR/output/"
echo "Figures:         $SCRIPT_DIR/figures/"
echo "Tables:          $SCRIPT_DIR/tables/"
