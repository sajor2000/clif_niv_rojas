#!/usr/bin/env python3
"""
run_all.py -- Cross-platform pipeline runner for NIPPV-Pred.

Works on macOS, Linux, and Windows (no bash required).
Equivalent to run_all.sh but uses Python subprocess calls.

Usage:
    uv run python run_all.py
    # or
    python run_all.py
"""

import subprocess
import sys
import os
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(SCRIPT_DIR, "code")

# Log file with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(SCRIPT_DIR, f"pipeline_{timestamp}.log")


def run_step(step_num, total, description, cmd):
    """Run a pipeline step, printing output and logging to file."""
    header = f"[Step {step_num}/{total}] {description}..."
    print(header)
    with open(log_file, "a") as lf:
        lf.write(header + "\n")

    result = subprocess.run(
        cmd,
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.stdout:
        print(result.stdout, end="")
        with open(log_file, "a") as lf:
            lf.write(result.stdout)

    if result.returncode != 0:
        msg = f"FAILED at step {step_num}: {description} (exit code {result.returncode})"
        print(msg)
        with open(log_file, "a") as lf:
            lf.write(msg + "\n")
        sys.exit(result.returncode)


def main():
    print("=== NIPPV-Pred Pipeline ===")
    print(f"Started: {datetime.datetime.now()}")
    print(f"Log: {log_file}")
    print()

    with open(log_file, "w") as lf:
        lf.write(f"=== NIPPV-Pred Pipeline ===\n")
        lf.write(f"Started: {datetime.datetime.now()}\n\n")

    # Detect whether to use 'uv run python' or plain 'python'
    uv = "uv"
    python = [uv, "run", "python"]

    # Step 0: Install dependencies
    run_step(0, 5, "Installing dependencies",
             [uv, "sync"])

    # Step 1: Wide dataset generation
    run_step(1, 5, "Generating wide dataset",
             python + [os.path.join("code", "01_wide_generator.py")])

    # Step 2: Study cohort (notebook execution)
    run_step(2, 5, "Identifying study cohort",
             [uv, "run", "jupyter", "nbconvert",
              "--to", "notebook", "--execute",
              os.path.join("code", "02_study_cohort.ipynb")])

    # Step 3: Descriptive characteristics
    run_step(3, 5, "Computing descriptive characteristics",
             python + [os.path.join("code", "03_descriptive_characteristics.py")])

    # Step 4: Logistic regression (no interaction)
    run_step(4, 5, "Running logistic regression (no interaction)",
             python + [os.path.join("code", "04_analysis_no_interaction.py")])

    # Step 5: Logistic regression (with interaction)
    run_step(5, 5, "Running logistic regression (with interaction)",
             python + [os.path.join("code", "05_analysis_interaction.py")])

    # Done
    print()
    print("=== Pipeline completed successfully ===")
    print(f"Finished: {datetime.datetime.now()}")
    print(f"Results in: output_to_share/")
    print(f"Log saved to: {log_file}")

    with open(log_file, "a") as lf:
        lf.write(f"\n=== Pipeline completed successfully ===\n")
        lf.write(f"Finished: {datetime.datetime.now()}\n")
        lf.write(f"Results in: output_to_share/\n")


if __name__ == "__main__":
    main()
