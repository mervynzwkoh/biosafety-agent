#!/bin/bash
# ==============================================================================
# Run Biosafety Defense Agent Batch Evaluation Job
# ==============================================================================
#SBATCH --job-name=biosafety-eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_%j.log

set -e

echo "Starting Biosafety Defense Agent evaluation pipeline..."
mkdir -p logs

python -m pytest tests/ -v --junitxml=logs/test_report.xml

echo "Evaluation job completed successfully."
