#!/bin/bash
#SBATCH --job-name=bssn_ae_inference
#SBATCH --account=bcqs-delta-gpu
#SBATCH --output=logs/bssn_ae_inference_%j.out
#SBATCH --error=logs/bssn_ae_inference_%j.err
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

# Load conda module - try common names
module load anaconda3_cpu 2>/dev/null || \
module load miniconda3 2>/dev/null || \
echo "No conda module found, assuming conda is in PATH"

# Activate conda environment
source activate liif_env

# Verify environment
echo "Conda environment: $CONDA_DEFAULT_ENV"
which python

# Run the inference script
python run_inference.py

echo "Inference script finished."
