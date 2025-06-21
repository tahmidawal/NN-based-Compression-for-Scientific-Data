#!/bin/bash
#SBATCH --job-name=bssn_specific
#SBATCH --output=logs/bssn_specific_samples_%j.out
#SBATCH --error=logs/bssn_specific_samples_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account=bcqs-delta-gpu
#SBATCH --partition=gpuA100x4

# Load necessary modules
module purge
module load anaconda3

# Activate Conda environment
CONDA_ENV_NAME="liif_env"
CONDA_BASE_PATH=$(conda info --base)
source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
echo "Activating Conda environment: ${CONDA_ENV_NAME} at ${CONDA_BASE_PATH}"
conda activate "${CONDA_ENV_NAME}"
echo "Conda environment activated."

# Print Python version
python --version

# Create output directory
mkdir -p visualizations/vti_files/specific_samples

# Run the inference script for specific samples
echo "Running inference script for specific samples (2706, 1446, 42)..."
python run_inference_specific_samples.py

echo "Job finished."
