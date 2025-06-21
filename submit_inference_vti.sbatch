#!/bin/bash
#SBATCH --job-name=bssn_ae_vti
#SBATCH --output=logs/bssn_ae_inference_vti_%j.out
#SBATCH --error=logs/bssn_ae_inference_vti_%j.err
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

# Run the inference script
echo "Running inference script with VTI export..."
python run_inference_with_vti.py

echo "Job finished."
