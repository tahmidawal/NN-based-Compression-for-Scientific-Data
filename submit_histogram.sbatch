#!/bin/bash
#SBATCH --job-name=uchi_hist
#SBATCH --output=logs/uchi_histogram_%j.out
#SBATCH --error=logs/uchi_histogram_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00 # 10 minutes should be sufficient
#SBATCH --account=bcqs-delta-cpu
#SBATCH --partition=cpu

# Load necessary modules
module purge
module load anaconda3/2023.09

# Activate Conda environment
# IMPORTANT: Replace 'liif_env' with your actual Conda environment name if different
CONDA_ENV_NAME="liif_env"

# Find Conda environment path
CONDA_BASE_PATH=$(conda info --base)
CONDA_ENV_PATH=$(conda env list | grep "^${CONDA_ENV_NAME} " | awk '{print $NF}')

if [ -z "${CONDA_ENV_PATH}" ] || [ ! -d "${CONDA_ENV_PATH}" ]; then
    echo "Error: Conda environment '${CONDA_ENV_NAME}' not found or path is incorrect."
    echo "Please check the CONDA_ENV_NAME variable in this script."
    echo "Found path: '${CONDA_ENV_PATH}'"
    conda env list
    exit 1
fi

echo "Activating Conda environment: ${CONDA_ENV_NAME} at ${CONDA_ENV_PATH}"

source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PATH}"

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate Conda environment '${CONDA_ENV_NAME}'."
    exit 1
fi

echo "Conda environment activated."
python --version

# Navigate to the script directory
cd "$SLURM_SUBMIT_DIR"

# Run the Python script
echo "Running plot_uchi_histogram.py..."
python plot_uchi_histogram.py

if [ $? -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "Python script failed. Check error logs."
fi

conda deactivate
echo "Job finished."
