#!/bin/bash
#SBATCH --job-name=bssn_viz_val_uchi
#SBATCH --account=bcqs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/bssn_viz_val_uchi_%j.out
#SBATCH --error=logs/bssn_viz_val_uchi_%j.err

# Create logs directory if it doesn't exist
MAIN_DIR=$(pwd)
mkdir -p "${MAIN_DIR}/logs"

# Load conda module
module load anaconda3_cpu 2>/dev/null || \
module load miniconda3 2>/dev/null || \
module load conda 2>/dev/null || \
{
    echo "No conda module found, attempting to use system conda."
    if ! command -v conda &> /dev/null; then
        echo "Conda command not found. Please ensure Conda is in your PATH or a module is available."
        exit 1
    fi
}

# Initialize conda for bash shell
CONDA_BASE=$(conda info --base)
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)" 2>/dev/null || {
        echo "Failed to initialize conda shell. Please check your conda installation."
        exit 1
    }
fi

# Activate the LIIF environment
conda activate liif_env
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment 'liif_env'."
    exit 1
fi
echo "Conda environment activated: $(conda env list | grep \*)"

# Navigate to the script directory
cd "/u/tawal/NN-based-Compression-for-Scientific-Data/" || { echo "Failed to cd into script directory"; exit 1; }

# Run the Python validation visualization script
# visualize_validation_reconstructions.py is now configured to run for U_CHI by default (MODEL_TYPE='U_CHI')
echo "Running visualize_validation_reconstructions.py for U_CHI model..."
python visualize_validation_reconstructions.py
if [ $? -ne 0 ]; then
    echo "Python script visualize_validation_reconstructions.py (for U_CHI) failed"
    exit 1
fi

echo "U_CHI validation visualization script finished successfully."
