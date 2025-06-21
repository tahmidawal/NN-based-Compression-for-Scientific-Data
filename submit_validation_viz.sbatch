#!/bin/bash
#SBATCH --job-name=bssn_val_viz
#SBATCH --account=bcqs-delta-gpu         # Account from training script
#SBATCH --partition=gpuA100x4          # GPU partition from training script
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8            # CPUs from training script
#SBATCH --mem=64G                    # Memory from training script
#SBATCH --gpus-per-node=1            # GPUs from training script
#SBATCH --time=00:30:00              # Keep time reasonable for visualization
#SBATCH --output=logs/bssn_val_viz_%j.out # Specific log for this job
#SBATCH --error=logs/bssn_val_viz_%j.err  # Specific error log for this job

# Create logs directory if it doesn't exist
MAIN_DIR=$(pwd)
mkdir -p "${MAIN_DIR}/logs"

# Load conda module - try common names (from training script)
module load anaconda3_cpu 2>/dev/null || \
module load miniconda3 2>/dev/null || \
module load conda 2>/dev/null || \
{
    echo "No conda module found, attempting to use system conda."
    # Attempt to find conda in PATH if module load fails
    if ! command -v conda &> /dev/null; then
        echo "Conda command not found. Please ensure Conda is in your PATH or a module is available."
        exit 1
    fi
}

# Initialize conda for bash shell (from training script)
CONDA_BASE=$(conda info --base)
if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    echo "Conda initialization script not found at ${CONDA_BASE}/etc/profile.d/conda.sh"
    # As a fallback, try eval if the above source fails (less robust)
    eval "$(conda shell.bash hook)" 2>/dev/null || {
        echo "Failed to initialize conda shell. Please check your conda installation."
        exit 1
    }
fi

# Activate the LIIF environment
conda activate liif_env
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment 'liif_env'."
    echo "Please ensure the environment exists and is correctly configured."
    conda env list # List available environments for debugging
    exit 1
fi
echo "Conda environment activated: $(conda env list | grep \*)"

# Navigate to the script directory
# Ensure this path is correct
cd "/u/tawal/NN-based-Compression-for-Scientific-Data/" || { echo "Failed to cd into script directory"; exit 1; }

# Run the Python script
echo "Running visualize_validation_reconstructions.py..."
python visualize_validation_reconstructions.py
if [ $? -ne 0 ]; then
    echo "Python script visualize_validation_reconstructions.py failed"
    exit 1
fi

echo "Validation visualization script finished successfully."

