#!/bin/bash
#SBATCH --job-name=bssn_ae_train
#SBATCH --account=bcqs-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8 # Increased CPUs for data loading/processing
#SBATCH --mem=64G # Increased memory for larger dataset
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00 # Increased time for more epochs
#SBATCH --output=logs/bssn_ae_train_%j.out
#SBATCH --error=logs/bssn_ae_train_%j.err

# Create logs directory if it doesn't exist
MAIN_DIR=$(pwd)
mkdir -p "${MAIN_DIR}/logs"

# Load conda module - try common names
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

# Initialize conda for bash shell
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

# Verify environment is working by checking for python and tensorflow
echo "Verifying environment and TensorFlow installation..."
python -c "import tensorflow as tf; print(f'Python and TensorFlow check passed. TensorFlow version: {tf.__version__}')"
if [ $? -ne 0 ]; then
    echo "TensorFlow check failed. Ensure TensorFlow is installed in 'liif_env'."
    exit 1
fi

# Set CUDA device (already in your template, good practice)
export CUDA_VISIBLE_DEVICES=0 
# (SLURM usually handles GPU allocation, but this can be a safeguard or for non-SLURM contexts)

# Navigate to the directory containing the script, if necessary
# cd /path/to/your/script/directory # Uncomment and set if your script is not in pwd

# Run the training script
echo "Starting BSSN Autoencoder training script..."
python train_autoencoder.py

echo "Training script finished."
