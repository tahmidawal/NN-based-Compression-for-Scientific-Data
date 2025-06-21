#!/bin/bash
#SBATCH --job-name=install_pyevtk
#SBATCH --output=logs/install_pyevtk_%j.out
#SBATCH --error=logs/install_pyevtk_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --account=bcqs-delta-cpu
#SBATCH --partition=cpu

# Load necessary modules
module purge
module load anaconda3

# Activate Conda environment
CONDA_ENV_NAME="liif_env"
CONDA_BASE_PATH=$(conda info --base)
source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

# Print Python and conda info
echo "Python version:"
python --version
echo "Conda environment:"
conda info

# Install pyevtk
echo "Installing pyevtk..."
pip install pyevtk

# Verify installation
echo "Verifying installation..."
pip list | grep evtk

echo "Installation complete."
