#!/bin/bash
#SBATCH --job-name=swae_u_chi_inference
#SBATCH --output=logs/swae_u_chi_inference_%j.out
#SBATCH --error=logs/swae_u_chi_inference_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcqs-delta-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Print GPU information
nvidia-smi

# Print Python and PyTorch information
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Run the inference script
echo "Starting SWAE U_CHI validation inference with VTI output and visualization..."

# Use the latest FIXED model checkpoint with corrected lambda_reg
latest_model="save/swae_u_chi_fixed_20250623_125435/best_model.pth"

if [ ! -f "$latest_model" ]; then
    echo "Error: Fixed model checkpoint not found: $latest_model"
    exit 1
fi

echo "Using FIXED model checkpoint: $latest_model"
echo "This model was trained with lambda_reg=1.0 (vs. previous lambda_reg=10.0)"

# Run inference with enhanced visualization
python inference_swae_u_chi_validation.py \
    --model-path "$latest_model" \
    --data-folder /u/tawal/0620-NN-based-compression-thera/tt_q01/ \
    --output-dir validation_u_chi_results_$(date +%Y%m%d_%H%M%S) \
    --num-samples 50 \
    --num-vti-samples 5 \
    --batch-size 32 \
    --device auto

echo "Job completed at: $(date)"
echo ""
echo "Results include:"
echo "  - HDF5 files for all 50 samples"  
echo "  - VTI files for 5 random samples"
echo "  - Comparison slice plots (6 slices x 3 views per sample)"
echo "  - Average metrics summary" 