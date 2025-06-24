#!/bin/bash
#SBATCH --job-name=swae_u_chi_7x7x7
#SBATCH --output=logs/swae_u_chi_%j.out
#SBATCH --error=logs/swae_u_chi_%j.err
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcqs-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

# Create logs directory
mkdir -p logs

# Use current Python environment (pyenv)

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "Working Directory: $(pwd)"

# Print GPU information
nvidia-smi

echo "Starting SWAE training on U_CHI dataset (7x7x7 blocks)"
echo "Dataset: /u/tawal/0620-NN-based-compression-thera/tt_q01/"
echo "Total samples: ~29K (23K train, 6K val)"
echo "Compression ratio: 21.4:1 (343 -> 16 dimensions)"

# Run U_CHI SWAE training with FIXED parameters (lambda_reg=1.0, lr=2e-4)
python train_swae_u_chi.py \
    --data-folder /u/tawal/0620-NN-based-compression-thera/tt_q01/ \
    --normalize \
    --normalize-method minmax \
    --latent-dim 32 \
    --lambda-reg 0.9 \
    --batch-size 128 \
    --epochs 1000 \
    --lr 2e-4 \
    --train-split 0.8 \
    --num-workers 8 \
    --device auto \
    --save-dir ./save/swae_u_chi_fixed_$(date +%Y%m%d_%H%M%S) \
    --eval-interval 10 \
    --save-interval 25

echo "Training completed at $(date)"
echo "Check results in save directory" 