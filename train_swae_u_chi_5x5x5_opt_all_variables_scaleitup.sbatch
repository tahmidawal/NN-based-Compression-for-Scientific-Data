#!/bin/bash
#SBATCH --job-name=swae_all_vars_5x5x5_opt_scaleitup
#SBATCH --output=logs/swae_all_vars_5x5x5_opt_scaleitup_%j.out
#SBATCH --error=logs/swae_all_vars_5x5x5_opt_scaleitup_%j.err
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcqs-delta-gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00


# Create logs directory if it doesn't exist
mkdir -p logs

# Print GPU info
nvidia-smi

echo "Starting SWAE 3D All Variables training (5x5x5 OPTIMIZED version with SCALEITUP)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

# Set scale target
SCALE_TARGET=${SCALE_TARGET:-2.5e-3}
echo "Scale target: $SCALE_TARGET"

# Run training
python train_swae_u_chi_5x5x5_opt_all_variables.py \
    --data-folder "/u/tawal/BSSN-Extracted-Data/tt_q01/" \
    --epochs 210 \
    --batch-size 128 \
    --lr 0.001 \
    --train-split 0.8 \
    --normalize \
    --normalize-method pos_log \
    --scaleitup \
    --scale-target $SCALE_TARGET \
    --latent-dim 8 \
    --lambda-reg 0.5 \
    --arch mlp \
    --save-dir "./save/swae_all_vars_5x5x5_opt_mlp_scaleitup" \
    --save-interval 20 \
    --eval-interval 10

echo "Training completed!"