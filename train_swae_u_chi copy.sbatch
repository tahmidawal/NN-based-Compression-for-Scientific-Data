#!/bin/bash
#SBATCH --job-name=swae_u_chi_multi_latent
#SBATCH --output=logs/swae_u_chi_multi_%j.out
#SBATCH --error=logs/swae_u_chi_multi_%j.err
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcqs-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Create logs directory
mkdir -p logs

# Load conda module if available
module load anaconda3_cpu 2>/dev/null || \
module load miniconda3 2>/dev/null || \
module load conda 2>/dev/null || \
echo "No conda module found, using system conda..."

# Initialize conda for bash
eval "$(conda shell.bash hook)" 2>/dev/null || true

# Activate the LIIF environment
echo "Activating conda environment: liif_env"
conda activate liif_env

# Verify environment is working
echo "Verifying environment..."
python -c "import torch, numpy, matplotlib; print('Environment check passed!')"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

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

# Define latent dimensions to test
# Range from high compression (4) to low compression (64)
LATENT_DIMS=(4 8 16 32 64)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_SAVE_DIR="./save/swae_u_chi_multi_latent_${TIMESTAMP}"

echo "=========================================="
echo "Multi-Latent Dimension SWAE Training"
echo "=========================================="
echo "Testing latent dimensions: ${LATENT_DIMS[@]}"
echo "Compression ratios:"
for latent_dim in "${LATENT_DIMS[@]}"; do
    compression_ratio=$(python -c "print(f'{343/$latent_dim:.1f}:1')")
    echo "  latent_dim=$latent_dim -> compression ratio=$compression_ratio"
done
echo "Base save directory: $BASE_SAVE_DIR"
echo "Dataset: /u/tawal/0620-NN-based-compression-thera/tt_q01/"
echo "Data splits: 80% train (~23K), 15% val (~4K), 5% test (~1.5K)"
echo "🔒 Test set (5%) held out for final unbiased evaluation"
echo "=========================================="

# Create base save directory
mkdir -p "$BASE_SAVE_DIR"

# Create summary file to track training progress
SUMMARY_FILE="$BASE_SAVE_DIR/training_summary.txt"
echo "Multi-Latent Dimension SWAE Training Summary" > "$SUMMARY_FILE"
echo "Started: $(date)" >> "$SUMMARY_FILE"
echo "Latent dimensions tested: ${LATENT_DIMS[@]}" >> "$SUMMARY_FILE"
echo "==========================================" >> "$SUMMARY_FILE"

# Loop through each latent dimension
for latent_dim in "${LATENT_DIMS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Training with latent_dim = $latent_dim"
    echo "=========================================="
    
    # Calculate compression ratio
    compression_ratio=$(python -c "print(f'{343/$latent_dim:.1f}:1')")
    echo "Compression ratio: $compression_ratio"
    
    # Create specific save directory for this latent dimension
    LATENT_SAVE_DIR="$BASE_SAVE_DIR/latent_dim_${latent_dim}"
    mkdir -p "$LATENT_SAVE_DIR"
    
    # Log training start for this latent dimension
    echo "" >> "$SUMMARY_FILE"
    echo "Latent dim $latent_dim (compression ${compression_ratio}):" >> "$SUMMARY_FILE"
    echo "  Started: $(date)" >> "$SUMMARY_FILE"
    echo "  Save dir: $LATENT_SAVE_DIR" >> "$SUMMARY_FILE"
    
    # Run training for this latent dimension
    echo "Starting training..."
    python train_swae_u_chi.py \
        --data-folder /u/tawal/0620-NN-based-compression-thera/tt_q01/ \
        --normalize \
        --normalize-method pos_log \
        --latent-dim $latent_dim \
        --lambda-reg 0.9 \
        --batch-size 64 \
        --epochs 500 \
        --lr 1e-4 \
        --train-split 0.8 \
        --num-workers 8 \
        --device auto \
        --save-dir "$LATENT_SAVE_DIR" \
        --eval-interval 10 \
        --save-interval 500 \
        2>&1 | tee "$LATENT_SAVE_DIR/training_log.txt"
    
    # Check if training completed successfully
    if [ $? -eq 0 ]; then
        echo "✅ Training completed successfully for latent_dim = $latent_dim"
        echo "  Completed: $(date)" >> "$SUMMARY_FILE"
        echo "  Status: SUCCESS" >> "$SUMMARY_FILE"
        
        # List saved models
        echo "  Saved models:" >> "$SUMMARY_FILE"
        ls -la "$LATENT_SAVE_DIR"/*.pth >> "$SUMMARY_FILE" 2>/dev/null || echo "    No .pth files found" >> "$SUMMARY_FILE"
    else
        echo "❌ Training failed for latent_dim = $latent_dim"
        echo "  Completed: $(date)" >> "$SUMMARY_FILE"
        echo "  Status: FAILED" >> "$SUMMARY_FILE"
    fi
    
    echo "Model saved to: $LATENT_SAVE_DIR"
    echo "Training log: $LATENT_SAVE_DIR/training_log.txt"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "All Training Jobs Completed"
echo "=========================================="
echo "Summary saved to: $SUMMARY_FILE"
echo "All models saved under: $BASE_SAVE_DIR"
echo "Completed at: $(date)"

# Add completion info to summary
echo "" >> "$SUMMARY_FILE"
echo "All training completed: $(date)" >> "$SUMMARY_FILE"

# List all saved models
echo "" >> "$SUMMARY_FILE"
echo "Final model directory structure:" >> "$SUMMARY_FILE"
find "$BASE_SAVE_DIR" -name "*.pth" -type f >> "$SUMMARY_FILE"

echo "🎯 Next step: Run inference script on all trained models"
echo "🎯 Base directory for inference: $BASE_SAVE_DIR" 