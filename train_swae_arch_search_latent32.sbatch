#!/bin/bash
#SBATCH --job-name=swae_arch_latent32
#SBATCH --output=logs/swae_arch_latent32_%j.out
#SBATCH --error=logs/swae_arch_latent32_%j.err
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcqs-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

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

# Configuration for this job
LATENT_DIM=32
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_SAVE_DIR="./save/swae_architecture_search_parallel_${TIMESTAMP}"

# Architecture configurations for latent dim 32 (quality focused)
ARCH_CONFIGS=("baseline" "shallow" "deep" "accurate")
CHANNELS_BASELINE="32,64,128"
CHANNELS_SHALLOW="24,48,96"
CHANNELS_DEEP="32,64,128,256"
CHANNELS_ACCURATE="64,128,256,512"

# Hyperparameters to search
LR_VALUES=(1e-4 2e-4)
BATCH_SIZES=(32 64)
LAMBDA_REG_VALUES=(0.9 1.5)

echo "=========================================="
echo "SWAE Architecture Search - Latent Dim $LATENT_DIM"
echo "=========================================="
compression_ratio=$(python -c "print(f'{343/$LATENT_DIM:.1f}:1')")
echo "Compression ratio: $compression_ratio"
echo "Architectures: ${ARCH_CONFIGS[@]}"
echo "Learning rates: ${LR_VALUES[@]}"
echo "Batch sizes: ${BATCH_SIZES[@]}"
echo "Lambda reg: ${LAMBDA_REG_VALUES[@]}"
echo "Expected configurations: $((${#ARCH_CONFIGS[@]} * ${#LR_VALUES[@]} * ${#BATCH_SIZES[@]} * ${#LAMBDA_REG_VALUES[@]}))"
echo "Base save directory: $BASE_SAVE_DIR"
echo "Dataset: /u/tawal/0620-NN-based-compression-thera/tt_q01/"
echo "=========================================="

# Create base save directory
mkdir -p "$BASE_SAVE_DIR"

# Create CSV for tracking results
RESULTS_CSV="$BASE_SAVE_DIR/latent_${LATENT_DIM}_results.csv"
echo "latent_dim,arch_config,channels,lr,batch_size,lambda_reg,compression_ratio,final_psnr,best_psnr,training_time,status" > "$RESULTS_CSV"

# Function to get channels for architecture
get_channels() {
    local arch_config=$1
    case $arch_config in
        baseline) echo "$CHANNELS_BASELINE" ;;
        shallow) echo "$CHANNELS_SHALLOW" ;;
        deep) echo "$CHANNELS_DEEP" ;;
        accurate) echo "$CHANNELS_ACCURATE" ;;
        *) echo "$CHANNELS_BASELINE" ;;
    esac
}

# Main search loop
total_configs=0
completed_configs=0

for arch_config in "${ARCH_CONFIGS[@]}"; do
    channels=$(get_channels $arch_config)
    
    for lr in "${LR_VALUES[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do
            for lambda_reg in "${LAMBDA_REG_VALUES[@]}"; do
                total_configs=$((total_configs + 1))
                
                # Create unique identifier for this configuration
                config_id="${LATENT_DIM}_${arch_config}_lr${lr}_bs${batch_size}_reg${lambda_reg}"
                config_save_dir="$BASE_SAVE_DIR/config_${config_id}"
                mkdir -p "$config_save_dir"
                
                echo ""
                echo "🔍 Testing Configuration $total_configs:"
                echo "   Latent Dim: $LATENT_DIM (${compression_ratio})"
                echo "   Architecture: $arch_config"
                echo "   Channels: [$channels]"
                echo "   Learning Rate: $lr"
                echo "   Batch Size: $batch_size"
                echo "   Lambda Reg: $lambda_reg"
                echo "   Save Dir: $config_save_dir"
                
                # Record start time
                start_time=$(date +%s)
                
                # Run training with current configuration
                python train_swae_u_chi.py \
                    --data-folder /u/tawal/0620-NN-based-compression-thera/tt_q01/ \
                    --normalize \
                    --normalize-method pos_log \
                    --latent-dim $LATENT_DIM \
                    --encoder-channels "$channels" \
                    --lambda-reg $lambda_reg \
                    --batch-size $batch_size \
                    --epochs 1000 \
                    --lr $lr \
                    --train-split 0.8 \
                    --num-workers 8 \
                    --device auto \
                    --save-dir "$config_save_dir" \
                    --eval-interval 10 \
                    --save-interval 1000 \
                    --early-stopping-patience 40 \
                    2>&1 | tee "$config_save_dir/training_log.txt"
                
                # Calculate training time
                end_time=$(date +%s)
                training_time=$((end_time - start_time))
                
                # Check if training completed successfully
                if [ $? -eq 0 ]; then
                    completed_configs=$((completed_configs + 1))
                    echo "✅ Configuration $total_configs completed successfully"
                    
                    # Extract metrics from training log
                    best_psnr=$(grep -o "Best PSNR: [0-9]*\.[0-9]*" "$config_save_dir/training_log.txt" | tail -1 | grep -o "[0-9]*\.[0-9]*" || echo "N/A")
                    final_psnr=$(grep -o "Final PSNR: [0-9]*\.[0-9]*" "$config_save_dir/training_log.txt" | tail -1 | grep -o "[0-9]*\.[0-9]*" || echo "N/A")
                    
                    # Add to results CSV
                    echo "$LATENT_DIM,$arch_config,$channels,$lr,$batch_size,$lambda_reg,$compression_ratio,$final_psnr,$best_psnr,${training_time},SUCCESS" >> "$RESULTS_CSV"
                    
                else
                    echo "❌ Configuration $total_configs failed"
                    
                    # Add failure to results CSV
                    echo "$LATENT_DIM,$arch_config,$channels,$lr,$batch_size,$lambda_reg,$compression_ratio,N/A,N/A,${training_time},FAILED" >> "$RESULTS_CSV"
                fi
                
                echo "Progress: $completed_configs/$total_configs configurations completed for latent dim $LATENT_DIM"
                echo "----------------------------------------"
            done
        done
    done
done

echo ""
echo "=========================================="
echo "Latent Dim $LATENT_DIM Search Completed"
echo "=========================================="
echo "Total configurations tested: $total_configs"
echo "Successfully completed: $completed_configs"
echo "Success rate: $(python -c "print(f'{$completed_configs/$total_configs*100:.1f}%')")" 
echo "Results saved to: $RESULTS_CSV"
echo "All models saved under: $BASE_SAVE_DIR"
echo "Completed at: $(date)"

# Create summary for this latent dimension
echo "Latent Dimension $LATENT_DIM Search Summary" > "$BASE_SAVE_DIR/latent_${LATENT_DIM}_summary.txt"
echo "==========================================" >> "$BASE_SAVE_DIR/latent_${LATENT_DIM}_summary.txt"
echo "Completed: $(date)" >> "$BASE_SAVE_DIR/latent_${LATENT_DIM}_summary.txt"
echo "Compression ratio: $compression_ratio" >> "$BASE_SAVE_DIR/latent_${LATENT_DIM}_summary.txt"
echo "Total configurations: $total_configs" >> "$BASE_SAVE_DIR/latent_${LATENT_DIM}_summary.txt"
echo "Successful configurations: $completed_configs" >> "$BASE_SAVE_DIR/latent_${LATENT_DIM}_summary.txt"
echo "Success rate: $(python -c "print(f'{$completed_configs/$total_configs*100:.1f}%')")" >> "$BASE_SAVE_DIR/latent_${LATENT_DIM}_summary.txt"

echo "🎯 Latent dimension $LATENT_DIM search completed successfully!" 