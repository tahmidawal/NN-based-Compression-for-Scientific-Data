#!/bin/bash
#SBATCH --job-name=swae_arch_search
#SBATCH --output=logs/swae_arch_search_%j.out
#SBATCH --error=logs/swae_arch_search_%j.err
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

# Define latent dimensions and architecture configurations
LATENT_DIMS=(4 8 16 32 64)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_SAVE_DIR="./save/swae_architecture_search_${TIMESTAMP}"

# REDUCED HYPERPARAMETER SEARCH SPACE
# Focus on most promising combinations to reduce from 540 to ~80 configurations

# Training hyperparameters (REDUCED)
LR_VALUES=(1e-4 2e-4)  # Reduced from 3 to 2 values - focus on proven range
BATCH_SIZES=(32 64)    # Reduced from 3 to 2 values - avoid 128 for speed
LAMBDA_REG_VALUES=(0.9 1.5)  # Reduced from 3 to 2 values - focus on effective range

echo "=========================================="
echo "SWAE Architecture Search (FOCUSED)"
echo "=========================================="
echo "Testing latent dimensions: ${LATENT_DIMS[@]}"
echo "ENHANCED search space with speed variants:"
echo "  - Architectures: 4 per latent dim (baseline, shallow, deep, accurate)"
echo "  - Learning rates: ${LR_VALUES[@]} (focused range)"
echo "  - Batch sizes: ${BATCH_SIZES[@]} (optimized for speed)" 
echo "  - Lambda regularization: ${LAMBDA_REG_VALUES[@]} (effective range)"
echo "Expected total configurations: ~160 (4×2×2×2×5 dims)"
echo "Base save directory: $BASE_SAVE_DIR"
echo "Dataset: /u/tawal/0620-NN-based-compression-thera/tt_q01/"
echo "🎯 Optimizing for: Speed + Accuracy trade-offs"
echo "=========================================="

# Create base save directory
mkdir -p "$BASE_SAVE_DIR"

# Create comprehensive summary file
SUMMARY_FILE="$BASE_SAVE_DIR/architecture_search_summary.txt"
echo "SWAE Architecture Search Summary (FOCUSED)" > "$SUMMARY_FILE"
echo "Started: $(date)" >> "$SUMMARY_FILE"
echo "Search Space: Latent dims ${LATENT_DIMS[@]}" >> "$SUMMARY_FILE"
echo "Total configurations planned: ~80" >> "$SUMMARY_FILE"
echo "Reduced hyperparameter space for efficiency" >> "$SUMMARY_FILE"
echo "==========================================" >> "$SUMMARY_FILE"

# Create CSV for tracking results
RESULTS_CSV="$BASE_SAVE_DIR/architecture_search_results.csv"
echo "latent_dim,arch_config,channels,lr,batch_size,lambda_reg,compression_ratio,final_psnr,best_psnr,training_time,inference_speed,model_params,status" > "$RESULTS_CSV"

# Function to get architecture configs based on latent dimension (4 VARIANTS PER DIM)
get_arch_configs() {
    local latent_dim=$1
    # All latent dimensions get the same 4 architecture variants
    echo "baseline shallow deep accurate"
}

# Function to get channel config string
get_channels() {
    local latent_dim=$1
    local config_name=$2
    
    case $config_name in
        baseline) echo "32,64,128" ;;
        shallow) echo "24,48,96" ;;
        deep) echo "32,64,128,256" ;;
        accurate)
            if [ $latent_dim -le 16 ]; then
                echo "48,96,192,384"  # For high compression dims
            else
                echo "64,128,256,512"  # For low compression dims  
            fi
            ;;
        *) echo "32,64,128" ;;
    esac
}

# Main search loop
total_configs=0
completed_configs=0

for latent_dim in "${LATENT_DIMS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Architecture Search for latent_dim = $latent_dim"
    echo "=========================================="
    
    compression_ratio=$(python -c "print(f'{343/$latent_dim:.1f}:1')")
    echo "Compression ratio: $compression_ratio"
    
    # Get architecture configurations for this latent dimension
    arch_configs=$(get_arch_configs $latent_dim)
    
    for arch_config in $arch_configs; do
        channels=$(get_channels $latent_dim $arch_config)
        
        # Test different hyperparameter combinations for each architecture
        for lr in "${LR_VALUES[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
                for lambda_reg in "${LAMBDA_REG_VALUES[@]}"; do
                    total_configs=$((total_configs + 1))
                    
                    # Create unique identifier for this configuration
                    config_id="${latent_dim}_${arch_config}_lr${lr}_bs${batch_size}_reg${lambda_reg}"
                    config_save_dir="$BASE_SAVE_DIR/config_${config_id}"
                    mkdir -p "$config_save_dir"
                    
                    echo ""
                    echo "🔍 Testing Configuration $total_configs:"
                    echo "   Latent Dim: $latent_dim (${compression_ratio})"
                    echo "   Architecture: $arch_config"
                    echo "   Channels: [$channels]"
                    echo "   Learning Rate: $lr"
                    echo "   Batch Size: $batch_size"
                    echo "   Lambda Reg: $lambda_reg"
                    echo "   Save Dir: $config_save_dir"
                    
                    # Log configuration start
                    echo "Config $total_configs: $config_id" >> "$SUMMARY_FILE"
                    echo "  Started: $(date)" >> "$SUMMARY_FILE"
                    
                    # Record start time
                    start_time=$(date +%s)
                    
                    # Run training with current configuration
                    python train_swae_u_chi.py \
                        --data-folder /u/tawal/0620-NN-based-compression-thera/tt_q01/ \
                        --normalize \
                        --normalize-method pos_log \
                        --latent-dim $latent_dim \
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
                        echo "  Status: SUCCESS, Training time: ${training_time}s" >> "$SUMMARY_FILE"
                        
                        # Extract metrics from training log (simplified - you might want to enhance this)
                        best_psnr=$(grep -o "Best PSNR: [0-9]*\.[0-9]*" "$config_save_dir/training_log.txt" | tail -1 | grep -o "[0-9]*\.[0-9]*" || echo "N/A")
                        final_psnr=$(grep -o "Final PSNR: [0-9]*\.[0-9]*" "$config_save_dir/training_log.txt" | tail -1 | grep -o "[0-9]*\.[0-9]*" || echo "N/A")
                        
                        # Count model parameters (simplified)
                        model_params=$(grep -o "Total parameters: [0-9,]*" "$config_save_dir/training_log.txt" | tail -1 | grep -o "[0-9,]*" | tr -d ',' || echo "N/A")
                        
                        # Add to results CSV
                        echo "$latent_dim,$arch_config,$channels,$lr,$batch_size,$lambda_reg,$compression_ratio,$final_psnr,$best_psnr,${training_time},N/A,$model_params,SUCCESS" >> "$RESULTS_CSV"
                        
                    else
                        echo "❌ Configuration $total_configs failed"
                        echo "  Status: FAILED, Training time: ${training_time}s" >> "$SUMMARY_FILE"
                        
                        # Add failure to results CSV
                        echo "$latent_dim,$arch_config,$channels,$lr,$batch_size,$lambda_reg,$compression_ratio,N/A,N/A,${training_time},N/A,N/A,FAILED" >> "$RESULTS_CSV"
                    fi
                    
                    echo "Progress: $completed_configs/$total_configs configurations completed"
                    echo "----------------------------------------"
                done
            done
        done
    done
    
    echo "Completed latent dimension $latent_dim"
    echo "==========================================" >> "$SUMMARY_FILE"
done

echo ""
echo "=========================================="
echo "Architecture Search Completed"
echo "=========================================="
echo "Total configurations tested: $total_configs"
echo "Successfully completed: $completed_configs"
echo "Success rate: $(python -c "print(f'{$completed_configs/$total_configs*100:.1f}%')")" 
echo "Results saved to: $RESULTS_CSV"
echo "Summary saved to: $SUMMARY_FILE"
echo "All models saved under: $BASE_SAVE_DIR"
echo "Completed at: $(date)"

# Add final summary to files
echo "" >> "$SUMMARY_FILE"
echo "Architecture search completed: $(date)" >> "$SUMMARY_FILE"
echo "Total configurations: $total_configs" >> "$SUMMARY_FILE"
echo "Successful configurations: $completed_configs" >> "$SUMMARY_FILE"
echo "Success rate: $(python -c "print(f'{$completed_configs/$total_configs*100:.1f}%')")" >> "$SUMMARY_FILE"

# Create analysis script for results
cat > "$BASE_SAVE_DIR/analyze_results.py" << 'EOF'
#!/usr/bin/env python3
"""
Analyze Architecture Search Results
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_architecture_search(csv_path):
    # Load results
    df = pd.read_csv(csv_path)
    successful = df[df['status'] == 'SUCCESS'].copy()
    
    if len(successful) == 0:
        print("No successful configurations found!")
        return
    
    # Convert numeric columns
    successful['final_psnr'] = pd.to_numeric(successful['final_psnr'], errors='coerce')
    successful['training_time'] = pd.to_numeric(successful['training_time'], errors='coerce')
    
    print(f"Architecture Search Results Analysis")
    print(f"====================================")
    print(f"Total configurations: {len(df)}")
    print(f"Successful configurations: {len(successful)}")
    print(f"Success rate: {len(successful)/len(df)*100:.1f}%")
    
    # Find best configurations per latent dimension
    print(f"\nBest Configurations per Latent Dimension:")
    print(f"-" * 60)
    
    for latent_dim in sorted(successful['latent_dim'].unique()):
        dim_results = successful[successful['latent_dim'] == latent_dim]
        if len(dim_results) > 0:
            best_config = dim_results.loc[dim_results['final_psnr'].idxmax()]
            compression_ratio = 343 / latent_dim
            
            print(f"Latent Dim {latent_dim} ({compression_ratio:.1f}:1):")
            print(f"  Best PSNR: {best_config['final_psnr']:.2f} dB")
            print(f"  Architecture: {best_config['arch_config']}")
            print(f"  Channels: {best_config['channels']}")
            print(f"  Hyperparams: lr={best_config['lr']}, bs={best_config['batch_size']}, reg={best_config['lambda_reg']}")
            print(f"  Training time: {best_config['training_time']:.0f}s")
            print()
    
    # Save summary to file
    with open(csv_path.replace('.csv', '_summary.txt'), 'w') as f:
        f.write("Architecture Search Results Summary\n")
        f.write("==================================\n\n")
        f.write(f"Total configurations: {len(df)}\n")
        f.write(f"Successful configurations: {len(successful)}\n")
        f.write(f"Success rate: {len(successful)/len(df)*100:.1f}%\n\n")
        
        f.write("Best Configurations per Latent Dimension:\n")
        f.write("-" * 50 + "\n")
        
        for latent_dim in sorted(successful['latent_dim'].unique()):
            dim_results = successful[successful['latent_dim'] == latent_dim]
            if len(dim_results) > 0:
                best_config = dim_results.loc[dim_results['final_psnr'].idxmax()]
                compression_ratio = 343 / latent_dim
                
                f.write(f"\nLatent Dim {latent_dim} ({compression_ratio:.1f}:1):\n")
                f.write(f"  Best PSNR: {best_config['final_psnr']:.2f} dB\n")
                f.write(f"  Architecture: {best_config['arch_config']}\n")
                f.write(f"  Channels: {best_config['channels']}\n")
                f.write(f"  Hyperparams: lr={best_config['lr']}, bs={best_config['batch_size']}, reg={best_config['lambda_reg']}\n")
                f.write(f"  Training time: {best_config['training_time']:.0f}s\n")

if __name__ == "__main__":
    analyze_architecture_search('architecture_search_results.csv')
EOF

chmod +x "$BASE_SAVE_DIR/analyze_results.py"

echo ""
echo "🎯 Next Steps:"
echo "1. Analyze results: cd $BASE_SAVE_DIR && python analyze_results.py"
echo "2. Run inference on best configurations"
echo "3. Select optimal architectures for production"
echo ""
echo "🏁 Architecture search completed!" 