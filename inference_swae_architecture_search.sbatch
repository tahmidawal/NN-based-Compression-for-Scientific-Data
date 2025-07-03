#!/bin/bash
#SBATCH --job-name=swae_arch_inference
#SBATCH --output=logs/swae_arch_inference_%j.out
#SBATCH --error=logs/swae_arch_inference_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcqs-delta-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Create logs directory if it doesn't exist
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

# Function to extract architecture info from config path
get_arch_info() {
    local config_path=$1
    # Extract info from path like: config_4_baseline_lr1e-4_bs64_reg0.9
    local config_name=$(basename "$config_path")
    
    # Parse components
    local latent_dim=$(echo "$config_name" | sed 's/config_\([0-9]*\)_.*/\1/')
    local arch_config=$(echo "$config_name" | sed 's/config_[0-9]*_\([^_]*\)_.*/\1/')
    local lr=$(echo "$config_name" | sed 's/.*_lr\([^_]*\)_.*/\1/')
    local batch_size=$(echo "$config_name" | sed 's/.*_bs\([^_]*\)_.*/\1/')
    local lambda_reg=$(echo "$config_name" | sed 's/.*_reg\([^_]*\)$/\1/')
    
    echo "$latent_dim,$arch_config,$lr,$batch_size,$lambda_reg"
}

# Main inference script
if [ $# -eq 0 ]; then
    echo "Usage: $0 <architecture_search_results_directory>"
    echo "Example: $0 ./save/swae_architecture_search_20250101_120000"
    exit 1
fi

SEARCH_RESULTS_DIR="$1"
INFERENCE_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
INFERENCE_RESULTS_DIR="${SEARCH_RESULTS_DIR}/inference_results_${INFERENCE_TIMESTAMP}"

echo "=========================================="
echo "SWAE Architecture Search Inference"
echo "=========================================="
echo "Search results directory: $SEARCH_RESULTS_DIR"
echo "Inference results directory: $INFERENCE_RESULTS_DIR"
echo "Dataset: /u/tawal/0620-NN-based-compression-thera/tt_q01/"
echo "🎯 Evaluating: Speed + Accuracy for all trained models"
echo "=========================================="

# Create inference results directory
mkdir -p "$INFERENCE_RESULTS_DIR"

# Create comprehensive results CSV
INFERENCE_CSV="$INFERENCE_RESULTS_DIR/architecture_inference_results.csv"
echo "latent_dim,arch_config,channels,lr,batch_size,lambda_reg,compression_ratio,model_path,psnr,mse,mae,correlation,mean_rel_error,rms_rel_error,p95_rel_error,compression_speed_gbps,decompression_speed_gbps,compression_speed_mbps,decompression_speed_mbps,samples_per_sec_comp,samples_per_sec_decomp,model_params,inference_time,status" > "$INFERENCE_CSV"

# Create summary log
INFERENCE_LOG="$INFERENCE_RESULTS_DIR/inference_summary.txt"
echo "SWAE Architecture Search Inference Summary" > "$INFERENCE_LOG"
echo "Started: $(date)" >> "$INFERENCE_LOG"
echo "Search directory: $SEARCH_RESULTS_DIR" >> "$INFERENCE_LOG"
echo "==========================================" >> "$INFERENCE_LOG"

# Find all trained models
echo "Searching for trained models..."
model_configs=($(find "$SEARCH_RESULTS_DIR" -name "config_*" -type d | sort))

echo "Found ${#model_configs[@]} model configurations"

if [ ${#model_configs[@]} -eq 0 ]; then
    echo "❌ No model configurations found in $SEARCH_RESULTS_DIR"
    echo "Make sure the architecture search has completed successfully."
    exit 1
fi

# Track progress
total_models=0
successful_inferences=0

# Loop through each model configuration
for config_dir in "${model_configs[@]}"; do
    config_name=$(basename "$config_dir")
    echo ""
    echo "🔍 Processing configuration: $config_name"
    
    # Find best model in this configuration
    best_model="$config_dir/best_model.pth"
    final_model="$config_dir/final_model.pth"
    
    # Prefer best model, fallback to final model
    if [ -f "$best_model" ]; then
        model_path="$best_model"
        model_type="best"
    elif [ -f "$final_model" ]; then
        model_path="$final_model"
        model_type="final"
    else
        echo "❌ No model found in $config_dir"
        echo "Config $config_name: NO_MODEL" >> "$INFERENCE_LOG"
        continue
    fi
    
    total_models=$((total_models + 1))
    
    # Extract architecture information
    arch_info=$(get_arch_info "$config_dir")
    IFS=',' read -r latent_dim arch_config lr batch_size lambda_reg <<< "$arch_info"
    
    # Calculate compression ratio
    compression_ratio=$(python -c "print(f'{343/$latent_dim:.1f}:1')")
    
    echo "   Model: $model_type ($model_path)"
    echo "   Latent dim: $latent_dim (${compression_ratio})"
    echo "   Architecture: $arch_config"
    echo "   Hyperparams: lr=$lr, bs=$batch_size, reg=$lambda_reg"
    
    # Create output directory for this inference
    inference_output_dir="$INFERENCE_RESULTS_DIR/inference_${config_name}"
    mkdir -p "$inference_output_dir"
    
    # Log inference start
    echo "Config $config_name ($model_type model):" >> "$INFERENCE_LOG"
    echo "  Started: $(date)" >> "$INFERENCE_LOG"
    echo "  Model: $model_path" >> "$INFERENCE_LOG"
    
    # Record start time
    start_time=$(date +%s)
    
    # Run inference with speed benchmarking
    echo "   Running inference and speed benchmarks..."
    python inference_swae_u_chi_validation.py \
        --data-folder /u/tawal/0620-NN-based-compression-thera/tt_q01/ \
        --model-path "$model_path" \
        --output-dir "$inference_output_dir" \
        --num-samples 20 \
        --num-vti-samples 3 \
        --batch-size 32 \
        --normalize-method pos_log \
        --device auto \
        2>&1 | tee "$inference_output_dir/inference_log.txt"
    
    # Calculate inference time
    end_time=$(date +%s)
    inference_time=$((end_time - start_time))
    
    # Check if inference completed successfully
    if [ $? -eq 0 ] && [ -f "$inference_output_dir/average_metrics.txt" ]; then
        successful_inferences=$((successful_inferences + 1))
        echo "✅ Inference completed successfully for $config_name"
        echo "  Status: SUCCESS, Time: ${inference_time}s" >> "$INFERENCE_LOG"
        
        # Extract metrics from results
        python << EOF
import re
import os

# Read the metrics file
metrics_file = '$inference_output_dir/average_metrics.txt'
if os.path.exists(metrics_file):
    with open(metrics_file, 'r') as f:
        content = f.read()
    
    # Extract metrics using regex
    def extract_metric(content, metric_name):
        pattern = f'{metric_name}: ([0-9.-]+(?:e[-+]?[0-9]+)?)'
        match = re.search(pattern, content)
        return match.group(1) if match else 'N/A'
    
    # Extract all metrics
    psnr = extract_metric(content, 'Average PSNR')
    mse = extract_metric(content, 'Average MSE')
    mae = extract_metric(content, 'Average MAE')
    correlation = extract_metric(content, 'Average Correlation')
    mean_rel_err = extract_metric(content, 'Mean relative error')
    rms_rel_err = extract_metric(content, 'RMS relative error')
    p95_rel_err = extract_metric(content, 'P95 relative error')
    
    # Extract speed metrics
    comp_speed_gbps = extract_metric(content, 'Compression speed')
    decomp_speed_gbps = extract_metric(content, 'Decompression speed')
    comp_speed_mbps = str(float(comp_speed_gbps) * 1000) if comp_speed_gbps != 'N/A' else 'N/A'
    decomp_speed_mbps = str(float(decomp_speed_gbps) * 1000) if decomp_speed_gbps != 'N/A' else 'N/A'
    
    # Extract throughput
    comp_throughput = extract_metric(content, 'Compression throughput')
    decomp_throughput = extract_metric(content, 'Decompression throughput')
    
    # Count model parameters (try to extract from training log)
    training_log = '$config_dir/training_log.txt'
    model_params = 'N/A'
    if os.path.exists(training_log):
        with open(training_log, 'r') as f:
            log_content = f.read()
        params_match = re.search(r'Total parameters: ([0-9,]+)', log_content)
        if params_match:
            model_params = params_match.group(1).replace(',', '')
    
    # Get channels from training log
    channels = 'N/A'
    if os.path.exists(training_log):
        with open(training_log, 'r') as f:
            log_content = f.read()
        channels_match = re.search(r'channels=\[([0-9, ]+)\]', log_content)
        if channels_match:
            channels = channels_match.group(1).replace(' ', '')
    
    # Write to CSV
    csv_line = f"$latent_dim,$arch_config,{channels},$lr,$batch_size,$lambda_reg,$compression_ratio,$model_path,{psnr},{mse},{mae},{correlation},{mean_rel_err},{rms_rel_err},{p95_rel_err},{comp_speed_gbps},{decomp_speed_gbps},{comp_speed_mbps},{decomp_speed_mbps},{comp_throughput},{decomp_throughput},{model_params},$inference_time,SUCCESS"
    
    with open('$INFERENCE_CSV', 'a') as f:
        f.write(csv_line + '\\n')
        
    print(f"✅ Metrics extracted and saved for $config_name")
else:
    print(f"❌ Metrics file not found for $config_name")
EOF
        
    else
        echo "❌ Inference failed for $config_name"
        echo "  Status: FAILED, Time: ${inference_time}s" >> "$INFERENCE_LOG"
        
        # Add failure to CSV
        echo "$latent_dim,$arch_config,N/A,$lr,$batch_size,$lambda_reg,$compression_ratio,$model_path,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,$inference_time,FAILED" >> "$INFERENCE_CSV"
    fi
    
    echo "Progress: $successful_inferences/$total_models configurations completed"
    echo "----------------------------------------"
done

echo ""
echo "=========================================="
echo "Architecture Search Inference Completed"
echo "=========================================="
echo "Total models processed: $total_models"
echo "Successful inferences: $successful_inferences"
echo "Success rate: $(python -c "print(f'{$successful_inferences/$total_models*100:.1f}%')" 2>/dev/null || echo 'N/A')"
echo "Results saved to: $INFERENCE_CSV"
echo "Summary saved to: $INFERENCE_LOG"
echo "Individual results in: $INFERENCE_RESULTS_DIR"
echo "Completed at: $(date)"

# Add final summary to log
echo "" >> "$INFERENCE_LOG"
echo "Architecture search inference completed: $(date)" >> "$INFERENCE_LOG"
echo "Total models: $total_models" >> "$INFERENCE_LOG"
echo "Successful inferences: $successful_inferences" >> "$INFERENCE_LOG"
echo "Success rate: $(python -c "print(f'{$successful_inferences/$total_models*100:.1f}%')" 2>/dev/null || echo 'N/A')" >> "$INFERENCE_LOG"

# Create enhanced analysis script
cat > "$INFERENCE_RESULTS_DIR/analyze_architecture_results.py" << 'EOF'
#!/usr/bin/env python3
"""
Enhanced Architecture Search Results Analysis
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_architecture_results(csv_path):
    # Load results
    df = pd.read_csv(csv_path)
    successful = df[df['status'] == 'SUCCESS'].copy()
    
    if len(successful) == 0:
        print("No successful inferences found!")
        return
    
    # Convert numeric columns
    numeric_cols = ['latent_dim', 'psnr', 'mse', 'mae', 'correlation', 
                   'compression_speed_mbps', 'decompression_speed_mbps', 
                   'model_params', 'inference_time']
    
    for col in numeric_cols:
        if col in successful.columns:
            successful[col] = pd.to_numeric(successful[col], errors='coerce')
    
    # Calculate compression ratio numerically
    successful['compression_ratio_num'] = 343 / successful['latent_dim']
    
    print(f"Enhanced Architecture Search Analysis")
    print(f"===================================")
    print(f"Total configurations: {len(df)}")
    print(f"Successful inferences: {len(successful)}")
    print(f"Success rate: {len(successful)/len(df)*100:.1f}%")
    
    # Find optimal configurations
    print(f"\n🏆 Top Configurations by Different Criteria:")
    print(f"=" * 60)
    
    # Best quality (PSNR)
    best_quality = successful.loc[successful['psnr'].idxmax()]
    print(f"\n🥇 Best Quality (PSNR):")
    print(f"   Latent Dim: {best_quality['latent_dim']} ({best_quality['compression_ratio_num']:.1f}:1)")
    print(f"   PSNR: {best_quality['psnr']:.2f} dB")
    print(f"   Architecture: {best_quality['arch_config']}")
    print(f"   Speed: {best_quality['compression_speed_mbps']:.0f} MBps")
    
    # Best compression speed
    if 'compression_speed_mbps' in successful.columns:
        best_speed = successful.loc[successful['compression_speed_mbps'].idxmax()]
        print(f"\n🚀 Fastest Compression:")
        print(f"   Latent Dim: {best_speed['latent_dim']} ({best_speed['compression_ratio_num']:.1f}:1)")
        print(f"   Speed: {best_speed['compression_speed_mbps']:.0f} MBps")
        print(f"   PSNR: {best_speed['psnr']:.2f} dB")
        print(f"   Architecture: {best_speed['arch_config']}")
    
    # Best efficiency (PSNR per parameter)
    successful['efficiency'] = successful['psnr'] / (successful['model_params'] / 1e6)  # PSNR per million params
    best_efficiency = successful.loc[successful['efficiency'].idxmax()]
    print(f"\n⚡ Most Efficient (PSNR/M-params):")
    print(f"   Latent Dim: {best_efficiency['latent_dim']} ({best_efficiency['compression_ratio_num']:.1f}:1)")
    print(f"   Efficiency: {best_efficiency['efficiency']:.2f} dB/M-params")
    print(f"   PSNR: {best_efficiency['psnr']:.2f} dB")
    print(f"   Parameters: {best_efficiency['model_params']/1e6:.1f}M")
    
    # Create comprehensive visualizations
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # 1. PSNR vs Compression Ratio
    axes[0,0].scatter(successful['compression_ratio_num'], successful['psnr'], 
                     c=successful['latent_dim'], cmap='viridis', s=60, alpha=0.7)
    axes[0,0].set_xlabel('Compression Ratio')
    axes[0,0].set_ylabel('PSNR (dB)')
    axes[0,0].set_title('Quality vs Compression Trade-off')
    
    # 2. Speed vs Quality
    if 'compression_speed_mbps' in successful.columns:
        axes[0,1].scatter(successful['compression_speed_mbps'], successful['psnr'],
                         c=successful['latent_dim'], cmap='viridis', s=60, alpha=0.7)
        axes[0,1].set_xlabel('Compression Speed (MBps)')
        axes[0,1].set_ylabel('PSNR (dB)')
        axes[0,1].set_title('Speed vs Quality Trade-off')
    
    # 3. Model Size vs Quality
    axes[0,2].scatter(successful['model_params']/1e6, successful['psnr'],
                     c=successful['latent_dim'], cmap='viridis', s=60, alpha=0.7)
    axes[0,2].set_xlabel('Model Parameters (Millions)')
    axes[0,2].set_ylabel('PSNR (dB)')
    axes[0,2].set_title('Model Size vs Quality')
    
    # 4. Architecture Performance
    arch_performance = successful.groupby('arch_config')['psnr'].agg(['mean', 'std']).reset_index()
    axes[1,0].bar(arch_performance['arch_config'], arch_performance['mean'], 
                 yerr=arch_performance['std'], capsize=5)
    axes[1,0].set_xlabel('Architecture Configuration')
    axes[1,0].set_ylabel('Average PSNR (dB)')
    axes[1,0].set_title('Architecture Performance Comparison')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 5. Latent Dimension Analysis
    latent_performance = successful.groupby('latent_dim').agg({
        'psnr': ['mean', 'std'],
        'compression_speed_mbps': 'mean',
        'compression_ratio_num': 'first'
    }).reset_index()
    
    axes[1,1].errorbar(latent_performance['latent_dim'], 
                      latent_performance['psnr']['mean'],
                      yerr=latent_performance['psnr']['std'],
                      marker='o', capsize=5, capthick=2)
    axes[1,1].set_xlabel('Latent Dimension')
    axes[1,1].set_ylabel('Average PSNR (dB)')
    axes[1,1].set_title('Quality vs Latent Dimension')
    
    # 6. Speed Distribution
    if 'compression_speed_mbps' in successful.columns:
        axes[1,2].hist(successful['compression_speed_mbps'], bins=15, alpha=0.7, edgecolor='black')
        axes[1,2].set_xlabel('Compression Speed (MBps)')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].set_title('Speed Distribution')
    
    # 7. Hyperparameter Impact - Learning Rate
    lr_performance = successful.groupby('lr')['psnr'].mean().sort_index()
    axes[2,0].bar(range(len(lr_performance)), lr_performance.values)
    axes[2,0].set_xticks(range(len(lr_performance)))
    axes[2,0].set_xticklabels([f'{lr:.0e}' for lr in lr_performance.index])
    axes[2,0].set_xlabel('Learning Rate')
    axes[2,0].set_ylabel('Average PSNR (dB)')
    axes[2,0].set_title('Learning Rate Impact')
    
    # 8. Batch Size Impact
    bs_performance = successful.groupby('batch_size')['psnr'].mean().sort_index()
    axes[2,1].bar(bs_performance.index, bs_performance.values)
    axes[2,1].set_xlabel('Batch Size')
    axes[2,1].set_ylabel('Average PSNR (dB)')
    axes[2,1].set_title('Batch Size Impact')
    
    # 9. Efficiency Analysis
    axes[2,2].scatter(successful['model_params']/1e6, successful['efficiency'],
                     c=successful['latent_dim'], cmap='viridis', s=60, alpha=0.7)
    axes[2,2].set_xlabel('Model Parameters (Millions)')
    axes[2,2].set_ylabel('Efficiency (PSNR/M-params)')
    axes[2,2].set_title('Parameter Efficiency')
    
    plt.tight_layout()
    plt.savefig(csv_path.replace('.csv', '_comprehensive_analysis.png'), 
                dpi=300, bbox_inches='tight')
    print(f"\n📊 Comprehensive analysis plots saved!")
    
    # Save top configurations
    print(f"\n📋 Saving top configurations...")
    top_configs = pd.DataFrame({
        'Metric': ['Best Quality', 'Fastest Compression', 'Most Efficient'],
        'Latent_Dim': [best_quality['latent_dim'], 
                      best_speed['latent_dim'] if 'compression_speed_mbps' in successful.columns else 'N/A',
                      best_efficiency['latent_dim']],
        'Architecture': [best_quality['arch_config'],
                        best_speed['arch_config'] if 'compression_speed_mbps' in successful.columns else 'N/A',
                        best_efficiency['arch_config']],
        'PSNR': [best_quality['psnr'],
                best_speed['psnr'] if 'compression_speed_mbps' in successful.columns else 'N/A',
                best_efficiency['psnr']],
        'Compression_Ratio': [f"{best_quality['compression_ratio_num']:.1f}:1",
                             f"{best_speed['compression_ratio_num']:.1f}:1" if 'compression_speed_mbps' in successful.columns else 'N/A',
                             f"{best_efficiency['compression_ratio_num']:.1f}:1"]
    })
    
    top_configs.to_csv(csv_path.replace('.csv', '_top_configurations.csv'), index=False)
    print(f"📁 Top configurations saved to: {csv_path.replace('.csv', '_top_configurations.csv')}")

if __name__ == "__main__":
    analyze_architecture_results('architecture_inference_results.csv')
EOF

chmod +x "$INFERENCE_RESULTS_DIR/analyze_architecture_results.py"

echo ""
echo "🎯 Next Steps:"
echo "1. Analyze results: cd $INFERENCE_RESULTS_DIR && python analyze_architecture_results.py"
echo "2. Review comprehensive analysis plots and top configurations"
echo "3. Select optimal architectures for production deployment"
echo ""
echo "🏁 Architecture search inference completed successfully!"