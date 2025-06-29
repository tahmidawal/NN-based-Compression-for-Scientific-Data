#!/bin/bash
#SBATCH --job-name=swae_u_chi_multi_inference
#SBATCH --output=logs/swae_u_chi_multi_inference_%j.out
#SBATCH --error=logs/swae_u_chi_multi_inference_%j.err
#SBATCH --time=04:00:00
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
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Print GPU information
nvidia-smi

# Print Python and PyTorch information
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo "=========================================="
echo "Multi-Latent Dimension SWAE Inference"
echo "=========================================="

# Find the multi-latent training directory
# Look for the most recent multi_latent directory
MULTI_LATENT_DIR=$(ls -td ./save/swae_u_chi_multi_latent_* 2>/dev/null | head -1)

if [ -z "$MULTI_LATENT_DIR" ] || [ ! -d "$MULTI_LATENT_DIR" ]; then
    echo "❌ Error: No multi-latent training directory found!"
    echo "Expected pattern: ./save/swae_u_chi_multi_latent_*"
    echo "Available directories:"
    ls -la ./save/ 2>/dev/null | grep swae_u_chi || echo "No SWAE directories found"
    exit 1
fi

echo "Found multi-latent training directory: $MULTI_LATENT_DIR"

# Create output directory for all inference results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE_DIR="inference_multi_latent_results_${TIMESTAMP}"
mkdir -p "$OUTPUT_BASE_DIR"

# Create summary files
SUMMARY_CSV="$OUTPUT_BASE_DIR/latent_dim_comparison.csv"
SUMMARY_LOG="$OUTPUT_BASE_DIR/inference_summary.log"

echo "Output directory: $OUTPUT_BASE_DIR"
echo "Summary CSV: $SUMMARY_CSV"
echo "Detailed log: $SUMMARY_LOG"

# Initialize summary files
echo "Multi-Latent Dimension SWAE Inference Results" > "$SUMMARY_LOG"
echo "Started: $(date)" >> "$SUMMARY_LOG"
echo "Base training dir: $MULTI_LATENT_DIR" >> "$SUMMARY_LOG"
echo "==========================================" >> "$SUMMARY_LOG"

# Create CSV header
echo "latent_dim,compression_ratio,mse,psnr,mae,correlation,mean_relative_error_percent,rms_relative_error_percent,p95_relative_error_percent,compression_speed_gbps,decompression_speed_gbps,compression_speed_mbps,decompression_speed_mbps,samples_per_sec_comp,samples_per_sec_decomp,model_path,status" > "$SUMMARY_CSV"

# Find all latent dimension directories
LATENT_DIRS=($(find "$MULTI_LATENT_DIR" -name "latent_dim_*" -type d | sort -V))

if [ ${#LATENT_DIRS[@]} -eq 0 ]; then
    echo "❌ Error: No latent dimension directories found in $MULTI_LATENT_DIR"
    echo "Expected pattern: latent_dim_*"
    ls -la "$MULTI_LATENT_DIR"
    exit 1
fi

echo "Found ${#LATENT_DIRS[@]} latent dimension directories:"
for dir in "${LATENT_DIRS[@]}"; do
    echo "  $(basename "$dir")"
done

echo "=========================================="

# Process each latent dimension
for latent_dir in "${LATENT_DIRS[@]}"; do
    # Extract latent dimension from directory name
    latent_dim=$(basename "$latent_dir" | sed 's/latent_dim_//')
    
    echo ""
    echo "=========================================="
    echo "Processing latent_dim = $latent_dim"
    echo "=========================================="
    
    # Calculate compression ratio
    compression_ratio=$(python -c "print(f'{343/$latent_dim:.1f}')")
    echo "Compression ratio: ${compression_ratio}:1"
    
    # Look for best model in this directory
    best_model="$latent_dir/best_model.pth"
    if [ ! -f "$best_model" ]; then
        # Try other possible model names
        best_model=$(find "$latent_dir" -name "*.pth" -type f | head -1)
    fi
    
    if [ ! -f "$best_model" ]; then
        echo "❌ No model found for latent_dim = $latent_dim"
        echo "  Directory: $latent_dir"
        echo "  Available files:"
        ls -la "$latent_dir" 2>/dev/null || echo "    Directory not accessible"
        
        # Log failure and add to CSV
        echo "" >> "$SUMMARY_LOG"
        echo "Latent dim $latent_dim: FAILED - No model file found" >> "$SUMMARY_LOG"
        echo "$latent_dim,$compression_ratio,,,,,,,,,,,,,,$latent_dir,NO_MODEL" >> "$SUMMARY_CSV"
        continue
    fi
    
    echo "Found model: $best_model"
    
    # Create output directory for this latent dimension
    LATENT_OUTPUT_DIR="$OUTPUT_BASE_DIR/latent_dim_${latent_dim}"
    mkdir -p "$LATENT_OUTPUT_DIR"
    
    # Log start of inference
    echo "" >> "$SUMMARY_LOG"
    echo "Latent dim $latent_dim (compression ${compression_ratio}:1):" >> "$SUMMARY_LOG"
    echo "  Model: $best_model" >> "$SUMMARY_LOG"
    echo "  Output: $LATENT_OUTPUT_DIR" >> "$SUMMARY_LOG"
    echo "  Started: $(date)" >> "$SUMMARY_LOG"
    
    # Run inference
    echo "Running inference..."
    python inference_swae_u_chi_validation.py \
        --model-path "$best_model" \
        --data-folder /u/tawal/0620-NN-based-compression-thera/tt_q01/ \
        --output-dir "$LATENT_OUTPUT_DIR" \
        --normalize-method pos_log \
        --batch-size 32 \
        --num-samples 10 \
        --num-vti-samples 2 \
        --device auto \
        2>&1 | tee "$LATENT_OUTPUT_DIR/inference_log.txt"
    
    # Check if inference completed successfully
    if [ $? -eq 0 ] && [ -f "$LATENT_OUTPUT_DIR/average_metrics.txt" ]; then
        echo "✅ Inference completed successfully for latent_dim = $latent_dim"
        echo "  Completed: $(date)" >> "$SUMMARY_LOG"
        echo "  Status: SUCCESS" >> "$SUMMARY_LOG"
        
        # Extract key metrics from the results file
        METRICS_FILE="$LATENT_OUTPUT_DIR/average_metrics.txt"
        
        # Use Python to extract metrics from the file
        python << EOF
import re

# Read the metrics file
with open('$METRICS_FILE', 'r') as f:
    content = f.read()

# Extract metrics using regex
def extract_metric(content, metric_name):
    pattern = f'{metric_name}: ([0-9.-]+(?:e[-+]?[0-9]+)?)'
    match = re.search(pattern, content)
    return match.group(1) if match else 'N/A'

mse = extract_metric(content, 'mse')
psnr = extract_metric(content, 'psnr')
mae = extract_metric(content, 'mae')
correlation = extract_metric(content, 'correlation')
mean_rel_err = extract_metric(content, 'mean_relative_error_percent')
rms_rel_err = extract_metric(content, 'rms_relative_error_percent')
p95_rel_err_match = re.search(r'p95_relative_error: ([0-9.-]+(?:e[-+]?[0-9]+)?)', content)
p95_rel_err = str(float(p95_rel_err_match.group(1)) * 100) if p95_rel_err_match else 'N/A'

# Extract speed metrics
comp_speed_gbps = extract_metric(content, 'Compression Speed')
decomp_speed_gbps = extract_metric(content, 'Decompression Speed')
samples_comp_match = re.search(r'Samples/second \(compression\): ([0-9,]+)', content)
samples_decomp_match = re.search(r'Samples/second \(decompression\): ([0-9,]+)', content)

samples_comp = samples_comp_match.group(1).replace(',', '') if samples_comp_match else 'N/A'
samples_decomp = samples_decomp_match.group(1).replace(',', '') if samples_decomp_match else 'N/A'

# Convert GBps to MBps (multiply by 1000)
comp_speed_mbps = str(float(comp_speed_gbps) * 1000) if comp_speed_gbps != 'N/A' else 'N/A'
decomp_speed_mbps = str(float(decomp_speed_gbps) * 1000) if decomp_speed_gbps != 'N/A' else 'N/A'

# Write to CSV
csv_line = f"$latent_dim,$compression_ratio,{mse},{psnr},{mae},{correlation},{mean_rel_err},{rms_rel_err},{p95_rel_err},{comp_speed_gbps},{decomp_speed_gbps},{comp_speed_mbps},{decomp_speed_mbps},{samples_comp},{samples_decomp},$best_model,SUCCESS"
with open('$SUMMARY_CSV', 'a') as f:
    f.write(csv_line + '\n')

print(f"Extracted metrics for latent_dim=$latent_dim:")
print(f"  MSE: {mse}")
print(f"  PSNR: {psnr}")
print(f"  MAE: {mae}")
print(f"  Correlation: {correlation}")
print(f"  Mean Rel Error %: {mean_rel_err}")
print(f"  RMS Rel Error %: {rms_rel_err}")
print(f"  P95 Rel Error %: {p95_rel_err}")
print(f"  Compression Speed: {comp_speed_gbps} GBps ({comp_speed_mbps} MBps)")
print(f"  Decompression Speed: {decomp_speed_gbps} GBps ({decomp_speed_mbps} MBps)")
print(f"  Samples/sec (comp): {samples_comp}")
print(f"  Samples/sec (decomp): {samples_decomp}")
EOF
        
        # Add metrics summary to log
        echo "  Metrics extracted and added to CSV" >> "$SUMMARY_LOG"
        
    else
        echo "❌ Inference failed for latent_dim = $latent_dim"
        echo "  Completed: $(date)" >> "$SUMMARY_LOG"
        echo "  Status: FAILED" >> "$SUMMARY_LOG"
        echo "$latent_dim,$compression_ratio,,,,,,,,,,,,,,$best_model,FAILED" >> "$SUMMARY_CSV"
    fi
    
    echo "Results saved to: $LATENT_OUTPUT_DIR"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "All Inference Jobs Completed"
echo "=========================================="
echo "Results summary:"
echo "  Base directory: $OUTPUT_BASE_DIR"
echo "  CSV summary: $SUMMARY_CSV"
echo "  Detailed log: $SUMMARY_LOG"
echo "Completed at: $(date)"

# Add completion info to summary
echo "" >> "$SUMMARY_LOG"
echo "All inference completed: $(date)" >> "$SUMMARY_LOG"

# Display final CSV summary
echo ""
echo "📊 FINAL RESULTS SUMMARY:"
echo "=========================="
if [ -f "$SUMMARY_CSV" ]; then
    echo "CSV contents:"
    cat "$SUMMARY_CSV"
    echo ""
    echo "📈 Quick Analysis:"
    python << EOF
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('$SUMMARY_CSV')
    successful = df[df['status'] == 'SUCCESS']
    
    if len(successful) > 0:
        print(f"Successfully processed {len(successful)} out of {len(df)} models")
        print("\nCompression vs Quality vs Speed Trade-off:")
        print("==========================================")
        for _, row in successful.iterrows():
            if pd.notna(row['psnr']) and row['psnr'] != '':
                comp_speed_mbps = float(row['compression_speed_mbps']) if pd.notna(row['compression_speed_mbps']) and row['compression_speed_mbps'] != 'N/A' else 0
                print(f"Latent dim {int(row['latent_dim']):2d} (compression {row['compression_ratio']:5.1f}:1) -> PSNR: {float(row['psnr']):5.1f} dB, RMS error: {float(row['rms_relative_error_percent']):5.2f}%, Speed: {comp_speed_mbps:5.0f} MBps")
    else:
        print("No successful inference results to analyze")
        
except Exception as e:
    print(f"Error analyzing results: {e}")
EOF
else
    echo "❌ Summary CSV not found!"
fi

echo ""
echo "🎯 Analysis complete! Check the CSV file for detailed results."
echo "🎯 Use the CSV data to plot compression ratio vs PSNR/error curves." 