#!/bin/bash

# Consolidate Parallel Architecture Search Results
# This script combines results from all 5 parallel jobs and creates comprehensive analysis
#!/bin/bash
#SBATCH --job-name=swae_arch_latent4
#SBATCH --output=logs/swae_arch_latent4_%j.out
#SBATCH --error=logs/swae_arch_latent4_%j.err
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcqs-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
echo "=========================================="
echo "SWAE Parallel Architecture Search Results"
echo "=========================================="

# Check if job IDs file exists
if [[ ! -f ".parallel_job_ids.txt" ]]; then
    echo "❌ No parallel job IDs found. Please run submit_parallel_arch_search.sh first."
    exit 1
fi

# Read job IDs
JOB_IDS=($(cat .parallel_job_ids.txt))
echo "Found ${#JOB_IDS[@]} parallel jobs: ${JOB_IDS[@]}"

# Check job status
echo ""
echo "📊 Checking job status..."
sacct -j $(IFS=,; echo "${JOB_IDS[*]}") --format=JobID,JobName,State,Start,End,Elapsed

# Find result directories
echo ""
echo "🔍 Searching for result directories..."
RESULT_DIRS=($(find ./save -name "swae_architecture_search_parallel_*" -type d | sort))

if [[ ${#RESULT_DIRS[@]} -eq 0 ]]; then
    echo "❌ No result directories found. Jobs may still be running or failed."
    echo "💡 Check job status and logs:"
    echo "   squeue -u \$USER"
    echo "   Check log files in logs/ directory"
    exit 1
fi

echo "Found ${#RESULT_DIRS[@]} result directories:"
for dir in "${RESULT_DIRS[@]}"; do
    echo "   $dir"
done

# Create consolidated results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONSOLIDATED_DIR="./save/swae_architecture_search_consolidated_${TIMESTAMP}"
mkdir -p "$CONSOLIDATED_DIR"

echo ""
echo "📁 Creating consolidated results in: $CONSOLIDATED_DIR"

# Initialize consolidated CSV
CONSOLIDATED_CSV="$CONSOLIDATED_DIR/consolidated_architecture_search_results.csv"
echo "latent_dim,arch_config,channels,lr,batch_size,lambda_reg,compression_ratio,final_psnr,best_psnr,training_time,status" > "$CONSOLIDATED_CSV"

# Consolidate all CSV results
echo ""
echo "📊 Consolidating CSV results..."
total_configs=0
successful_configs=0

for result_dir in "${RESULT_DIRS[@]}"; do
    # Find all CSV files in this directory
    csv_files=($(find "$result_dir" -name "latent_*_results.csv" -type f))
    
    for csv_file in "${csv_files[@]}"; do
        if [[ -f "$csv_file" ]]; then
            echo "   Processing: $csv_file"
            # Skip header and append to consolidated file
            tail -n +2 "$csv_file" >> "$CONSOLIDATED_CSV"
            
            # Count configurations
            configs_in_file=$(tail -n +2 "$csv_file" | wc -l)
            successful_in_file=$(tail -n +2 "$csv_file" | grep -c "SUCCESS" || echo 0)
            total_configs=$((total_configs + configs_in_file))
            successful_configs=$((successful_configs + successful_in_file))
        fi
    done
done

echo ""
echo "📈 Consolidation Summary:"
echo "   Total configurations: $total_configs"
echo "   Successful configurations: $successful_configs"
echo "   Success rate: $(python -c "print(f'{$successful_configs/$total_configs*100:.1f}%')" 2>/dev/null || echo "N/A")"

# Create comprehensive analysis script
cat > "$CONSOLIDATED_DIR/comprehensive_analysis.py" << 'EOF'
#!/usr/bin/env python3
"""
Comprehensive Analysis of Parallel Architecture Search Results
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_consolidated_results(csv_path):
    print("🔬 SWAE Parallel Architecture Search - Comprehensive Analysis")
    print("=" * 65)
    
    # Load results
    try:
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} total configurations")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return
    
    # Filter successful results
    successful = df[df['status'] == 'SUCCESS'].copy()
    
    if len(successful) == 0:
        print("❌ No successful configurations found!")
        return
    
    print(f"✅ Found {len(successful)} successful configurations")
    print(f"📈 Success rate: {len(successful)/len(df)*100:.1f}%")
    
    # Convert numeric columns
    successful['final_psnr'] = pd.to_numeric(successful['final_psnr'], errors='coerce')
    successful['best_psnr'] = pd.to_numeric(successful['best_psnr'], errors='coerce')
    successful['training_time'] = pd.to_numeric(successful['training_time'], errors='coerce')
    
    print("\n" + "="*65)
    print("🏆 BEST CONFIGURATIONS PER LATENT DIMENSION")
    print("="*65)
    
    best_configs = []
    
    for latent_dim in sorted(successful['latent_dim'].unique()):
        dim_results = successful[successful['latent_dim'] == latent_dim]
        if len(dim_results) > 0:
            best_config = dim_results.loc[dim_results['final_psnr'].idxmax()]
            compression_ratio = 343 / latent_dim
            
            print(f"\n🎯 Latent Dim {latent_dim} ({compression_ratio:.1f}:1 compression):")
            print(f"   Best PSNR: {best_config['final_psnr']:.2f} dB")
            print(f"   Architecture: {best_config['arch_config']}")
            print(f"   Channels: [{best_config['channels']}]")
            print(f"   Hyperparams: lr={best_config['lr']}, bs={best_config['batch_size']}, λ={best_config['lambda_reg']}")
            print(f"   Training time: {best_config['training_time']:.0f}s ({best_config['training_time']/60:.1f} min)")
            
            # Store for comparison
            best_configs.append({
                'latent_dim': latent_dim,
                'compression_ratio': compression_ratio,
                'best_psnr': best_config['final_psnr'],
                'architecture': best_config['arch_config'],
                'channels': best_config['channels'],
                'training_time': best_config['training_time']
            })
    
    print("\n" + "="*65)
    print("📊 ARCHITECTURE PERFORMANCE COMPARISON")
    print("="*65)
    
    # Architecture comparison
    arch_performance = successful.groupby('arch_config').agg({
        'final_psnr': ['mean', 'std', 'count']
    }).round(2)
    
    print("📈 Average PSNR by Architecture:")
    for arch in arch_performance.index:
        mean_psnr = arch_performance.loc[arch, ('final_psnr', 'mean')]
        std_psnr = arch_performance.loc[arch, ('final_psnr', 'std')]
        count = arch_performance.loc[arch, ('final_psnr', 'count')]
        print(f"   {arch:10s}: {mean_psnr:5.2f} ± {std_psnr:4.2f} dB ({count:2.0f} configs)")
    
    # Hyperparameter analysis
    print("\n🔧 Hyperparameter Impact:")
    
    # Learning rate impact
    lr_performance = successful.groupby('lr')['final_psnr'].agg(['mean', 'std', 'count']).round(2)
    print("   Learning Rate Impact:")
    for lr in lr_performance.index:
        mean_psnr = lr_performance.loc[lr, 'mean']
        std_psnr = lr_performance.loc[lr, 'std']
        count = lr_performance.loc[lr, 'count']
        print(f"     {lr:.0e}: {mean_psnr:5.2f} ± {std_psnr:4.2f} dB ({count:2.0f} configs)")
    
    # Batch size impact
    bs_performance = successful.groupby('batch_size')['final_psnr'].agg(['mean', 'std', 'count']).round(2)
    print("   Batch Size Impact:")
    for bs in bs_performance.index:
        mean_psnr = bs_performance.loc[bs, 'mean']
        std_psnr = bs_performance.loc[bs, 'std']
        count = bs_performance.loc[bs, 'count']
        print(f"     {bs:3.0f}: {mean_psnr:5.2f} ± {std_psnr:4.2f} dB ({count:2.0f} configs)")
    
    print("\n" + "="*65)
    print("🎯 COMPRESSION-QUALITY TRADE-OFF ANALYSIS")
    print("="*65)
    
    best_df = pd.DataFrame(best_configs)
    
    print("📊 Compression Ratio vs Quality Trade-offs:")
    print("   Ratio    | PSNR   | Architecture | Training Time")
    print("   ---------|--------|--------------|---------------")
    for _, row in best_df.iterrows():
        print(f"   {row['compression_ratio']:6.1f}:1 | {row['best_psnr']:5.2f}  | {row['architecture']:10s} | {row['training_time']/60:6.1f} min")
    
    # Save detailed summary
    summary_path = csv_path.replace('.csv', '_detailed_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("SWAE Parallel Architecture Search - Detailed Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Total configurations tested: {len(df)}\n")
        f.write(f"Successful configurations: {len(successful)}\n")
        f.write(f"Success rate: {len(successful)/len(df)*100:.1f}%\n\n")
        
        f.write("Best Configurations per Latent Dimension:\n")
        f.write("-" * 45 + "\n")
        
        for _, row in best_df.iterrows():
            f.write(f"\nLatent Dim {row['latent_dim']} ({row['compression_ratio']:.1f}:1):\n")
            f.write(f"  Best PSNR: {row['best_psnr']:.2f} dB\n")
            f.write(f"  Architecture: {row['architecture']}\n")
            f.write(f"  Channels: {row['channels']}\n")
            f.write(f"  Training time: {row['training_time']:.0f}s\n")
    
    print(f"\n💾 Detailed summary saved to: {summary_path}")
    
    # Create visualization
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Compression vs PSNR
        ax1.scatter(best_df['compression_ratio'], best_df['best_psnr'], 
                   c='red', s=100, alpha=0.7, edgecolors='black')
        ax1.set_xlabel('Compression Ratio')
        ax1.set_ylabel('Best PSNR (dB)')
        ax1.set_title('Compression Ratio vs PSNR')
        ax1.grid(True, alpha=0.3)
        
        # Architecture comparison
        arch_means = successful.groupby('arch_config')['final_psnr'].mean()
        ax2.bar(arch_means.index, arch_means.values, color=['skyblue', 'lightcoral'])
        ax2.set_xlabel('Architecture')
        ax2.set_ylabel('Average PSNR (dB)')
        ax2.set_title('Architecture Performance')
        ax2.tick_params(axis='x', rotation=45)
        
        # Training time analysis
        ax3.scatter(successful['training_time']/60, successful['final_psnr'], 
                   c=successful['latent_dim'], cmap='viridis', alpha=0.6)
        ax3.set_xlabel('Training Time (minutes)')
        ax3.set_ylabel('Final PSNR (dB)')
        ax3.set_title('Training Time vs PSNR')
        ax3.grid(True, alpha=0.3)
        
        # PSNR distribution by latent dimension
        latent_dims = sorted(successful['latent_dim'].unique())
        psnr_data = [successful[successful['latent_dim']==ld]['final_psnr'].values 
                     for ld in latent_dims]
        ax4.boxplot(psnr_data, labels=latent_dims)
        ax4.set_xlabel('Latent Dimension')
        ax4.set_ylabel('Final PSNR (dB)')
        ax4.set_title('PSNR Distribution by Latent Dimension')
        
        plt.tight_layout()
        plot_path = csv_path.replace('.csv', '_comprehensive_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comprehensive plots saved to: {plot_path}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not create plots: {e}")
    
    print("\n" + "="*65)
    print("🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
    print("="*65)
    print(f"📁 All results consolidated in: {os.path.dirname(csv_path)}")
    print("🚀 Ready for production model selection!")

if __name__ == "__main__":
    analyze_consolidated_results('consolidated_architecture_search_results.csv')
EOF

# Make analysis script executable
chmod +x "$CONSOLIDATED_DIR/comprehensive_analysis.py"

# Copy individual summaries and logs
echo ""
echo "📋 Copying individual summaries..."
for result_dir in "${RESULT_DIRS[@]}"; do
    summary_files=($(find "$result_dir" -name "latent_*_summary.txt" -type f))
    for summary_file in "${summary_files[@]}"; do
        if [[ -f "$summary_file" ]]; then
            cp "$summary_file" "$CONSOLIDATED_DIR/"
            echo "   Copied: $(basename "$summary_file")"
        fi
    done
done

# Run comprehensive analysis
echo ""
echo "🔬 Running comprehensive analysis..."
cd "$CONSOLIDATED_DIR"
python comprehensive_analysis.py

echo ""
echo "=========================================="
echo "✅ CONSOLIDATION COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo "📁 Consolidated results: $CONSOLIDATED_DIR"
echo "📊 Total configurations: $total_configs"
echo "✅ Successful configurations: $successful_configs"
echo "📈 Success rate: $(python -c "print(f'{$successful_configs/$total_configs*100:.1f}%')" 2>/dev/null || echo "N/A")"
echo ""
echo "🎯 Next steps:"
echo "1. Review comprehensive analysis results"
echo "2. Select best architectures for each compression ratio"
echo "3. Run inference validation on selected models"
echo "4. Deploy optimal configurations for production"
echo ""
echo "🚀 Parallel architecture search completed successfully!" 