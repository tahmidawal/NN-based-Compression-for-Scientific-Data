#!/usr/bin/env python3
"""
Compression Ratio Architecture Analysis Plots
Generates three specific plots with compression ratio focus and detailed architectural configuration legends
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_compression_ratio_plots(csv_path):
    print("🎨 Creating Compression Ratio Architecture Analysis Plots")
    print("=" * 55)
    
    # Load and prepare data
    df = pd.read_csv(csv_path)
    successful = df[df['status'] == 'SUCCESS'].copy()
    
    # Convert numeric columns
    successful['final_psnr'] = pd.to_numeric(successful['final_psnr'], errors='coerce')
    successful['training_time'] = pd.to_numeric(successful['training_time'], errors='coerce')
    
    # Remove rows with NaN PSNR or training time
    successful = successful.dropna(subset=['final_psnr', 'training_time'])
    
    # Calculate compression ratio
    successful['compression_ratio'] = 343 / successful['latent_dim']
    
    # Create architecture labels combining arch_config and channels
    successful['arch_label'] = successful['arch_config'] + ' [' + successful['channels'] + ']'
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Define colors and markers for different architectures
    arch_configs = successful['arch_label'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(arch_configs)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    arch_style = {}
    for i, arch in enumerate(arch_configs):
        arch_style[arch] = {
            'color': colors[i % len(colors)], 
            'marker': markers[i % len(markers)],
            's': 64,  # size for scatter plot
            'alpha': 0.7
        }
    
    # Plot 1: PSNR vs Compression Ratio
    print("📊 Creating Plot 1: PSNR vs Compression Ratio")
    
    for arch in arch_configs:
        arch_data = successful[successful['arch_label'] == arch]
        if len(arch_data) > 0:
            ax1.scatter(arch_data['compression_ratio'], arch_data['final_psnr'], 
                       label=arch, **arch_style[arch])
    
    ax1.set_xlabel('Compression Ratio (X:1)', fontsize=12)
    ax1.set_ylabel('Final PSNR (dB)', fontsize=12)
    ax1.set_title('PSNR vs Compression Ratio\nby Architecture Configuration', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')  # Log scale for better visualization of compression ratios
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: PSNR vs Training Time
    print("📊 Creating Plot 2: PSNR vs Training Time")
    
    for arch in arch_configs:
        arch_data = successful[successful['arch_label'] == arch]
        if len(arch_data) > 0:
            ax2.scatter(arch_data['training_time']/60, arch_data['final_psnr'], 
                       label=arch, **arch_style[arch])
    
    ax2.set_xlabel('Training Time (minutes)', fontsize=12)
    ax2.set_ylabel('Final PSNR (dB)', fontsize=12)
    ax2.set_title('PSNR vs Training Time\nby Architecture Configuration', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 3: Training Time vs Compression Ratio
    print("📊 Creating Plot 3: Training Time vs Compression Ratio")
    
    for arch in arch_configs:
        arch_data = successful[successful['arch_label'] == arch]
        if len(arch_data) > 0:
            ax3.scatter(arch_data['compression_ratio'], arch_data['training_time']/60, 
                       label=arch, **arch_style[arch])
    
    ax3.set_xlabel('Compression Ratio (X:1)', fontsize=12)
    ax3.set_ylabel('Training Time (minutes)', fontsize=12)
    ax3.set_title('Training Time vs Compression Ratio\nby Architecture Configuration', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')  # Log scale for better visualization of compression ratios
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    # Save the plot
    plot_path = csv_path.replace('.csv', '_compression_ratio_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Compression ratio plots saved to: {plot_path}")
    
    # Print summary statistics
    print("\n📈 Compression Ratio Analysis Summary:")
    print("=" * 55)
    
    # Best configuration per compression ratio range
    print("\n🏆 Best Configuration per Compression Ratio:")
    compression_ranges = [
        (80, 100, "Ultra-High Compression"),
        (40, 50, "High Compression"),
        (20, 25, "Medium Compression"),
        (10, 12, "Low Compression"),
        (5, 6, "Minimal Compression")
    ]
    
    for min_ratio, max_ratio, description in compression_ranges:
        range_data = successful[
            (successful['compression_ratio'] >= min_ratio) & 
            (successful['compression_ratio'] <= max_ratio)
        ]
        if len(range_data) > 0:
            best_config = range_data.loc[range_data['final_psnr'].idxmax()]
            print(f"  {description:20s} ({min_ratio:2.0f}-{max_ratio:2.0f}:1): "
                  f"{best_config['final_psnr']:5.2f} dB - {best_config['arch_label']}")
    
    # Architecture performance vs compression efficiency
    print(f"\n📊 Architecture Efficiency Analysis:")
    arch_efficiency = successful.groupby('arch_label').agg({
        'final_psnr': ['mean', 'std'],
        'compression_ratio': ['mean', 'min', 'max'],
        'training_time': ['mean']
    }).round(2)
    
    print(f"{'Architecture':<25} {'Avg PSNR':<12} {'Compression Range':<18} {'Avg Time (min)':<15}")
    print("-" * 75)
    
    for arch in arch_efficiency.index:
        avg_psnr = arch_efficiency.loc[arch, ('final_psnr', 'mean')]
        min_compression = arch_efficiency.loc[arch, ('compression_ratio', 'min')]
        max_compression = arch_efficiency.loc[arch, ('compression_ratio', 'max')]
        avg_time = arch_efficiency.loc[arch, ('training_time', 'mean')] / 60
        
        compression_range = f"{min_compression:.1f}-{max_compression:.1f}:1"
        print(f"{arch:<25} {avg_psnr:<12.2f} {compression_range:<18} {avg_time:<15.1f}")
    
    # Quality vs Compression trade-off analysis
    print(f"\n🎯 Quality vs Compression Trade-off:")
    print("   Compression | PSNR Range | Best Architecture")
    print("   ------------|------------|------------------")
    
    for latent_dim in sorted(successful['latent_dim'].unique()):
        dim_data = successful[successful['latent_dim'] == latent_dim]
        compression = 343 / latent_dim
        min_psnr = dim_data['final_psnr'].min()
        max_psnr = dim_data['final_psnr'].max()
        best_arch = dim_data.loc[dim_data['final_psnr'].idxmax(), 'arch_label']
        
        print(f"   {compression:8.1f}:1   | {min_psnr:4.1f}-{max_psnr:4.1f} | {best_arch}")
    
    print("\n🎉 Compression ratio analysis plots completed!")

if __name__ == "__main__":
    create_compression_ratio_plots('consolidated_architecture_search_results.csv') 