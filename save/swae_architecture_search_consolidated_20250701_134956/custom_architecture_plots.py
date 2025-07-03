#!/usr/bin/env python3
"""
Custom Architecture Analysis Plots
Generates three specific plots with detailed architectural configuration legends
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_custom_plots(csv_path):
    print("🎨 Creating Custom Architecture Analysis Plots")
    print("=" * 50)
    
    # Load and prepare data
    df = pd.read_csv(csv_path)
    successful = df[df['status'] == 'SUCCESS'].copy()
    
    # Convert numeric columns
    successful['final_psnr'] = pd.to_numeric(successful['final_psnr'], errors='coerce')
    successful['training_time'] = pd.to_numeric(successful['training_time'], errors='coerce')
    
    # Remove rows with NaN PSNR or training time
    successful = successful.dropna(subset=['final_psnr', 'training_time'])
    
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
    
    # Plot 1: PSNR vs Latent Dimensions
    print("📊 Creating Plot 1: PSNR vs Latent Dimensions")
    
    for arch in arch_configs:
        arch_data = successful[successful['arch_label'] == arch]
        if len(arch_data) > 0:
            ax1.scatter(arch_data['latent_dim'], arch_data['final_psnr'], 
                       label=arch, **arch_style[arch])
    
    ax1.set_xlabel('Latent Dimensions', fontsize=12)
    ax1.set_ylabel('Final PSNR (dB)', fontsize=12)
    ax1.set_title('PSNR vs Latent Dimensions\nby Architecture Configuration', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
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
    
    # Plot 3: Training Time vs Latent Dimensions
    print("📊 Creating Plot 3: Training Time vs Latent Dimensions")
    
    for arch in arch_configs:
        arch_data = successful[successful['arch_label'] == arch]
        if len(arch_data) > 0:
            ax3.scatter(arch_data['latent_dim'], arch_data['training_time']/60, 
                       label=arch, **arch_style[arch])
    
    ax3.set_xlabel('Latent Dimensions', fontsize=12)
    ax3.set_ylabel('Training Time (minutes)', fontsize=12)
    ax3.set_title('Training Time vs Latent Dimensions\nby Architecture Configuration', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    # Save the plot
    plot_path = csv_path.replace('.csv', '_custom_architecture_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Custom plots saved to: {plot_path}")
    
    # Print summary statistics
    print("\n📈 Architecture Performance Summary:")
    print("=" * 50)
    
    # Best configuration per latent dimension
    print("\n🏆 Best Configuration per Latent Dimension:")
    for latent_dim in sorted(successful['latent_dim'].unique()):
        dim_data = successful[successful['latent_dim'] == latent_dim]
        if len(dim_data) > 0:
            best_config = dim_data.loc[dim_data['final_psnr'].idxmax()]
            compression_ratio = 343 / latent_dim
            print(f"  Latent {latent_dim:2d} ({compression_ratio:5.1f}:1): {best_config['final_psnr']:5.2f} dB - {best_config['arch_label']}")
    
    # Architecture performance comparison
    print(f"\n📊 Architecture Performance Comparison:")
    arch_stats = successful.groupby('arch_label').agg({
        'final_psnr': ['mean', 'std', 'count'],
        'training_time': ['mean', 'std']
    }).round(2)
    
    print(f"{'Architecture':<25} {'Avg PSNR':<12} {'Std PSNR':<10} {'Count':<8} {'Avg Time (min)':<15}")
    print("-" * 75)
    
    for arch in arch_stats.index:
        avg_psnr = arch_stats.loc[arch, ('final_psnr', 'mean')]
        std_psnr = arch_stats.loc[arch, ('final_psnr', 'std')]
        count = int(arch_stats.loc[arch, ('final_psnr', 'count')])
        avg_time = arch_stats.loc[arch, ('training_time', 'mean')] / 60
        print(f"{arch:<25} {avg_psnr:<12.2f} {std_psnr:<10.2f} {count:<8} {avg_time:<15.1f}")
    
    print("\n🎉 Custom architecture plots completed!")

if __name__ == "__main__":
    create_custom_plots('consolidated_architecture_search_results.csv') 