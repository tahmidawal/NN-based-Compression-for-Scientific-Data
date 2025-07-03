#!/usr/bin/env python3
"""
Analyze Inference Speeds from Architecture Search
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_inference_speeds(csv_path):
    # Load results
    df = pd.read_csv(csv_path)
    successful = df[df['speed_test_status'] == 'SUCCESS'].copy()
    
    if len(successful) == 0:
        print("No successful speed tests found!")
        return
    
    # Convert numeric columns
    numeric_cols = ['latent_dim', 'final_psnr', 'inference_time_sec', 
                   'compression_speed_mbps', 'decompression_speed_mbps',
                   'compression_throughput_samples_sec', 'decompression_throughput_samples_sec']
    
    for col in numeric_cols:
        if col in successful.columns:
            successful[col] = pd.to_numeric(successful[col], errors='coerce')
    
    # Calculate compression ratio numerically
    successful['compression_ratio_num'] = 343 / successful['latent_dim']
    
    print(f"Inference Speed Analysis")
    print(f"=======================")
    print(f"Total configurations: {len(df)}")
    print(f"Successful speed tests: {len(successful)}")
    print(f"Success rate: {len(successful)/len(df)*100:.1f}%")
    
    # Speed statistics
    print(f"\n📊 Speed Statistics:")
    print(f"=" * 40)
    print(f"Compression Speed:")
    print(f"  Mean: {successful['compression_speed_mbps'].mean():.1f} MBps")
    print(f"  Median: {successful['compression_speed_mbps'].median():.1f} MBps")
    print(f"  Range: {successful['compression_speed_mbps'].min():.1f} - {successful['compression_speed_mbps'].max():.1f} MBps")
    
    print(f"\nDecompression Speed:")
    print(f"  Mean: {successful['decompression_speed_mbps'].mean():.1f} MBps")
    print(f"  Median: {successful['decompression_speed_mbps'].median():.1f} MBps")
    print(f"  Range: {successful['decompression_speed_mbps'].min():.1f} - {successful['decompression_speed_mbps'].max():.1f} MBps")
    
    print(f"\nInference Time:")
    print(f"  Mean: {successful['inference_time_sec'].mean():.2f} seconds")
    print(f"  Median: {successful['inference_time_sec'].median():.2f} seconds")
    print(f"  Range: {successful['inference_time_sec'].min():.2f} - {successful['inference_time_sec'].max():.2f} seconds")
    
    # Find optimal configurations
    print(f"\n🏆 Top Performing Configurations:")
    print(f"=" * 50)
    
    # Fastest compression
    fastest_comp = successful.loc[successful['compression_speed_mbps'].idxmax()]
    print(f"\n🚀 Fastest Compression:")
    print(f"   Speed: {fastest_comp['compression_speed_mbps']:.1f} MBps")
    print(f"   Model: {fastest_comp['latent_dim']}D {fastest_comp['arch_config']} ({fastest_comp['compression_ratio_num']:.1f}:1)")
    print(f"   Quality: {fastest_comp['final_psnr']:.2f} dB PSNR")
    
    # Fastest decompression
    fastest_decomp = successful.loc[successful['decompression_speed_mbps'].idxmax()]
    print(f"\n⚡ Fastest Decompression:")
    print(f"   Speed: {fastest_decomp['decompression_speed_mbps']:.1f} MBps")
    print(f"   Model: {fastest_decomp['latent_dim']}D {fastest_decomp['arch_config']} ({fastest_decomp['compression_ratio_num']:.1f}:1)")
    print(f"   Quality: {fastest_decomp['final_psnr']:.2f} dB PSNR")
    
    # Best speed/quality trade-off
    successful['speed_quality_score'] = (successful['compression_speed_mbps'] * successful['final_psnr']) / 1000
    best_tradeoff = successful.loc[successful['speed_quality_score'].idxmax()]
    print(f"\n⚖️ Best Speed/Quality Trade-off:")
    print(f"   Score: {best_tradeoff['speed_quality_score']:.2f}")
    print(f"   Speed: {best_tradeoff['compression_speed_mbps']:.1f} MBps")
    print(f"   Quality: {best_tradeoff['final_psnr']:.2f} dB PSNR")
    print(f"   Model: {best_tradeoff['latent_dim']}D {best_tradeoff['arch_config']} ({best_tradeoff['compression_ratio_num']:.1f}:1)")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Compression Speed vs Quality
    axes[0,0].scatter(successful['compression_speed_mbps'], successful['final_psnr'], 
                     c=successful['latent_dim'], cmap='viridis', s=60, alpha=0.7)
    axes[0,0].set_xlabel('Compression Speed (MBps)')
    axes[0,0].set_ylabel('PSNR (dB)')
    axes[0,0].set_title('Compression Speed vs Quality')
    
    # 2. Decompression Speed vs Quality
    axes[0,1].scatter(successful['decompression_speed_mbps'], successful['final_psnr'],
                     c=successful['latent_dim'], cmap='viridis', s=60, alpha=0.7)
    axes[0,1].set_xlabel('Decompression Speed (MBps)')
    axes[0,1].set_ylabel('PSNR (dB)')
    axes[0,1].set_title('Decompression Speed vs Quality')
    
    # 3. Speed vs Compression Ratio
    axes[0,2].scatter(successful['compression_ratio_num'], successful['compression_speed_mbps'],
                     c=successful['latent_dim'], cmap='viridis', s=60, alpha=0.7)
    axes[0,2].set_xlabel('Compression Ratio')
    axes[0,2].set_ylabel('Compression Speed (MBps)')
    axes[0,2].set_title('Speed vs Compression Ratio')
    
    # 4. Architecture Performance - Speed
    arch_speed = successful.groupby('arch_config')['compression_speed_mbps'].agg(['mean', 'std']).reset_index()
    axes[1,0].bar(arch_speed['arch_config'], arch_speed['mean'], 
                 yerr=arch_speed['std'], capsize=5)
    axes[1,0].set_xlabel('Architecture')
    axes[1,0].set_ylabel('Compression Speed (MBps)')
    axes[1,0].set_title('Architecture Speed Comparison')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 5. Latent Dimension vs Speed
    latent_speed = successful.groupby('latent_dim').agg({
        'compression_speed_mbps': ['mean', 'std'],
        'decompression_speed_mbps': 'mean'
    }).reset_index()
    
    axes[1,1].errorbar(latent_speed['latent_dim'], 
                      latent_speed['compression_speed_mbps']['mean'],
                      yerr=latent_speed['compression_speed_mbps']['std'],
                      marker='o', capsize=5, capthick=2, label='Compression')
    axes[1,1].plot(latent_speed['latent_dim'], 
                   latent_speed['decompression_speed_mbps']['mean'],
                   marker='s', label='Decompression')
    axes[1,1].set_xlabel('Latent Dimension')
    axes[1,1].set_ylabel('Speed (MBps)')
    axes[1,1].set_title('Speed vs Latent Dimension')
    axes[1,1].legend()
    
    # 6. Speed Distribution
    axes[1,2].hist(successful['compression_speed_mbps'], bins=15, alpha=0.6, 
                   label='Compression', edgecolor='black')
    axes[1,2].hist(successful['decompression_speed_mbps'], bins=15, alpha=0.6, 
                   label='Decompression', edgecolor='black')
    axes[1,2].set_xlabel('Speed (MBps)')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].set_title('Speed Distribution')
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.savefig(csv_path.replace('.csv', '_speed_analysis.png'), 
                dpi=300, bbox_inches='tight')
    print(f"\n📊 Speed analysis plots saved!")
    
    # Save summary statistics
    summary_stats = pd.DataFrame({
        'Metric': ['Compression Speed (MBps)', 'Decompression Speed (MBps)', 'Inference Time (sec)'],
        'Mean': [successful['compression_speed_mbps'].mean(),
                successful['decompression_speed_mbps'].mean(),
                successful['inference_time_sec'].mean()],
        'Median': [successful['compression_speed_mbps'].median(),
                  successful['decompression_speed_mbps'].median(),
                  successful['inference_time_sec'].median()],
        'Min': [successful['compression_speed_mbps'].min(),
               successful['decompression_speed_mbps'].min(),
               successful['inference_time_sec'].min()],
        'Max': [successful['compression_speed_mbps'].max(),
               successful['decompression_speed_mbps'].max(),
               successful['inference_time_sec'].max()]
    })
    
    summary_stats.to_csv(csv_path.replace('.csv', '_speed_summary.csv'), index=False)
    print(f"📁 Speed summary saved!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_inference_speeds(sys.argv[1])
    else:
        analyze_inference_speeds('consolidated_architecture_search_results_with_inference_speeds.csv')
