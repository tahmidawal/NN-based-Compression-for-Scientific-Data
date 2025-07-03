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
