#!/usr/bin/env python3
"""
Analyze Multi-Latent Dimension SWAE Results
Visualizes compression ratio vs quality trade-offs
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import argparse

def load_results(csv_path):
    """Load results from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        # Filter successful results only
        successful = df[df['status'] == 'SUCCESS'].copy()
        
        # Convert numeric columns
        numeric_cols = ['latent_dim', 'compression_ratio', 'mse', 'psnr', 'mae', 
                       'correlation', 'mean_relative_error_percent', 
                       'rms_relative_error_percent', 'p95_relative_error_percent',
                       'compression_speed_gbps', 'decompression_speed_gbps',
                       'compression_speed_mbps', 'decompression_speed_mbps',
                       'samples_per_sec_comp', 'samples_per_sec_decomp']
        
        for col in numeric_cols:
            if col in successful.columns:
                successful[col] = pd.to_numeric(successful[col], errors='coerce')
        
        # Sort by latent dimension
        successful = successful.sort_values('latent_dim')
        
        return successful
        
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def create_plots(df, output_dir):
    """Create visualization plots"""
    
    if df is None or len(df) == 0:
        print("No data to plot")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    fig_size = (12, 8)
    
    # 1. Compression Ratio vs PSNR
    plt.figure(figsize=fig_size)
    plt.plot(df['compression_ratio'], df['psnr'], 'bo-', linewidth=2, markersize=8)
    for i, row in df.iterrows():
        plt.annotate(f'{int(row["latent_dim"])}', 
                    (row['compression_ratio'], row['psnr']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    plt.xlabel('Compression Ratio (X:1)')
    plt.ylabel('PSNR (dB)')
    plt.title('Compression vs Quality Trade-off\n(Higher PSNR is better)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compression_vs_psnr.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'compression_vs_psnr.pdf'), bbox_inches='tight')
    print(f"Saved: compression_vs_psnr.png/pdf")
    
    # 2. Compression Ratio vs Relative Error
    plt.figure(figsize=fig_size)
    plt.plot(df['compression_ratio'], df['rms_relative_error_percent'], 'ro-', linewidth=2, markersize=8, label='RMS Error')
    plt.plot(df['compression_ratio'], df['mean_relative_error_percent'], 'go-', linewidth=2, markersize=8, label='Mean Error')
    plt.plot(df['compression_ratio'], df['p95_relative_error_percent'], 'mo-', linewidth=2, markersize=8, label='P95 Error')
    
    for i, row in df.iterrows():
        plt.annotate(f'{int(row["latent_dim"])}', 
                    (row['compression_ratio'], row['rms_relative_error_percent']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Compression Ratio (X:1)')
    plt.ylabel('Relative Error (%)')
    plt.title('Compression vs Relative Error\n(Lower error is better)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compression_vs_error.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'compression_vs_error.pdf'), bbox_inches='tight')
    print(f"Saved: compression_vs_error.png/pdf")
    
    # 3. Compression vs Speed Trade-off
    plt.figure(figsize=fig_size)
    if 'compression_speed_mbps' in df.columns and not df['compression_speed_mbps'].isna().all():
        plt.plot(df['compression_ratio'], df['compression_speed_mbps'], 'go-', linewidth=2, markersize=8, label='Compression')
        plt.plot(df['compression_ratio'], df['decompression_speed_mbps'], 'ro-', linewidth=2, markersize=8, label='Decompression')
        
        for i, row in df.iterrows():
            plt.annotate(f'{int(row["latent_dim"])}', 
                        (row['compression_ratio'], row['compression_speed_mbps']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.xlabel('Compression Ratio (X:1)')
        plt.ylabel('Speed (MBps)')
        plt.title('Compression Ratio vs Inference Speed\n(Higher speed is better)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'compression_vs_speed.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'compression_vs_speed.pdf'), bbox_inches='tight')
        print(f"Saved: compression_vs_speed.png/pdf")
    
    # 4. Samples per Second Analysis
    plt.figure(figsize=fig_size)
    if 'samples_per_sec_comp' in df.columns and not df['samples_per_sec_comp'].isna().all():
        plt.plot(df['compression_ratio'], df['samples_per_sec_comp'], 'co-', linewidth=2, markersize=8, label='Compression')
        plt.plot(df['compression_ratio'], df['samples_per_sec_decomp'], 'mo-', linewidth=2, markersize=8, label='Decompression')
        
        for i, row in df.iterrows():
            plt.annotate(f'{int(row["latent_dim"])}', 
                        (row['compression_ratio'], row['samples_per_sec_comp']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.xlabel('Compression Ratio (X:1)')
        plt.ylabel('Samples per Second')
        plt.title('Compression Ratio vs Throughput\n(Higher throughput is better)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'compression_vs_throughput.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'compression_vs_throughput.pdf'), bbox_inches='tight')
        print(f"Saved: compression_vs_throughput.png/pdf")
    
    # 5. Latent Dimension vs Multiple Metrics (subplot)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # PSNR vs Latent Dim
    axes[0,0].plot(df['latent_dim'], df['psnr'], 'bo-', linewidth=2, markersize=8)
    axes[0,0].set_xlabel('Latent Dimensions')
    axes[0,0].set_ylabel('PSNR (dB)')
    axes[0,0].set_title('PSNR vs Latent Dimensions')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xscale('log', base=2)
    
    # Correlation vs Latent Dim
    axes[0,1].plot(df['latent_dim'], df['correlation'], 'go-', linewidth=2, markersize=8)
    axes[0,1].set_xlabel('Latent Dimensions')
    axes[0,1].set_ylabel('Correlation')
    axes[0,1].set_title('Correlation vs Latent Dimensions')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_xscale('log', base=2)
    
    # MSE vs Latent Dim
    axes[1,0].plot(df['latent_dim'], df['mse'], 'ro-', linewidth=2, markersize=8)
    axes[1,0].set_xlabel('Latent Dimensions')
    axes[1,0].set_ylabel('MSE')
    axes[1,0].set_title('MSE vs Latent Dimensions')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_yscale('log')
    axes[1,0].set_xscale('log', base=2)
    
    # RMS Error vs Latent Dim
    axes[1,1].plot(df['latent_dim'], df['rms_relative_error_percent'], 'mo-', linewidth=2, markersize=8)
    axes[1,1].set_xlabel('Latent Dimensions')
    axes[1,1].set_ylabel('RMS Relative Error (%)')
    axes[1,1].set_title('RMS Error vs Latent Dimensions')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_yscale('log')
    axes[1,1].set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_vs_latent_dims.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'metrics_vs_latent_dims.pdf'), bbox_inches='tight')
    print(f"Saved: metrics_vs_latent_dims.png/pdf")
    
    # 6. Summary Table Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table
    table_data = []
    for _, row in df.iterrows():
        comp_speed = f"{row['compression_speed_mbps']:.0f}" if 'compression_speed_mbps' in row and pd.notna(row['compression_speed_mbps']) else 'N/A'
        decomp_speed = f"{row['decompression_speed_mbps']:.0f}" if 'decompression_speed_mbps' in row and pd.notna(row['decompression_speed_mbps']) else 'N/A'
        table_data.append([
            f"{int(row['latent_dim'])}",
            f"{row['compression_ratio']:.1f}:1",
            f"{row['psnr']:.1f} dB",
            f"{row['correlation']:.4f}",
            f"{row['mse']:.2e}",
            f"{row['rms_relative_error_percent']:.2f}%",
            f"{comp_speed} MBps",
            f"{decomp_speed} MBps"
        ])
    
    headers = ['Latent Dim', 'Compression', 'PSNR', 'Correlation', 'MSE', 'RMS Error', 'Comp Speed', 'Decomp Speed']
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('SWAE Compression Results Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'results_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'results_summary_table.pdf'), bbox_inches='tight')
    print(f"Saved: results_summary_table.png/pdf")

def print_analysis(df):
    """Print detailed analysis"""
    
    if df is None or len(df) == 0:
        print("No successful results to analyze")
        return
    
    print("\n" + "="*80)
    print("SWAE COMPRESSION ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nSuccessfully analyzed {len(df)} models:")
    print(f"Latent dimensions tested: {list(df['latent_dim'].astype(int))}")
    print(f"Compression ratios: {[f'{x:.1f}:1' for x in df['compression_ratio']]}")
    
    print(f"\nQuality Metrics Range:")
    print(f"  PSNR: {df['psnr'].min():.1f} - {df['psnr'].max():.1f} dB")
    print(f"  Correlation: {df['correlation'].min():.4f} - {df['correlation'].max():.4f}")
    print(f"  RMS Error: {df['rms_relative_error_percent'].min():.2f}% - {df['rms_relative_error_percent'].max():.2f}%")
    
    if 'compression_speed_mbps' in df.columns and not df['compression_speed_mbps'].isna().all():
        print(f"\nSpeed Metrics Range:")
        print(f"  Compression Speed: {df['compression_speed_mbps'].min():.0f} - {df['compression_speed_mbps'].max():.0f} MBps")
        print(f"  Decompression Speed: {df['decompression_speed_mbps'].min():.0f} - {df['decompression_speed_mbps'].max():.0f} MBps")
        if 'samples_per_sec_comp' in df.columns:
            print(f"  Compression Throughput: {df['samples_per_sec_comp'].min():.0f} - {df['samples_per_sec_comp'].max():.0f} samples/sec")
            print(f"  Decompression Throughput: {df['samples_per_sec_decomp'].min():.0f} - {df['samples_per_sec_decomp'].max():.0f} samples/sec")
    
    # Find best trade-offs
    best_quality = df.loc[df['psnr'].idxmax()]
    best_compression = df.loc[df['compression_ratio'].idxmax()]
    lowest_error = df.loc[df['rms_relative_error_percent'].idxmin()]
    
    print(f"\nKey Findings:")
    print(f"  Best Quality: {int(best_quality['latent_dim'])} dims → {best_quality['psnr']:.1f} dB, {best_quality['compression_ratio']:.1f}:1")
    print(f"  Best Compression: {int(best_compression['latent_dim'])} dims → {best_compression['compression_ratio']:.1f}:1, {best_compression['psnr']:.1f} dB")
    print(f"  Lowest Error: {int(lowest_error['latent_dim'])} dims → {lowest_error['rms_relative_error_percent']:.2f}% RMS")
    
    # Speed analysis
    if 'compression_speed_mbps' in df.columns and not df['compression_speed_mbps'].isna().all():
        fastest_comp = df.loc[df['compression_speed_mbps'].idxmax()]
        fastest_decomp = df.loc[df['decompression_speed_mbps'].idxmax()]
        print(f"  Fastest Compression: {int(fastest_comp['latent_dim'])} dims → {fastest_comp['compression_speed_mbps']:.0f} MBps")
        print(f"  Fastest Decompression: {int(fastest_decomp['latent_dim'])} dims → {fastest_decomp['decompression_speed_mbps']:.0f} MBps")
    
    # Efficiency analysis
    df['efficiency'] = df['psnr'] / np.log10(df['compression_ratio'])  # PSNR per log compression
    best_efficiency = df.loc[df['efficiency'].idxmax()]
    print(f"  Best Efficiency: {int(best_efficiency['latent_dim'])} dims → {best_efficiency['psnr']:.1f} dB at {best_efficiency['compression_ratio']:.1f}:1")
    
    print(f"\nDetailed Results:")
    print("-" * 110)
    has_speed = 'compression_speed_mbps' in df.columns and not df['compression_speed_mbps'].isna().all()
    
    if has_speed:
        print(f"{'Latent':>7} {'Compress':>10} {'PSNR':>8} {'Corr':>8} {'RMS Err':>10} {'Comp Speed':>12} {'Decomp Speed':>14} {'Quality':>12}")
        print("-" * 110)
        
        for _, row in df.iterrows():
            quality = "Excellent" if row['psnr'] > 30 else "Good" if row['psnr'] > 25 else "Fair"
            comp_speed = f"{row['compression_speed_mbps']:.0f} MBps" if pd.notna(row['compression_speed_mbps']) else "N/A"
            decomp_speed = f"{row['decompression_speed_mbps']:.0f} MBps" if pd.notna(row['decompression_speed_mbps']) else "N/A"
            print(f"{int(row['latent_dim']):>7} {row['compression_ratio']:>8.1f}:1 "
                  f"{row['psnr']:>6.1f} dB {row['correlation']:>7.4f} "
                  f"{row['rms_relative_error_percent']:>8.2f}% {comp_speed:>12} {decomp_speed:>14} {quality:>12}")
    else:
        print(f"{'Latent':>7} {'Compress':>10} {'PSNR':>8} {'Corr':>8} {'RMS Err':>10} {'Model Quality':>15}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            quality = "Excellent" if row['psnr'] > 30 else "Good" if row['psnr'] > 25 else "Fair"
            print(f"{int(row['latent_dim']):>7} {row['compression_ratio']:>8.1f}:1 "
                  f"{row['psnr']:>6.1f} dB {row['correlation']:>7.4f} "
                  f"{row['rms_relative_error_percent']:>8.2f}% {quality:>15}")

def main():
    parser = argparse.ArgumentParser(description='Analyze SWAE multi-latent dimension results')
    parser.add_argument('--csv-file', type=str, 
                        help='Path to CSV results file (if not provided, will search for latest)')
    parser.add_argument('--output-dir', type=str, default='analysis_plots',
                        help='Directory to save plots')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip plot generation, only print analysis')
    
    args = parser.parse_args()
    
    # Find CSV file if not provided
    if args.csv_file is None:
        # Look for the most recent inference results
        pattern = "inference_multi_latent_results_*/latent_dim_comparison.csv"
        csv_files = glob.glob(pattern)
        if csv_files:
            args.csv_file = max(csv_files, key=os.path.getctime)
            print(f"Found results file: {args.csv_file}")
        else:
            print("No results CSV file found. Please run inference first or specify --csv-file")
            return
    
    # Load and analyze results
    df = load_results(args.csv_file)
    
    if df is None:
        print("Failed to load results")
        return
    
    # Print analysis
    print_analysis(df)
    
    # Create plots
    if not args.no_plots:
        print(f"\nGenerating plots in: {args.output_dir}")
        create_plots(df, args.output_dir)
        print(f"\nAll plots saved to: {args.output_dir}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main() 