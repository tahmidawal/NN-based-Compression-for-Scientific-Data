#!/usr/bin/env python
"""
Analyze magnitude distribution of all variables from BSSN dataset.
Shows how many samples fall into different magnitude ranges.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm
import pandas as pd

def poslog_transform(data, epsilon=1e-6):
    """Apply positive-shift log transformation (poslog) with new epsilon."""
    data_min = data.min()
    if data_min <= 0:
        data_shifted = data - data_min + epsilon
    else:
        data_shifted = data.copy()
    return np.log(data_shifted + epsilon)

def load_all_variables(data_folder, exclude_vars=['U_CHI']):
    """Load all variables from HDF5 files (except excluded ones)."""
    file_pattern = os.path.join(data_folder, "bssn_gr_*_extracted.hdf5")
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        raise ValueError(f"No HDF5 files found in {data_folder}")
    
    print(f"Found {len(files)} HDF5 files")
    
    # First, get the list of variables from the first file
    with h5py.File(files[0], 'r') as f:
        all_vars = list(f['vars'][:].astype(str))
        u_vars = [var for var in all_vars if var.startswith('U_') and var not in exclude_vars]
        print(f"\nU_ variables to analyze: {u_vars}")
    
    # Initialize dictionary to store data for each variable
    variable_data = {var: [] for var in u_vars}
    
    # Load data from all files
    for file_path in tqdm(files[:10], desc="Loading files"):  # Limit to first 10 files for faster analysis
        with h5py.File(file_path, 'r') as f:
            vars_list = f['vars'][:].astype(str)
            var_data = f['var_data'][:]  # Shape: (n_vars, n_blocks, 7, 7, 7)
            
            for var in u_vars:
                if var in vars_list:
                    var_idx = list(vars_list).index(var)
                    # Extract 5x5x5 center crop from 7x7x7 data
                    data_7x7x7 = var_data[var_idx]  # Shape: (n_blocks, 7, 7, 7)
                    data_5x5x5 = data_7x7x7[:, 1:6, 1:6, 1:6]  # Center crop
                    variable_data[var].append(data_5x5x5)
    
    # Concatenate all data for each variable
    for var in u_vars:
        if variable_data[var]:
            variable_data[var] = np.concatenate(variable_data[var], axis=0)
            print(f"{var}: loaded {variable_data[var].shape[0]} blocks")
        else:
            print(f"Warning: No data found for {var}")
    
    return variable_data

def analyze_magnitude_distribution(data, var_name="Data"):
    """Analyze the magnitude distribution of data."""
    # Flatten the data
    data_flat = data.flatten()
    
    # Calculate absolute values for magnitude
    data_abs = np.abs(data_flat)
    
    # Define magnitude bins (powers of 10)
    magnitude_bins = np.logspace(-20, 5, 26)  # From 1e-20 to 1e5
    
    # Count samples in each magnitude range
    hist, bin_edges = np.histogram(data_abs, bins=magnitude_bins)
    
    # Create a more detailed analysis with custom bins for small values
    small_value_threshold = 1e-6
    tiny_value_threshold = 1e-9
    
    # Analyze magnitude range within each 5x5x5 block
    n_blocks = data.shape[0]
    block_magnitude_ranges = []
    block_min_magnitudes = []
    block_max_magnitudes = []
    
    for i in range(n_blocks):
        block = data[i].flatten()
        block_abs = np.abs(block)
        block_min = np.min(block_abs)
        block_max = np.max(block_abs)
        
        # Calculate magnitude range (difference between max and min magnitude)
        if block_max > 0 and block_min > 0:
            magnitude_range = block_max / block_min  # Ratio for better representation
        else:
            magnitude_range = block_max - block_min  # Fallback to difference
        
        block_magnitude_ranges.append(magnitude_range)
        block_min_magnitudes.append(block_min)
        block_max_magnitudes.append(block_max)
    
    stats = {
        'var_name': var_name,
        'total_samples': len(data_flat),
        'n_blocks': n_blocks,
        'zero_samples': np.sum(data_abs == 0),
        'tiny_samples': np.sum((data_abs > 0) & (data_abs < tiny_value_threshold)),
        'small_samples': np.sum((data_abs >= tiny_value_threshold) & (data_abs < small_value_threshold)),
        'normal_samples': np.sum(data_abs >= small_value_threshold),
        'min_nonzero': np.min(data_abs[data_abs > 0]) if np.any(data_abs > 0) else np.nan,
        'max_value': np.max(data_abs),
        'mean_value': np.mean(data_flat),
        'std_value': np.std(data_flat),
        'histogram': hist,
        'bin_edges': bin_edges,
        'block_magnitude_ranges': np.array(block_magnitude_ranges),
        'block_min_magnitudes': np.array(block_min_magnitudes),
        'block_max_magnitudes': np.array(block_max_magnitudes)
    }
    
    return stats

def plot_magnitude_distributions(all_stats, output_dir="magnitude_distributions"):
    """Plot magnitude distributions for all variables."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create color map for variables
    n_vars = len(all_stats)
    colors = plt.cm.tab20(np.linspace(0, 1, n_vars))
    
    # 1. Combined plot - Magnitude histogram
    plt.figure(figsize=(15, 8))
    
    for idx, stats in enumerate(all_stats):
        var_name = stats['var_name']
        hist = stats['histogram']
        bin_edges = stats['bin_edges']
        
        # Plot on log-log scale
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # Geometric mean for log scale
        mask = hist > 0  # Only plot non-zero counts
        
        plt.loglog(bin_centers[mask], hist[mask], 'o-', 
                  label=var_name, color=colors[idx], linewidth=2, markersize=6)
    
    plt.xlabel('Magnitude (absolute value)', fontsize=12)
    plt.ylabel('Number of samples', fontsize=12)
    plt.title('Magnitude Distribution of All Variables (Log-Log Scale)', fontsize=14)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_variables_magnitude_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Stacked bar chart showing proportion of tiny/small/normal values
    fig, ax = plt.subplots(figsize=(12, 8))
    
    var_names = [stats['var_name'] for stats in all_stats]
    tiny_props = [stats['tiny_samples'] / stats['total_samples'] * 100 for stats in all_stats]
    small_props = [stats['small_samples'] / stats['total_samples'] * 100 for stats in all_stats]
    normal_props = [stats['normal_samples'] / stats['total_samples'] * 100 for stats in all_stats]
    zero_props = [stats['zero_samples'] / stats['total_samples'] * 100 for stats in all_stats]
    
    x = np.arange(len(var_names))
    width = 0.6
    
    p1 = ax.bar(x, zero_props, width, label='Zero (= 0)', color='red')
    p2 = ax.bar(x, tiny_props, width, bottom=zero_props, label='Tiny (< 1e-9)', color='orange')
    p3 = ax.bar(x, small_props, width, bottom=np.array(zero_props)+np.array(tiny_props), 
                label='Small (1e-9 to 1e-6)', color='yellow')
    p4 = ax.bar(x, normal_props, width, 
                bottom=np.array(zero_props)+np.array(tiny_props)+np.array(small_props), 
                label='Normal (≥ 1e-6)', color='green')
    
    ax.set_ylabel('Percentage of samples (%)', fontsize=12)
    ax.set_title('Distribution of Sample Magnitudes by Variable', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(var_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'magnitude_proportions_stacked.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Plot magnitude range distributions for all variables combined
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Magnitude range distribution
    plt.subplot(2, 2, 1)
    for idx, stats in enumerate(all_stats[:-1]):  # Exclude "All Variables Combined"
        var_name = stats['var_name']
        ranges = stats['block_magnitude_ranges']
        
        # Plot histogram of log10(magnitude ranges)
        log_ranges = np.log10(ranges[ranges > 0])
        plt.hist(log_ranges, bins=50, alpha=0.5, label=var_name, density=True)
    
    plt.xlabel('log10(Max/Min Magnitude per Block)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Magnitude Ranges within 5x5x5 Blocks', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Box plot of magnitude ranges
    plt.subplot(2, 2, 2)
    magnitude_range_data = []
    labels = []
    for stats in all_stats[:-1]:  # Exclude "All Variables Combined"
        ranges = stats['block_magnitude_ranges']
        magnitude_range_data.append(np.log10(ranges[ranges > 0]))
        labels.append(stats['var_name'])
    
    plt.boxplot(magnitude_range_data, labels=labels)
    plt.ylabel('log10(Max/Min Magnitude per Block)', fontsize=12)
    plt.title('Magnitude Range Distribution by Variable', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Scatter plot of min vs max magnitudes
    plt.subplot(2, 2, 3)
    for idx, stats in enumerate(all_stats[:-1]):
        var_name = stats['var_name']
        mins = stats['block_min_magnitudes']
        maxs = stats['block_max_magnitudes']
        
        # Sample points to avoid overplotting
        n_samples = min(1000, len(mins))
        indices = np.random.choice(len(mins), n_samples, replace=False)
        
        plt.scatter(mins[indices], maxs[indices], alpha=0.5, label=var_name, s=10)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Min Magnitude per Block', fontsize=12)
    plt.ylabel('Max Magnitude per Block', fontsize=12)
    plt.title('Min vs Max Magnitude per 5x5x5 Block', fontsize=14)
    plt.plot([1e-20, 1e5], [1e-20, 1e5], 'k--', alpha=0.3, label='Equal min/max')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    # Subplot 4: Histogram of blocks with extreme ranges
    plt.subplot(2, 2, 4)
    extreme_threshold = 1e6  # Blocks with magnitude range > 1e6
    extreme_counts = []
    for stats in all_stats[:-1]:
        ranges = stats['block_magnitude_ranges']
        extreme_count = np.sum(ranges > extreme_threshold)
        extreme_pct = extreme_count / len(ranges) * 100
        extreme_counts.append(extreme_pct)
    
    plt.bar(range(len(labels)), extreme_counts)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.ylabel('Percentage of blocks (%)', fontsize=12)
    plt.title(f'Blocks with Extreme Magnitude Range (> 1e6)', fontsize=14)
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'magnitude_range_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Individual plots for each variable
    for stats in all_stats:
        var_name = stats['var_name']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Subplot 1: Magnitude histogram (log-log)
        ax = axes[0, 0]
        hist = stats['histogram']
        bin_edges = stats['bin_edges']
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        mask = hist > 0
        
        ax.loglog(bin_centers[mask], hist[mask], 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Magnitude (absolute value)')
        ax.set_ylabel('Number of samples')
        ax.set_title(f'{var_name} - Magnitude Distribution (Log-Log Scale)')
        ax.grid(True, which="both", alpha=0.3)
        
        # Subplot 2: Pie chart of magnitude categories
        ax = axes[0, 1]
        sizes = [stats['zero_samples'], stats['tiny_samples'], 
                stats['small_samples'], stats['normal_samples']]
        labels = ['Zero\n(= 0)', 'Tiny\n(< 1e-9)', 'Small\n(1e-9 to 1e-6)', 'Normal\n(≥ 1e-6)']
        colors_pie = ['red', 'orange', 'yellow', 'green']
        
        # Only show non-zero categories
        mask = np.array(sizes) > 0
        if np.any(mask):
            ax.pie(np.array(sizes)[mask], labels=np.array(labels)[mask], 
                   colors=np.array(colors_pie)[mask], autopct='%1.1f%%', startangle=90)
            ax.set_title(f'{var_name} - Sample Categories')
        
        # Subplot 3: Magnitude range histogram
        ax = axes[0, 2]
        if 'block_magnitude_ranges' in stats and len(stats['block_magnitude_ranges']) > 0:
            ranges = stats['block_magnitude_ranges']
            log_ranges = np.log10(ranges[ranges > 0])
            ax.hist(log_ranges, bins=50, alpha=0.7, color='purple')
            ax.set_xlabel('log10(Max/Min Magnitude per Block)')
            ax.set_ylabel('Number of blocks')
            ax.set_title(f'{var_name} - Magnitude Range Distribution')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            range_stats = f'Mean: {np.mean(log_ranges):.2f}\nStd: {np.std(log_ranges):.2f}\nMedian: {np.median(log_ranges):.2f}'
            ax.text(0.02, 0.98, range_stats, transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5))
        
        # Subplot 4: Statistics text
        ax = axes[1, 0]
        ax.axis('off')
        stats_text = f"""
        {var_name} Statistics:
        
        Total samples: {stats['total_samples']:,}
        Number of blocks: {stats.get('n_blocks', 'N/A'):,}
        
        Zero values: {stats['zero_samples']:,} ({stats['zero_samples']/stats['total_samples']*100:.2f}%)
        Tiny values (< 1e-9): {stats['tiny_samples']:,} ({stats['tiny_samples']/stats['total_samples']*100:.2f}%)
        Small values (1e-9 to 1e-6): {stats['small_samples']:,} ({stats['small_samples']/stats['total_samples']*100:.2f}%)
        Normal values (≥ 1e-6): {stats['normal_samples']:,} ({stats['normal_samples']/stats['total_samples']*100:.2f}%)
        
        Min non-zero value: {stats['min_nonzero']:.2e}
        Max value: {stats['max_value']:.2e}
        Mean: {stats['mean_value']:.2e}
        Std: {stats['std_value']:.2e}
        """
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10, family='monospace')
        
        # Subplot 5: Cumulative distribution
        ax = axes[1, 1]
        data_abs_sorted = np.sort(stats['histogram'].cumsum())
        ax.semilogx(bin_centers, stats['histogram'].cumsum() / stats['total_samples'] * 100)
        ax.set_xlabel('Magnitude (absolute value)')
        ax.set_ylabel('Cumulative percentage (%)')
        ax.set_title(f'{var_name} - Cumulative Distribution')
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylim(0, 100)
        
        # Subplot 6: Min vs Max scatter plot for this variable
        ax = axes[1, 2]
        if 'block_min_magnitudes' in stats and 'block_max_magnitudes' in stats:
            mins = stats['block_min_magnitudes']
            maxs = stats['block_max_magnitudes']
            
            # Sample points to avoid overplotting
            n_samples = min(1000, len(mins))
            if len(mins) > 0:
                indices = np.random.choice(len(mins), n_samples, replace=False)
                ax.scatter(mins[indices], maxs[indices], alpha=0.5, s=10, color='orange')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel('Min Magnitude per Block')
                ax.set_ylabel('Max Magnitude per Block')
                ax.set_title(f'{var_name} - Min vs Max per Block')
                ax.plot([1e-20, 1e5], [1e-20, 1e5], 'k--', alpha=0.3)
                ax.grid(True, which="both", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{var_name}_magnitude_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Create summary table
    summary_data = []
    for stats in all_stats:
        # Calculate magnitude range statistics
        if 'block_magnitude_ranges' in stats and len(stats['block_magnitude_ranges']) > 0:
            ranges = stats['block_magnitude_ranges']
            log_ranges = np.log10(ranges[ranges > 0])
            mean_range = np.mean(log_ranges) if len(log_ranges) > 0 else 0
            extreme_blocks = np.sum(ranges > 1e6) / len(ranges) * 100 if len(ranges) > 0 else 0
        else:
            mean_range = 0
            extreme_blocks = 0
        
        summary_data.append({
            'Variable': stats['var_name'],
            'Total Samples': f"{stats['total_samples']:,}",
            'Blocks': f"{stats.get('n_blocks', 0):,}",
            'Zero (%)': f"{stats['zero_samples']/stats['total_samples']*100:.2f}",
            'Tiny < 1e-9 (%)': f"{stats['tiny_samples']/stats['total_samples']*100:.2f}",
            'Small 1e-9 to 1e-6 (%)': f"{stats['small_samples']/stats['total_samples']*100:.2f}",
            'Normal ≥ 1e-6 (%)': f"{stats['normal_samples']/stats['total_samples']*100:.2f}",
            'Min Non-Zero': f"{stats['min_nonzero']:.2e}",
            'Max': f"{stats['max_value']:.2e}",
            'Mean log10(Range)': f"{mean_range:.2f}",
            'Extreme Blocks (%)': f"{extreme_blocks:.2f}"
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save as CSV
    df.to_csv(os.path.join(output_dir, 'magnitude_distribution_summary.csv'), index=False)
    
    # Also create a text summary
    with open(os.path.join(output_dir, 'magnitude_distribution_summary.txt'), 'w') as f:
        f.write("Magnitude Distribution Analysis Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("Key Findings:\n")
        f.write("-" * 40 + "\n")
        
        # Find variables with highest proportion of tiny values
        tiny_percentages = [(stats['var_name'], stats['tiny_samples']/stats['total_samples']*100) 
                           for stats in all_stats]
        tiny_percentages.sort(key=lambda x: x[1], reverse=True)
        
        f.write(f"\nVariables with highest proportion of tiny values (< 1e-9):\n")
        for var, pct in tiny_percentages[:5]:
            f.write(f"  {var}: {pct:.2f}%\n")
    
    print(f"\nPlots and summary saved to {output_dir}/")

def main():
    # Data folder path
    data_folder = "/u/tawal/BSSN-Extracted-Data/tt_q01/"
    
    # Check if folder exists
    if not os.path.exists(data_folder):
        print(f"Error: Data folder not found at {data_folder}")
        return
    
    print(f"Loading data from: {data_folder}")
    
    # Load all U_ variables except U_CHI
    variable_data = load_all_variables(data_folder, exclude_vars=['U_CHI'])
    
    # Analyze magnitude distribution for each variable
    all_stats = []
    for var_name, data in variable_data.items():
        if data.size > 0:
            print(f"\nAnalyzing {var_name}...")
            stats = analyze_magnitude_distribution(data, var_name)
            all_stats.append(stats)
    
    # Also analyze all variables combined
    print("\nAnalyzing all variables combined...")
    all_data_combined = np.concatenate([data.flatten() for data in variable_data.values() if data.size > 0])
    combined_stats = analyze_magnitude_distribution(all_data_combined, "All Variables Combined")
    all_stats.append(combined_stats)
    
    # Plot distributions
    plot_magnitude_distributions(all_stats)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()