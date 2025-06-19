import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import os

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_spatial_distribution(centers, levels, save_dir):
    # Plot centers distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (ax, dim, label) in enumerate(zip(axes, range(3), ['X', 'Y', 'Z'])):
        sns.histplot(data=centers[:, i], bins=50, ax=ax)
        ax.set_title(f'{label} coordinates distribution')
        ax.set_xlabel(f'{label} coordinate')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spatial_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3D scatter plot from different angles
    angles = [(45, 45), (0, 0), (90, 0), (0, 90)]
    for i, (elev, azim) in enumerate(angles):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                           c=levels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Level')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'3D Distribution (elevation={elev}°, azimuth={azim}°)')
        plt.savefig(os.path.join(save_dir, f'3d_distribution_angle_{i}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_variable_statistics(var_data, vars_names, save_dir):
    # Overall statistics
    var_stats = pd.DataFrame({
        'mean': np.mean(var_data.reshape(30, -1), axis=1),
        'std': np.std(var_data.reshape(30, -1), axis=1),
        'min': np.min(var_data.reshape(30, -1), axis=1),
        'max': np.max(var_data.reshape(30, -1), axis=1),
        'median': np.median(var_data.reshape(30, -1), axis=1)
    }, index=vars_names)
    
    # Plot distributions
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    var_stats[['mean', 'median']].plot(kind='bar', ax=axes[0])
    axes[0].set_title('Mean and Median Values by Variable')
    axes[0].tick_params(axis='x', rotation=90)
    
    var_stats[['min', 'max']].plot(kind='bar', ax=axes[1])
    axes[1].set_title('Min and Max Values by Variable')
    axes[1].tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'variable_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Correlation matrix
    corr_matrix = np.corrcoef(var_data.reshape(30, -1))
    plt.figure(figsize=(15, 15))
    sns.heatmap(corr_matrix, xticklabels=vars_names, yticklabels=vars_names,
                cmap='RdBu_r', center=0, annot=True, fmt='.2f', 
                square=True, cbar_kws={'label': 'Correlation'})
    plt.title('Variable Correlations')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_level_analysis(levels, centers, save_dir):
    # Level distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=levels, bins=np.arange(levels.min(), levels.max()+2)-0.5,
                discrete=True)
    plt.title('Distribution of Refinement Levels')
    plt.xlabel('Level')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, 'level_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Level vs. radial distance
    radial_dist = np.sqrt(np.sum(centers**2, axis=1))
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=radial_dist, y=levels, alpha=0.5)
    plt.title('Refinement Level vs. Radial Distance')
    plt.xlabel('Radial Distance')
    plt.ylabel('Level')
    plt.savefig(os.path.join(save_dir, 'level_vs_distance.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_variable_slices(var_data, vars_names, save_dir):
    # Create a subdirectory for variable slices
    slice_dir = os.path.join(save_dir, 'variable_slices')
    ensure_dir(slice_dir)
    
    # Plot middle slices for each variable
    mid_point = var_data.shape[1] // 2
    for var_idx, var_name in enumerate(vars_names):
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle(f'Variable: {var_name} - Middle Slices')
        
        # XY plane
        im0 = axes[0,0].imshow(var_data[var_idx, mid_point, :, :, 3], 
                              cmap='viridis')
        axes[0,0].set_title('XY Plane (Middle Z)')
        plt.colorbar(im0, ax=axes[0,0])
        
        # XZ plane
        im1 = axes[0,1].imshow(var_data[var_idx, mid_point, :, 3, :], 
                              cmap='viridis')
        axes[0,1].set_title('XZ Plane (Middle Y)')
        plt.colorbar(im1, ax=axes[0,1])
        
        # YZ plane
        im2 = axes[1,0].imshow(var_data[var_idx, mid_point, 3, :, :], 
                              cmap='viridis')
        axes[1,0].set_title('YZ Plane (Middle X)')
        plt.colorbar(im2, ax=axes[1,0])
        
        # Distribution
        sns.histplot(var_data[var_idx].flatten(), bins=50, ax=axes[1,1])
        axes[1,1].set_title('Value Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(slice_dir, f'variable_{var_name}_slices.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

def load_and_explore_data(file_path, save_dir='visualizations'):
    ensure_dir(save_dir)
    
    with h5py.File(file_path, 'r') as f:
        # Print the structure of the HDF5 file
        print("HDF5 file structure:")
        def print_structure(name, obj):
            print(f"{'  ' * name.count('/')}{name}: {type(obj)}")
        f.visititems(print_structure)
        
        # Load data
        centers = f['centers'][:]
        levels = f['levels'][:]
        var_data = f['var_data'][:]
        vars_names = [name.decode('utf-8') for name in f['vars'][:]]
        
        # Generate all plots
        print("\nGenerating spatial distribution plots...")
        plot_spatial_distribution(centers, levels, save_dir)
        
        print("Generating variable statistics plots...")
        plot_variable_statistics(var_data, vars_names, save_dir)
        
        print("Generating level analysis plots...")
        plot_level_analysis(levels, centers, save_dir)
        
        print("Generating variable slice plots...")
        plot_variable_slices(var_data, vars_names, save_dir)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Number of points: {len(centers)}")
        print(f"Number of variables: {len(vars_names)}")
        print(f"Variable names: {vars_names}")
        print(f"\nLevel statistics:")
        print(f"Min level: {levels.min()}")
        print(f"Max level: {levels.max()}")
        print(f"Mean level: {levels.mean():.2f}")
        
        print(f"\nAll visualizations have been saved to the '{save_dir}' directory.")

if __name__ == "__main__":
    file_path = "bssn_gr_0_extracted.hdf5"
    load_and_explore_data(file_path) 