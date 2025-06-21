import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_bssn_data(data_dir):
    """
    Load data from all HDF5 files in the given directory.
    
    Args:
        data_dir: Path to directory containing HDF5 files
        
    Returns:
        all_data: Array of shape (n_samples, n_vars, H, W, D)
        var_names: List of variable names
    """
    print(f"Loading data from {data_dir}")
    all_data = []
    var_names = None
    
    # Get list of HDF5 files
    hdf5_files = [f for f in os.listdir(data_dir) if f.endswith('.hdf5')]
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    for file_idx, file_name in enumerate(sorted(hdf5_files)):
        file_path = os.path.join(data_dir, file_name)
        print(f"Processing file {file_idx+1}/{len(hdf5_files)}: {file_name}")
        
        with h5py.File(file_path, 'r') as f:
            # Get variable data: shape (n_vars, n_blocks, H, W, D)
            var_data = f['var_data'][:]
            
            # Store variable names if not already stored
            if var_names is None:
                var_names = [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in f['vars'][:]]
            
            # Reshape to (n_blocks, n_vars, H, W, D) for each file
            n_vars, n_blocks, H, W, D = var_data.shape
            reshaped_data = np.transpose(var_data, (1, 0, 2, 3, 4))
            
            # Add to all_data
            all_data.append(reshaped_data)
    
    # Concatenate all data
    if all_data:
        all_data = np.vstack(all_data)
        print(f"Total data shape: {all_data.shape}")
    else:
        print("No data loaded!")
        return None, None
    
    return all_data, var_names

def plot_detailed_spatial_slices(datasets, all_variable_names, num_samples_per_set=5, slice_indices_to_plot=None, save_dir='visualizations/detailed_spatial_slices'):
    """
    Plot detailed spatial slices for each variable.
    Shows num_samples_per_set random samples from train, val, test sets.
    For each sample, plots multiple slices at specified slice_indices_to_plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if slice_indices_to_plot is None:
        slice_indices_to_plot = [0, 2, 4] # Assuming 5x5x5, take start, middle, end slice along one axis
    num_slices_per_sample = len(slice_indices_to_plot)

    set_names = ['train', 'val', 'test']
    num_sets = len(set_names)

    for var_idx, var_name in enumerate(all_variable_names):
        print(f"Plotting detailed spatial slices for variable: {var_name} (index {var_idx})")
        
        fig, axes = plt.subplots(num_sets * num_samples_per_set, num_slices_per_sample, 
                                 figsize=(num_slices_per_sample * 3, num_sets * num_samples_per_set * 1.2), 
                                 squeeze=False)
        fig.suptitle(f'Detailed Spatial Slices for {var_name}', fontsize=16, y=1.02)

        for set_row_idx, set_name in enumerate(set_names):
            dataset_for_set = datasets[set_name] # Shape: (n_samples_in_set, n_vars, H, W, D)
            if len(dataset_for_set) == 0:
                print(f"Skipping {set_name} for {var_name} as it's empty.")
                continue
            
            # Ensure num_samples_per_set is not greater than available samples
            actual_num_samples = min(num_samples_per_set, len(dataset_for_set))
            if actual_num_samples < num_samples_per_set:
                print(f"Warning: Requested {num_samples_per_set} samples for {set_name} set of {var_name}, but only {actual_num_samples} are available.")

            if actual_num_samples == 0:
                 for s_idx in range(num_samples_per_set):
                    for slice_k_idx in range(num_slices_per_sample):
                        ax = axes[set_row_idx * num_samples_per_set + s_idx, slice_k_idx]
                        ax.text(0.5, 0.5, 'No Samples', ha='center', va='center')
                        ax.axis('off')
                 continue

            rand_sample_indices = np.random.choice(len(dataset_for_set), actual_num_samples, replace=False)

            for s_idx in range(num_samples_per_set): # Loop up to requested samples
                if s_idx < actual_num_samples:
                    sample_data_idx = rand_sample_indices[s_idx]
                    # data_3d_block shape: (H, W, D) e.g. (5,5,5) for the specific variable
                    data_3d_block = dataset_for_set[sample_data_idx, var_idx, :, :, :]
                    min_val, max_val = np.min(data_3d_block), np.max(data_3d_block)

                    for slice_k_idx, slice_axis_idx in enumerate(slice_indices_to_plot):
                        ax = axes[set_row_idx * num_samples_per_set + s_idx, slice_k_idx]
                        # Assuming data_3d_block is (D, H, W) and we slice along D (axis 0)
                        # If it's (H,W,D), then slice_img = data_3d_block[:, :, slice_axis_idx]
                        # Based on previous code: dataset[rand_idx, var_idx, :, :, 2] implies last axis is depth
                        if slice_axis_idx >= data_3d_block.shape[2]: # Check bounds for the slicing axis
                            ax.text(0.5, 0.5, f'Slice {slice_axis_idx}\nOut of Bounds', ha='center', va='center')
                            ax.axis('off')
                            continue
                        slice_img = data_3d_block[:, :, slice_axis_idx] 
                        
                        im = ax.imshow(slice_img, cmap='viridis', vmin=min_val, vmax=max_val)
                        ax.axis('off')
                        title = f'{set_name.capitalize()} S{s_idx+1} Sl{slice_axis_idx}'
                        if s_idx == 0 and slice_k_idx == num_slices_per_sample // 2 : # Main title for the sample row
                           title = f'{set_name.capitalize()} - Sample {s_idx+1}\nSlice {slice_axis_idx}'
                        ax.set_title(title, fontsize=8)
                        # Add colorbar to the last slice of each sample
                        if slice_k_idx == num_slices_per_sample - 1:
                           fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else: # If fewer samples available than requested, blank out remaining subplots
                    for slice_k_idx in range(num_slices_per_sample):
                        ax = axes[set_row_idx * num_samples_per_set + s_idx, slice_k_idx]
                        ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                        ax.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
        plt.savefig(f'{save_dir}/{var_name}_detailed_spatial.png', dpi=100, bbox_inches='tight')
        plt.close(fig)

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Directory containing HDF5 files
    data_dir = '/u/tawal/NN-based-Compression-for-Scientific-Data/BSSN Extracted Data/tt_q01'
    
    # Load data
    all_data, var_names = load_bssn_data(data_dir)
    
    if all_data is None:
        print("Failed to load data.")
        return
    
    # Shuffle data
    print("Shuffling data...")
    indices = np.arange(all_data.shape[0])
    np.random.shuffle(indices)
    all_data = all_data[indices]
    
    # Split into train/val/test sets (70/15/15 split)
    print("Splitting data into train/val/test sets...")
    train_data, temp_data = train_test_split(all_data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"Train set shape: {train_data.shape}")
    print(f"Validation set shape: {val_data.shape}")
    print(f"Test set shape: {test_data.shape}")
    
    # Prepare datasets dictionary
    datasets = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    # Adjust slice indices based on the actual data shape
    _, _, _, _, D = all_data.shape
    slice_indices = [0, D//2, D-1] if D > 2 else [0]
    
    # Plot detailed spatial slices
    print("Generating visualizations...")
    plot_detailed_spatial_slices(
        datasets=datasets,
        all_variable_names=var_names,
        num_samples_per_set=3,
        slice_indices_to_plot=slice_indices,
        save_dir='visualizations/detailed_spatial_slices'
    )
    
    print("Done!")

if __name__ == "__main__":
    main()
