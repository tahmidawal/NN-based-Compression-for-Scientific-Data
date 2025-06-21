import os
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt

# --- Configuration ---
TARGET_VARIABLE_NAME = 'U_CHI'
DATA_DIR = 'BSSN Extracted Data/tt_q01/'
HDF5_PATTERN = os.path.join(DATA_DIR, '*.hdf5')
OUTPUT_DIR = 'visualizations/'
OUTPUT_FILENAME = 'uchi_magnitude_histogram.png'

# --- 1. Load and Aggregate Data for U_CHI ---
def load_uchi_data(pattern, target_variable_name):
    """Loads and aggregates U_CHI data from multiple HDF5 files."""
    print(f"--- Starting to load data for variable: {target_variable_name} ---")
    hdf5_files = sorted(glob.glob(pattern))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found matching pattern: {pattern}")
    print(f"Found {len(hdf5_files)} HDF5 files to process.")

    all_uchi_data = []
    var_names_from_file = None

    for i, file_path in enumerate(hdf5_files):
        print(f"Processing file {i+1}/{len(hdf5_files)}: {file_path}")
        try:
            with h5py.File(file_path, 'r') as hf:
                if var_names_from_file is None:
                    var_names_from_file = [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in hf['vars'][:]]
                
                try:
                    target_idx = var_names_from_file.index(target_variable_name)
                    # Shape of var_data: (n_vars, n_blocks, D, H, W)
                    # We want to extract only the U_CHI data: (n_blocks, D, H, W)
                    uchi_data_for_file = hf['var_data'][target_idx, ...]
                    all_uchi_data.append(uchi_data_for_file)
                    print(f"  Extracted '{target_variable_name}' data, shape for this file: {uchi_data_for_file.shape}")
                except ValueError:
                    print(f"ERROR - Target variable '{target_variable_name}' not found in {file_path}. Available: {var_names_from_file}")
                    # Optionally, skip this file or raise an error
                    continue 
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            # Optionally, re-raise or handle
            continue
    
    if not all_uchi_data:
        raise ValueError(f"No data loaded for '{target_variable_name}'. Check logs for errors.")

    # Concatenate data from all files. Each element in all_uchi_data is (n_blocks, D, H, W)
    # Resulting shape: (total_n_blocks, D, H, W)
    aggregated_uchi_data = np.concatenate(all_uchi_data, axis=0)
    print(f"Aggregated '{target_variable_name}' data shape: {aggregated_uchi_data.shape}")
    print(f"--- Finished loading data for {target_variable_name} ---")
    return aggregated_uchi_data

# --- 2. Plot Histogram of Magnitudes ---
def plot_magnitude_histogram(data, variable_name, output_dir, filename, bins=100):
    """Calculates magnitudes and plots a histogram."""
    print(f"--- Plotting magnitude histogram for {variable_name} ---")
    os.makedirs(output_dir, exist_ok=True)

    magnitudes = np.abs(data.flatten())
    
    plt.figure(figsize=(12, 7))
    plt.hist(magnitudes, bins=bins, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of Magnitudes for {variable_name}', fontsize=16)
    plt.xlabel('Magnitude', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.yscale('log') # Use log scale for y-axis due to potentially large range in frequencies
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Histogram saved to {output_path}")
    print(f"--- Finished plotting histogram for {variable_name} ---")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting U_CHI magnitude histogram generation...")
    # 1. Load U_CHI data
    uchi_data_aggregated = load_uchi_data(HDF5_PATTERN, TARGET_VARIABLE_NAME)
    
    # 2. Plot histogram
    plot_magnitude_histogram(uchi_data_aggregated, TARGET_VARIABLE_NAME, OUTPUT_DIR, OUTPUT_FILENAME)
    
    print("Script finished.")
