import os
import numpy as np
import h5py

def load_and_print_sample():
    """Load a single sample from an HDF5 file and print it in different shapes"""
    # Path to one of the HDF5 files
    file_path = '/u/tawal/NN-based-Compression-for-Scientific-Data/BSSN Extracted Data/tt_q01/bssn_gr_0_extracted.hdf5'
    
    print(f"Loading data from {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # Get variable data: shape (n_vars, n_blocks, H, W, D)
        var_data = f['var_data'][:]
        var_names = [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in f['vars'][:]]
        
        print(f"Original var_data shape: {var_data.shape}")
        
        # Get a single block (sample)
        block_idx = 0  # First block
        sample_data = var_data[:, block_idx, :, :, :]  # Shape: (30, 7, 7, 7)
        
        print("\n=== Sample in shape (30, 7, 7, 7) ===")
        print(f"Shape: {sample_data.shape}")
        print("First few values of the first variable:")
        print(sample_data[0, 0:3, 0:3, 0:3])  # Print a small subset for readability
        
        # Reshape to (7, 7, 7, 30)
        reshaped_sample = np.transpose(sample_data, (1, 2, 3, 0))
        
        print("\n=== Same sample in shape (7, 7, 7, 30) ===")
        print(f"Shape: {reshaped_sample.shape}")
        print("First few values at position [0,0,0] for all 30 variables:")
        print(reshaped_sample[0, 0, 0, :])  # Print values for all variables at position [0,0,0]
        
        # Print variable names for reference
        print("\n=== Variable names ===")
        for i, name in enumerate(var_names):
            print(f"{i}: {name}")

if __name__ == "__main__":
    load_and_print_sample()
