import os
import numpy as np
import h5py

def print_full_sample():
    """Print full sample data in both shapes (30, 7, 7, 7) and (7, 7, 7, 30)"""
    # Path to one of the HDF5 files
    file_path = '/u/tawal/NN-based-Compression-for-Scientific-Data/BSSN Extracted Data/tt_q01/bssn_gr_0_extracted.hdf5'
    
    print(f"Loading data from {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # Get variable data: shape (n_vars, n_blocks, H, W, D)
        var_data = f['var_data'][:]
        var_names = [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in f['vars'][:]]
        
        # Get a single block (sample)
        block_idx = 0  # First block
        sample_data = var_data[:, block_idx, :, :, :]  # Shape: (30, 7, 7, 7)
        
        # Print full sample in shape (30, 7, 7, 7)
        print("\n============ FULL SAMPLE IN SHAPE (30, 7, 7, 7) ============")
        print(f"Shape: {sample_data.shape}")
        
        # Print each variable separately
        for var_idx in range(5):  # Print first 5 variables to keep output manageable
            print(f"\nVariable {var_idx}: {var_names[var_idx]}")
            for i in range(7):
                for j in range(7):
                    print(f"Layer {i}, Row {j}: {sample_data[var_idx, i, j, :]}")
        
        # Reshape to (7, 7, 7, 30)
        reshaped_sample = np.transpose(sample_data, (1, 2, 3, 0))
        
        # Print full sample in shape (7, 7, 7, 30)
        print("\n============ FULL SAMPLE IN SHAPE (7, 7, 7, 30) ============")
        print(f"Shape: {reshaped_sample.shape}")
        
        # Print a few specific positions
        positions = [(0,0,0), (3,3,3), (6,6,6)]
        for pos in positions:
            i, j, k = pos
            print(f"\nPosition ({i},{j},{k}) - All 30 variables:")
            for var_idx in range(30):
                print(f"{var_names[var_idx]}: {reshaped_sample[i, j, k, var_idx]}")

if __name__ == "__main__":
    print_full_sample()
