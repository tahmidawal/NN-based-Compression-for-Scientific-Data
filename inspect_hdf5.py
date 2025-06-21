import h5py
import os
import sys

def inspect_hdf5(file_path):
    """Inspect the structure of an HDF5 file"""
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"File: {file_path}")
            print("Keys:", list(f.keys()))
            
            for key in f.keys():
                print(f"{key} shape:", f[key].shape)
                print(f"{key} dtype:", f[key].dtype)
                
                # If the dataset is small enough, print a sample
                if len(f[key].shape) > 0 and f[key].shape[0] < 10:
                    try:
                        print(f"{key} sample:", f[key][:2])
                    except:
                        print(f"{key} sample: Too large to display")
    except Exception as e:
        print(f"Error inspecting {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        inspect_hdf5(file_path)
    else:
        print("Please provide an HDF5 file path as argument")
