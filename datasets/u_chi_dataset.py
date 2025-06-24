#!/usr/bin/env python3
"""
U_CHI Dataset for SWAE 3D Training
Loads U_CHI variable from GR simulation HDF5 files
Data shape: (num_samples, 1, 7, 7, 7)
"""

import os
import glob
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class UCHIDataset(Dataset):
    """
    Dataset for loading U_CHI variable from GR simulation HDF5 files
    
    Args:
        data_folder: Path to folder containing HDF5 files
        split: 'train' or 'val'
        train_ratio: Ratio of data to use for training (rest for validation)
        normalize: Whether to normalize the data
        normalize_method: 'minmax', 'zscore', or 'none'
    """
    
    def __init__(self, data_folder, split='train', train_ratio=0.8, 
                 normalize=True, normalize_method='minmax'):
        self.data_folder = data_folder
        self.split = split
        self.train_ratio = train_ratio
        self.normalize = normalize
        self.normalize_method = normalize_method
        
        # Load U_CHI data from all HDF5 files
        self.data = self._load_u_chi_data()
        
        # Split data
        self._split_data()
        
        # Apply normalization if requested
        if self.normalize:
            self._normalize_data()
            
        print(f"U_CHI Dataset initialized:")
        print(f"  Split: {split}")
        print(f"  Data shape: {self.data.shape}")
        print(f"  Data range: [{self.data.min():.6f}, {self.data.max():.6f}]")
        print(f"  Data mean: {self.data.mean():.6f}")
        print(f"  Data std: {self.data.std():.6f}")
    
    def _load_u_chi_data(self):
        """Load U_CHI data from all HDF5 files"""
        # Find all HDF5 files
        hdf5_files = glob.glob(os.path.join(self.data_folder, "*.hdf5"))
        hdf5_files.sort()  # Sort for consistent ordering
        
        print(f"Found {len(hdf5_files)} HDF5 files in {self.data_folder}")
        
        # Load U_CHI data from all files
        all_u_chi_data = []
        
        for file_path in hdf5_files:
            print(f"Loading {os.path.basename(file_path)}...")
            
            with h5py.File(file_path, 'r') as f:
                # Check available keys
                if len(all_u_chi_data) == 0:  # Only print for first file
                    print(f"Available keys: {list(f.keys())}")
                
                # Get variable names
                var_names = [name.decode('utf-8') for name in f['vars'][:]]
                if len(all_u_chi_data) == 0:  # Only print for first file
                    print(f"Available variables: {var_names}")
                
                # Find U_CHI index
                try:
                    u_chi_idx = var_names.index('U_CHI')
                    print(f"Found U_CHI at index {u_chi_idx}")
                except ValueError:
                    raise ValueError(f"U_CHI not found in {file_path}. Available: {var_names}")
                
                # Extract U_CHI data
                var_data = f['var_data'][:]  # Shape: (variables, num_samples, 7, 7, 7)
                u_chi_data = var_data[u_chi_idx]  # Shape: (num_samples, 7, 7, 7)
                
                print(f"  Loaded U_CHI data shape: {u_chi_data.shape}")
                all_u_chi_data.append(u_chi_data)
        
        # Concatenate all data
        full_data = np.concatenate(all_u_chi_data, axis=0)  # (total_samples, 7, 7, 7)
        
        # Add channel dimension to make it (total_samples, 1, 7, 7, 7)
        full_data = full_data[:, np.newaxis, :, :, :]
        
        print(f"Total U_CHI data shape: {full_data.shape}")
        
        return full_data
        
    def _split_data(self):
        """Split data into train and validation sets"""
        total_samples = self.data.shape[0]
        n_train = int(total_samples * self.train_ratio)
        
        if self.split == 'train':
            self.data = self.data[:n_train]
            print(f"Using training data: {self.data.shape}")
        elif self.split == 'val':
            self.data = self.data[n_train:]
            print(f"Using validation data: {self.data.shape}")
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'val'")
    
    def _normalize_data(self):
        """Normalize the data based on the specified method"""
        if self.normalize_method == 'minmax':
            # Min-max normalization to [0, 1]
            self.data_min = self.data.min()
            self.data_max = self.data.max()
            self.data = (self.data - self.data_min) / (self.data_max - self.data_min)
            print(f"Applied min-max normalization: [{self.data_min:.6f}, {self.data_max:.6f}] -> [0, 1]")
            
        elif self.normalize_method == 'zscore':
            # Z-score normalization
            self.data_mean = self.data.mean()
            self.data_std = self.data.std()
            self.data = (self.data - self.data_mean) / self.data_std
            print(f"Applied z-score normalization: mean={self.data_mean:.6f}, std={self.data_std:.6f}")
            
        elif self.normalize_method == 'none':
            print("No normalization applied")
            
        else:
            raise ValueError(f"Invalid normalize_method: {self.normalize_method}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the data sample
        sample = self.data[idx]  # Shape: (1, 7, 7, 7)
        
        # Convert to tensor
        sample = torch.from_numpy(sample).float()
        
        # Return sample and metadata (empty dict for compatibility)
        metadata = {
            'index': idx,
            'original_shape': (7, 7, 7)
        }
        
        return sample, metadata
    
    def denormalize(self, data):
        """Denormalize data back to original range"""
        if not self.normalize:
            return data
            
        if self.normalize_method == 'minmax':
            return data * (self.data_max - self.data_min) + self.data_min
        elif self.normalize_method == 'zscore':
            return data * self.data_std + self.data_mean
        else:
            return data


def create_u_chi_datasets(data_folder, train_ratio=0.8, normalize=True, normalize_method='minmax'):
    """
    Create train and validation datasets for U_CHI data
    
    Args:
        data_folder: Path to folder containing HDF5 files
        train_ratio: Ratio of data to use for training
        normalize: Whether to normalize the data
        normalize_method: Normalization method ('minmax', 'zscore', 'none')
    
    Returns:
        train_dataset, val_dataset
    """
    train_dataset = UCHIDataset(
        data_folder=data_folder,
        split='train',
        train_ratio=train_ratio,
        normalize=normalize,
        normalize_method=normalize_method
    )
    
    val_dataset = UCHIDataset(
        data_folder=data_folder,
        split='val',
        train_ratio=train_ratio,
        normalize=normalize,
        normalize_method=normalize_method
    )
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    # Test the dataset
    data_folder = "/u/tawal/0620-NN-based-compression-thera/tt_q01/"
    
    print("Testing U_CHI dataset...")
    train_dataset, val_dataset = create_u_chi_datasets(
        data_folder=data_folder,
        train_ratio=0.8,
        normalize=True,
        normalize_method='minmax'
    )
    
    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Test data loading
    sample, metadata = train_dataset[0]
    print(f"\nSample shape: {sample.shape}")
    print(f"Sample dtype: {sample.dtype}")
    print(f"Sample range: [{sample.min():.6f}, {sample.max():.6f}]")
    print(f"Metadata: {metadata}")
    
    print("\nU_CHI dataset test completed successfully!") 