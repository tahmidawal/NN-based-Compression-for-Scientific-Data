#!/usr/bin/env python3
"""
All Variables Dataset for SWAE 3D Training (5x5x5 Version)
Loads all variables from GR simulation HDF5 files and extracts 5x5x5 center crop
Data shape: (num_samples, num_vars, 5, 5, 5) extracted from original (num_samples, num_vars, 7, 7, 7)
"""

import os
import glob
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class AllVariablesDataset5x5x5(Dataset):
    """
    Dataset for loading all variables from GR simulation HDF5 files
    Extracts 5x5x5 center crop from original 7x7x7 data
    
    Args:
        data_folder: Path to folder containing HDF5 files
        split: 'train', 'val', or 'test'
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation (rest goes to test)
        normalize: Whether to normalize the data
        normalize_method: 'minmax', 'zscore', 'pos_log', or 'none'
        exclude_vars: List of variables to exclude (default: empty list to include all)
    """
    
    def __init__(self, data_folder, split='train', train_ratio=0.8, val_ratio=0.15,
                 normalize=True, normalize_method='minmax', shuffle=True, seed=42, exclude_vars=None,
                 scaleitup=False, scale_target=2.5e-3):
        self.data_folder = data_folder
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.shuffle = shuffle
        self.seed = seed
        self.exclude_vars = exclude_vars if exclude_vars is not None else []
        self.scaleitup = scaleitup
        self.scale_target = scale_target
        
        # Validate ratios
        if self.test_ratio < 0:
            raise ValueError(f"Invalid ratios: train_ratio({train_ratio}) + val_ratio({val_ratio}) > 1.0")
        
        # Load all variables data from HDF5 files
        self.data, self.var_names, self.sample_var_mapping = self._load_all_variables_data()
        
        # CRITICAL: Fixed deterministic shuffle to ensure test set is always the same
        # This guarantees the test set is never contaminated across different runs
        if self.shuffle:
            # Use FIXED seed to ensure identical splits across all runs
            np.random.seed(self.seed)
            perm = np.random.permutation(self.data.shape[0])
            self.data = self.data[perm]
            # Also shuffle the sample_var_mapping to keep it aligned
            if hasattr(self, 'sample_var_mapping'):
                self.sample_var_mapping = [self.sample_var_mapping[i] for i in perm]
            print(f"🔒 Applied deterministic shuffle with seed={self.seed} for reproducible splits")
        
        # Split data into train/val/test
        self._split_data()
        
        # Apply normalization if requested
        if self.normalize:
            self._normalize_data()
            
        print(f"All Variables Dataset (5x5x5) initialized:")
        print(f"  Split: {split}")
        print(f"  Data shape: {self.data.shape}")
        print(f"  Variables included: {self.var_names}")
        print(f"  Number of variables: {len(self.var_names)}")
        print(f"  Split ratios: train={train_ratio:.1%}, val={val_ratio:.1%}, test={self.test_ratio:.1%}")
        print(f"  Data range: [{self.data.min():.6f}, {self.data.max():.6f}]")
        print(f"  Data mean: {self.data.mean():.6f}")
        print(f"  Data std: {self.data.std():.6f}")
    
    def _load_all_variables_data(self):
        """Load all variables data from HDF5 files and extract 5x5x5 center crop"""
        # Find all HDF5 files
        hdf5_files = glob.glob(os.path.join(self.data_folder, "*.hdf5"))
        hdf5_files.sort()  # Sort for consistent ordering
        
        print(f"Found {len(hdf5_files)} HDF5 files in {self.data_folder}")
        
        # Load all variables data from all files
        all_variables_data = []
        var_names = None
        
        for file_path in hdf5_files:
            print(f"Loading {os.path.basename(file_path)}...")
            
            with h5py.File(file_path, 'r') as f:
                # Check available keys
                if len(all_variables_data) == 0:  # Only print for first file
                    print(f"Available keys: {list(f.keys())}")
                
                # Get variable names
                file_var_names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in f['vars'][:]]
                
                # Initialize var_names on first file
                if var_names is None:
                    # Filter to only include U_ variables and exclude C_ variables
                    # Also respect the explicit exclude_vars list
                    var_names = [v for v in file_var_names 
                                if v.startswith('U_') and not v.startswith('C_') 
                                and v not in self.exclude_vars]
                    print(f"Available variables: {file_var_names}")
                    print(f"Variables to load (U_ prefix only): {var_names}")
                    print(f"Excluded variables: {[v for v in file_var_names if v.startswith('C_') or v in self.exclude_vars]}")
                
                # Extract all variables data
                var_data = f['var_data'][:]  # Shape: (variables, num_samples, 7, 7, 7)
                
                # Get indices of variables to include
                var_indices = [file_var_names.index(v) for v in var_names if v in file_var_names]
                
                # Extract data for selected variables and reshape
                # We want to treat each variable's samples as separate samples
                file_samples = []
                for var_idx in var_indices:
                    var_samples = var_data[var_idx]  # Shape: (num_samples, 7, 7, 7)
                    file_samples.append(var_samples)
                
                # Stack all variables' samples together
                file_samples = np.concatenate(file_samples, axis=0)  # Shape: (num_vars * num_samples, 7, 7, 7)
                
                print(f"  Loaded {len(var_indices)} variables, total samples in file: {file_samples.shape[0]}")
                all_variables_data.append(file_samples)
        
        # Concatenate all data
        full_data = np.concatenate(all_variables_data, axis=0)  # (total_samples_all_vars, 7, 7, 7)
        
        # Extract 5x5x5 center crop from 7x7x7 data
        print(f"Extracting 5x5x5 center crop from 7x7x7 data...")
        print(f"Original data shape: {full_data.shape}")
        
        # Center crop indices: start at (7-5)//2 = 1, end at 1+5 = 6
        crop_start = (7 - 5) // 2  # 1
        crop_end = crop_start + 5   # 6
        
        # Crop spatial dimensions
        cropped_data = full_data[:, crop_start:crop_end, crop_start:crop_end, crop_start:crop_end]
        print(f"Cropped data shape: {cropped_data.shape}")
        print(f"Crop indices: [:, {crop_start}:{crop_end}, {crop_start}:{crop_end}, {crop_start}:{crop_end}]")
        
        # Add channel dimension to make it (total_samples, 1, 5, 5, 5)
        cropped_data = cropped_data[:, np.newaxis, :, :, :]
        
        # Create sample_var_mapping to track which variable each sample came from
        sample_var_mapping = []
        for file_data in all_variables_data:
            # Each file contains samples from all variables concatenated
            # We need to know how many samples per variable per file
            samples_per_file = len(file_data)
            samples_per_var = samples_per_file // len(var_names)
            
            # Create mapping for this file
            for var_idx, var_name in enumerate(var_names):
                sample_var_mapping.extend([var_name] * samples_per_var)
        
        print(f"Final data shape: {cropped_data.shape}")
        print(f"Created sample_var_mapping with {len(sample_var_mapping)} entries")
        print(f"Sample variable distribution: {dict(zip(*np.unique(sample_var_mapping, return_counts=True)))}")
        
        return cropped_data, var_names, sample_var_mapping
        
    def _split_data(self):
        """Split data into train, validation, and test sets"""
        total_samples = self.data.shape[0]
        n_train = int(total_samples * self.train_ratio)
        n_val = int(total_samples * self.val_ratio)
        n_test = total_samples - n_train - n_val
        
        print(f"🔒 FIXED Data Split: {total_samples} total → {n_train} train ({self.train_ratio:.1%}), {n_val} val ({self.val_ratio:.1%}), {n_test} test ({self.test_ratio:.1%})")
        print(f"🎯 TEST SET: {n_test} samples (5%) - NEVER seen during training/validation")
        
        if self.split == 'train':
            self.data = self.data[:n_train]
            if hasattr(self, 'sample_var_mapping'):
                self.sample_var_mapping = self.sample_var_mapping[:n_train]
            print(f"Using training data: {self.data.shape}")
        elif self.split == 'val':
            self.data = self.data[n_train:n_train+n_val]
            if hasattr(self, 'sample_var_mapping'):
                self.sample_var_mapping = self.sample_var_mapping[n_train:n_train+n_val]
            print(f"Using validation data: {self.data.shape}")
        elif self.split == 'test':
            self.data = self.data[n_train+n_val:]
            if hasattr(self, 'sample_var_mapping'):
                self.sample_var_mapping = self.sample_var_mapping[n_train+n_val:]
            print(f"Using test data: {self.data.shape}")
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")
    
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
            
        elif self.normalize_method == 'pos_log':
            # FIXED: Positive shift then log transformation (PER-SAMPLE, not global)
            # This ensures we get the expected statistical properties (mean~4.5, std~1.1)
            self.epsilon = np.float64(1e-6)  # Increased from 1e-8 for better numerical stability
            
            # Store transformation parameters for each sample
            self.sample_data_mins = []
            self.sample_scale_factors = []
            normalized_data = []
            
            print(f"Applying per-sample positive-shift log normalization...")
            if self.scaleitup:
                print(f"  With scaleitup preprocessing (target: {self.scale_target})")
            
            for i in range(self.data.shape[0]):
                sample = self.data[i].copy()  # Shape: (1, 5, 5, 5) - make a copy to avoid modifying original
                
                # Apply scaleitup if enabled
                scale_factor = 1.0
                if self.scaleitup:
                    # Calculate mean magnitude
                    sample_abs = np.abs(sample)
                    mean_magnitude = np.mean(sample_abs)
                    
                    # Scale up if needed (skip if sample is all zeros)
                    if mean_magnitude > 0 and mean_magnitude < self.scale_target:
                        scale_factor = self.scale_target / mean_magnitude
                        # Round to nearest power of 10
                        scale_factor_log = np.log10(scale_factor)
                        # Clamp to reasonable range to avoid overflow
                        scale_factor_log = np.clip(scale_factor_log, -20, 20)
                        scale_factor = 10 ** round(scale_factor_log)
                        sample = np.asarray(sample * scale_factor, dtype=np.float64)
                
                # Apply per-sample transformation
                # Ensure sample is a numpy array with proper shape
                if not isinstance(sample, np.ndarray):
                    sample = np.asarray(sample, dtype=np.float64)
                
                # Ensure sample has the right shape (1, 5, 5, 5)
                if sample.ndim != 4:
                    print(f"Warning: sample {i} has unexpected shape {sample.shape}")
                    
                sample_min = float(sample.min())  # Convert to Python float
                
                # Shift data - ensure result is numpy array
                data_shifted = np.asarray(sample - sample_min, dtype=np.float64)
                
                # Add epsilon separately to ensure numpy array result
                log_input = np.asarray(data_shifted + self.epsilon, dtype=np.float64)
                
                # Final safety check and log
                if not isinstance(log_input, np.ndarray):
                    log_input = np.asarray(log_input, dtype=np.float64)
                
                log_sample = np.log(log_input)
                
                # Store parameters for denormalization
                self.sample_data_mins.append(sample_min)
                self.sample_scale_factors.append(scale_factor)
                normalized_data.append(log_sample)
            
            # Replace data with normalized version
            self.data = np.array(normalized_data)
            
            print(f"Applied per-sample positive-shift log normalization:")
            print(f"  epsilon={self.epsilon}")
            print(f"  Per-sample data_min range: [{min(self.sample_data_mins):.6f}, {max(self.sample_data_mins):.6f}]")
            if self.scaleitup:
                unique_scales = np.unique(self.sample_scale_factors)
                scale_counts = {scale: np.sum(np.array(self.sample_scale_factors) == scale) 
                               for scale in unique_scales}
                print(f"  Scale factors applied: {scale_counts}")
            print(f"  Result should have mean~4.5, std~1.1 properties")
            
        elif self.normalize_method == 'none':
            print("No normalization applied")
            
        else:
            raise ValueError(f"Invalid normalize_method: {self.normalize_method}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the data sample
        sample = self.data[idx]  # Shape: (1, 5, 5, 5)
        
        # Convert to tensor
        sample = torch.from_numpy(sample).float()
        
        # Get which variable this sample came from
        variable_name = self.sample_var_mapping[idx] if hasattr(self, 'sample_var_mapping') and idx < len(self.sample_var_mapping) else 'unknown'
        
        # Return sample and metadata (updated for 5x5x5)
        metadata = {
            'index': idx,
            'original_shape': (5, 5, 5),  # Updated for 5x5x5
            'variable_name': variable_name,
            'all_variables': self.var_names if hasattr(self, 'var_names') else [],
            'crop_info': {
                'original_size': (7, 7, 7),
                'crop_start': 1,
                'crop_end': 6,
                'crop_size': (5, 5, 5)
            }
        }
        
        return sample, metadata
    
    def denormalize(self, data, sample_idx=None):
        """Denormalize data back to original range"""
        if not self.normalize:
            return data
            
        if self.normalize_method == 'minmax':
            return data * (self.data_max - self.data_min) + self.data_min
        elif self.normalize_method == 'zscore':
            return data * self.data_std + self.data_mean
        elif self.normalize_method == 'pos_log':
            # FIXED: Per-sample denormalization with correct sample index
            if hasattr(self, 'sample_data_mins') and len(self.sample_data_mins) > 0:
                if sample_idx is not None and 0 <= sample_idx < len(self.sample_data_mins):
                    # Use the correct sample's parameters
                    sample_min = self.sample_data_mins[sample_idx]
                    result = np.exp(data) - self.epsilon + sample_min
                    
                    # Apply inverse scaling if scaleitup was used
                    if hasattr(self, 'sample_scale_factors') and len(self.sample_scale_factors) > sample_idx:
                        scale_factor = self.sample_scale_factors[sample_idx]
                        if scale_factor > 1.0:
                            result = result / scale_factor
                    
                    return result
                else:
                    # Fallback: return in log scale with warning
                    print(f"Warning: No valid sample_idx provided for denormalization. Returning log-scale data.")
                    return data
            else:
                # Fallback to old method if parameters not available
                return np.exp(data) - self.epsilon + getattr(self, 'data_min', 0)
        else:
            return data
    
    def denormalize_batch(self, data_batch, start_idx=0):
        """Denormalize a batch of data with correct per-sample parameters"""
        if not self.normalize or self.normalize_method != 'pos_log':
            return data_batch
            
        if not hasattr(self, 'sample_data_mins'):
            print("Warning: No per-sample parameters available for batch denormalization")
            return data_batch
        
        denormalized_batch = []
        for i, data in enumerate(data_batch):
            sample_idx = start_idx + i
            if sample_idx < len(self.sample_data_mins):
                sample_min = self.sample_data_mins[sample_idx]
                denormalized = np.exp(data) - self.epsilon + sample_min
                
                # Apply inverse scaling if scaleitup was used
                if hasattr(self, 'sample_scale_factors') and sample_idx < len(self.sample_scale_factors):
                    scale_factor = self.sample_scale_factors[sample_idx]
                    if scale_factor > 1.0:
                        denormalized = denormalized / scale_factor
                
                denormalized_batch.append(denormalized)
            else:
                print(f"Warning: Sample index {sample_idx} out of range, returning log-scale data")
                denormalized_batch.append(data)
        
        return np.array(denormalized_batch)


def create_all_variables_5x5x5_datasets(data_folder, train_ratio=0.8, val_ratio=0.15, normalize=True, normalize_method='minmax', shuffle=True, seed=42, exclude_vars=None, scaleitup=False, scale_target=2.5e-3):
    """
    Create train, validation, and test datasets for U_CHI 5x5x5 data
    
    CRITICAL: Uses FIXED seed to ensure test set is always the same 5% of data
    across all training runs and inference. Test set is NEVER seen during training.
    
    Args:
        data_folder: Path to folder containing HDF5 files
        train_ratio: Ratio of data to use for training (default: 0.8 = 80%)
        val_ratio: Ratio of data to use for validation (default: 0.15 = 15%)
        normalize: Whether to normalize the data
        normalize_method: Normalization method ('minmax', 'zscore', 'pos_log', 'none')
        seed: FIXED seed (42) for reproducible splits - DO NOT CHANGE!
    
    Returns:
        train_dataset, val_dataset, test_dataset
        
    Data Split:
        - Training: 80% (used for model training)
        - Validation: 15% (used for hyperparameter tuning and early stopping)  
        - Test: 5% (held out for final unbiased evaluation - NEVER seen during training)
    
    Data Processing:
        - Loads original 7x7x7 U_CHI data
        - Extracts 5x5x5 center crop from each sample
        - Applies normalization to 5x5x5 data
    """
    
    # Calculate test ratio
    test_ratio = 1.0 - train_ratio - val_ratio
    print(f"🔒 Creating 5x5x5 datasets with FIXED splits: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")
    print(f"🎯 Using FIXED seed={seed} to ensure test set is always the same samples")
    
    if test_ratio < 0:
        raise ValueError(f"Invalid ratios: train_ratio({train_ratio}) + val_ratio({val_ratio}) > 1.0")
    
    if test_ratio != 0.05:
        print(f"⚠️  Warning: Test ratio is {test_ratio:.1%}, not the intended 5%")
    
    train_dataset = AllVariablesDataset5x5x5(
        data_folder=data_folder,
        split='train', 
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize=normalize,
        normalize_method=normalize_method,
        shuffle=shuffle,
        seed=seed,
        exclude_vars=exclude_vars,
        scaleitup=scaleitup,
        scale_target=scale_target
    )
    
    val_dataset = AllVariablesDataset5x5x5(
        data_folder=data_folder,
        split='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize=normalize,
        normalize_method=normalize_method,
        shuffle=shuffle,
        seed=seed,
        exclude_vars=exclude_vars,
        scaleitup=scaleitup,
        scale_target=scale_target
    )
    
    test_dataset = AllVariablesDataset5x5x5(
        data_folder=data_folder,
        split='test',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        normalize=normalize,
        normalize_method=normalize_method,
        shuffle=shuffle,
        seed=seed,
        exclude_vars=exclude_vars,
        scaleitup=scaleitup,
        scale_target=scale_target
    )
    
    return train_dataset, val_dataset, test_dataset


# Backward compatibility - alias to the new function
create_u_chi_datasets = create_all_variables_5x5x5_datasets


if __name__ == "__main__":
    # Test the dataset
    data_folder = "/u/tawal/BSSN-Extracted-Data/tt_q01/"
    
    print("Testing All Variables 5x5x5 dataset...")
    train_dataset, val_dataset, test_dataset = create_all_variables_5x5x5_datasets(
        data_folder=data_folder,
        train_ratio=0.8,  # 80% train
        val_ratio=0.15,   # 15% val
        normalize=True,   # 5% test
        normalize_method='pos_log',
        exclude_vars=[]  # Include all variables
    )
    
    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test data loading
    sample, metadata = train_dataset[0]
    print(f"\nSample shape: {sample.shape}")
    print(f"Sample dtype: {sample.dtype}")
    print(f"Sample range: [{sample.min():.6f}, {sample.max():.6f}]")
    print(f"Metadata: {metadata}")
    
    # Show some samples from different variables
    print("\nChecking samples from different variables:")
    for i in range(0, min(100, len(train_dataset)), 10):
        _, meta = train_dataset[i]
        print(f"  Sample {i}: from variable '{meta['variable_name']}'")
    
    print("\nAll Variables 5x5x5 dataset test completed successfully!") 