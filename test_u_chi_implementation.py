#!/usr/bin/env python3
"""
Test script for U_CHI SWAE implementation
Verifies that all components work correctly before training
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.swae_pure_3d_7x7x7 import create_swae_3d_7x7x7_model
from datasets.u_chi_dataset import create_u_chi_datasets


def test_u_chi_dataset():
    """Test the U_CHI dataset loading"""
    print("=== Testing U_CHI Dataset ===")
    
    data_folder = "/u/tawal/0620-NN-based-compression-thera/tt_q01/"
    
    try:
        # Create datasets
        train_dataset, val_dataset = create_u_chi_datasets(
            data_folder=data_folder,
            train_ratio=0.8,
            normalize=True,
            normalize_method='minmax'
        )
        
        print(f"✓ Dataset creation successful")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        
        # Test data loading
        sample, metadata = train_dataset[0]
        print(f"✓ Data loading successful")
        print(f"  Sample shape: {sample.shape}")
        print(f"  Sample dtype: {sample.dtype}")
        print(f"  Sample range: [{sample.min():.6f}, {sample.max():.6f}]")
        print(f"  Metadata keys: {list(metadata.keys())}")
        
        # Test batch loading
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        batch, batch_metadata = next(iter(train_loader))
        print(f"✓ Batch loading successful")
        print(f"  Batch shape: {batch.shape}")
        print(f"  Expected: (4, 1, 7, 7, 7)")
        
        assert batch.shape == (4, 1, 7, 7, 7), f"Expected (4, 1, 7, 7, 7), got {batch.shape}"
        print("✓ All dataset tests passed!")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        raise


def test_swae_7x7x7_model():
    """Test the SWAE 7x7x7 model architecture"""
    print("\n=== Testing SWAE 7x7x7 Model ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Create model
        model = create_swae_3d_7x7x7_model(
            latent_dim=16,
            lambda_reg=10.0
        ).to(device)
        
        print(f"✓ Model creation successful")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test with dummy data
        batch_size = 4
        test_input = torch.randn(batch_size, 1, 7, 7, 7).to(device)
        
        print(f"✓ Test input created: {test_input.shape}")
        
        # Forward pass
        with torch.no_grad():
            x_recon, z = model(test_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Latent shape: {z.shape}")
        print(f"  Reconstruction shape: {x_recon.shape}")
        
        # Check shapes
        assert z.shape == (batch_size, 16), f"Expected latent shape (4, 16), got {z.shape}"
        assert x_recon.shape == test_input.shape, f"Expected reconstruction shape {test_input.shape}, got {x_recon.shape}"
        
        # Test loss computation
        loss_dict = model.loss_function(test_input, x_recon, z)
        
        print(f"✓ Loss computation successful")
        print(f"  Total loss: {loss_dict['loss']:.6f}")
        print(f"  Reconstruction loss: {loss_dict['recon_loss']:.6f}")
        print(f"  SW distance: {loss_dict['sw_distance']:.6f}")
        
        # Check loss values are reasonable
        assert loss_dict['loss'] > 0, "Total loss should be positive"
        assert loss_dict['recon_loss'] > 0, "Reconstruction loss should be positive"
        assert loss_dict['sw_distance'] >= 0, "SW distance should be non-negative"
        
        print("✓ All model tests passed!")
        
        return model
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        raise


def test_training_step():
    """Test a single training step with real data"""
    print("\n=== Testing Training Step ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Create model and dataset
        model = create_swae_3d_7x7x7_model().to(device)
        
        data_folder = "/u/tawal/0620-NN-based-compression-thera/tt_q01/"
        train_dataset, _ = create_u_chi_datasets(
            data_folder=data_folder,
            train_ratio=0.8,
            normalize=True,
            normalize_method='minmax'
        )
        
        # Create data loader with small batch for testing
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Get a batch
        data, metadata = next(iter(train_loader))
        data = data.to(device)
        
        print(f"✓ Real data batch loaded: {data.shape}")
        print(f"  Data range: [{data.min():.6f}, {data.max():.6f}]")
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        x_recon, z = model(data)
        
        # Compute loss
        loss_dict = model.loss_function(data, x_recon, z)
        loss = loss_dict['loss']
        
        print(f"✓ Forward pass successful")
        print(f"  Loss before backward: {loss.item():.6f}")
        print(f"  Reconstruction error range: [{(data - x_recon).min():.6f}, {(data - x_recon).max():.6f}]")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        print(f"✓ Backward pass successful")
        print(f"  Total gradient norm: {total_grad_norm:.6f}")
        assert total_grad_norm > 0, "Gradients should be non-zero"
        
        # Optimizer step
        optimizer.step()
        
        print("✓ Training step test passed!")
        
    except Exception as e:
        print(f"✗ Training step test failed: {e}")
        raise


def main():
    """Run all tests"""
    print("Testing U_CHI SWAE Implementation")
    print("=" * 50)
    
    try:
        # Test 1: Dataset
        train_dataset, val_dataset = test_u_chi_dataset()
        
        # Test 2: Model
        model = test_swae_7x7x7_model()
        
        # Test 3: Training step
        test_training_step()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The U_CHI SWAE implementation is ready for training!")
        
        # Print compression info
        original_size = 7 * 7 * 7  # 343
        compressed_size = 16       # latent dimension
        compression_ratio = original_size / compressed_size
        
        print(f"\nCompression Details:")
        print(f"  Original size: {original_size} values per block")
        print(f"  Compressed size: {compressed_size} latent dimensions")
        print(f"  Compression ratio: {compression_ratio:.1f}:1")
        print(f"  Total training samples: {len(train_dataset)}")
        print(f"  Total validation samples: {len(val_dataset)}")
        
    except Exception as e:
        print(f"\n❌ TESTS FAILED: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 