#!/usr/bin/env python3
"""
SWAE 3D Training Script for U_ Variables Dataset (5x5x5 Version)
Trains on 5x5x5 blocks extracted from GR simulation data for all U_ variables (excluding C_ variables)
Based on "Exploring Autoencoder-based Error-bounded Compression for Scientific Data"
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

# Try to import tensorboard, make it optional
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: tensorboard not available, logging will be disabled")
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.swae_pure_3d_5x5x5_opt import create_swae_3d_5x5x5_model
from datasets.all_variables_dataset_5x5x5_opt import create_all_variables_5x5x5_datasets


def calculate_psnr(original, reconstructed):
    """
    Calculate PSNR as defined in the paper:
    PSNR = 20 * log10(vrange(D)) - 10 * log10(mse(D, D'))
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0 or mse < 1e-12:
        return float('inf')
    
    value_range = np.max(original) - np.min(original)
    if value_range == 0 or value_range < 1e-12:
        return float('inf')
    
    # Add small epsilon to prevent log(0)
    psnr = 20 * np.log10(value_range) - 10 * np.log10(mse + 1e-12)
    return psnr


def calculate_compression_ratio():
    """
    Calculate theoretical compression ratio
    Original: 5x5x5 = 125 values
    Compressed: 16 latent dimensions
    Compression ratio: 125/16 ≈ 7.8:1
    """
    original_size = 5 * 5 * 5  # 125
    compressed_size = 16       # latent dimension
    return original_size / compressed_size


def train_epoch(model, train_loader, optimizer, device, epoch, writer=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_sw_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, metadata) in enumerate(train_loader):
        data = data.to(device)  # Shape: (batch_size, 1, 5, 5, 5)
        
        optimizer.zero_grad()
        
        # Forward pass
        x_recon, z = model(data)
        
        # Calculate loss
        loss_dict = model.loss_function(data, x_recon, z)
        loss = loss_dict['loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability (increased for new data distribution)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Reduced from 2.0 for better stability with small values
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += loss_dict['recon_loss'].item()
        total_sw_loss += loss_dict['sw_distance'].item()
        num_batches += 1
        
        # Log every 100 batches
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f} '
                  f'(Recon: {loss_dict["recon_loss"].item():.6f}, '
                  f'SW: {loss_dict["sw_distance"].item():.6f})')
            
            if writer:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                writer.add_scalar('Train/BatchReconLoss', loss_dict['recon_loss'].item(), global_step)
                writer.add_scalar('Train/BatchSWLoss', loss_dict['sw_distance'].item(), global_step)
    
    # Return average losses
    return {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'sw_loss': total_sw_loss / num_batches
    }


def validate_epoch(model, val_loader, device, epoch, writer=None):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_sw_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data, metadata in val_loader:
            data = data.to(device)
            
            # Forward pass
            x_recon, z = model(data)
            
            # Calculate loss
            loss_dict = model.loss_function(data, x_recon, z)
            
            # Accumulate losses
            total_loss += loss_dict['loss'].item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_sw_loss += loss_dict['sw_distance'].item()
            num_batches += 1
    
    # Return average losses
    avg_losses = {
        'loss': total_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'sw_loss': total_sw_loss / num_batches
    }
    
    print(f'Validation Epoch: {epoch}\t'
          f'Loss: {avg_losses["loss"]:.6f} '
          f'(Recon: {avg_losses["recon_loss"]:.6f}, '
          f'SW: {avg_losses["sw_loss"]:.6f})')
    
    if writer:
        writer.add_scalar('Val/Loss', avg_losses['loss'], epoch)
        writer.add_scalar('Val/ReconLoss', avg_losses['recon_loss'], epoch)
        writer.add_scalar('Val/SWLoss', avg_losses['sw_loss'], epoch)
    
    return avg_losses


def evaluate_reconstruction_quality(model, val_dataset, device, num_samples=10):
    """Evaluate reconstruction quality on validation samples"""
    model.eval()
    
    psnr_values = []
    mse_values = []
    correlation_values = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(val_dataset))):
            # Get sample
            sample, metadata = val_dataset[i]
            sample = sample.unsqueeze(0).to(device)  # Add batch dimension
            
            # Reconstruct
            x_recon, z = model(sample)
            
            # Convert to numpy
            original = sample.cpu().numpy().squeeze()
            reconstructed = x_recon.cpu().numpy().squeeze()
            
            # Denormalize if needed (FIXED: use correct sample index)
            if hasattr(val_dataset, 'denormalize'):
                original = val_dataset.denormalize(original, sample_idx=i)
                reconstructed = val_dataset.denormalize(reconstructed, sample_idx=i)
            
            # Calculate metrics
            mse = np.mean((original - reconstructed) ** 2)
            psnr = calculate_psnr(original, reconstructed)
            
            # Calculate correlation
            flat_orig = original.flatten()
            flat_recon = reconstructed.flatten()
            correlation = np.corrcoef(flat_orig, flat_recon)[0, 1]
            
            psnr_values.append(psnr)
            mse_values.append(mse)
            correlation_values.append(correlation)
            
            # Debug first sample
            if i == 0:
                print(f"  Debug Sample 0:")
                print(f"    Original range: [{original.min():.6f}, {original.max():.6f}]")
                print(f"    Reconstructed range: [{reconstructed.min():.6f}, {reconstructed.max():.6f}]")
                print(f"    Error range: [{(original-reconstructed).min():.6f}, {(original-reconstructed).max():.6f}]")
                print(f"    Latent range: [{z.min():.6f}, {z.max():.6f}]")
    
    avg_psnr = np.mean(psnr_values)
    avg_mse = np.mean(mse_values)
    avg_correlation = np.mean(correlation_values)
    
    print(f"\nReconstruction Quality Evaluation ({num_samples} samples):")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average Correlation: {avg_correlation:.6f}")
    print(f"Compression Ratio: {calculate_compression_ratio():.1f}:1")
    
    return avg_psnr, avg_mse


def main():
    parser = argparse.ArgumentParser(description='Train SWAE 3D for All Variables Dataset (5x5x5 Version)')
    
    # Data parameters
    parser.add_argument('--data-folder', type=str, 
                        default='/u/tawal/BSSN Extracted Data/tt_q01/',
                        help='Path to folder containing HDF5 files')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize the data')
    parser.add_argument('--normalize-method', type=str, default='pos_log',
                        choices=['minmax', 'zscore', 'pos_log', 'none'],
                        help='Normalization method')
    parser.add_argument('--exclude-vars', type=str, nargs='*', default=[],
                        help='List of variables to exclude from training')
    parser.add_argument('--scaleitup', action='store_true',
                        help='Scale up samples with tiny magnitudes before poslog')
    parser.add_argument('--scale-target', type=float, default=2.5e-3,
                        help='Target magnitude for scaling (default: 2.5e-3 for poslog ~-6)')
    
    # Model parameters (following Table VI for 3D data)
    parser.add_argument('--latent-dim', type=int, default=16,
                        help='Latent dimension (16 as per Table VI)')
    parser.add_argument('--lambda-reg', type=float, default=0.9,
                        help='Regularization weight for SW distance')
    parser.add_argument('--encoder-channels', type=str, default='32,64,128,256',
                        help='Encoder channel configuration (comma-separated, e.g., "32,64,128,256")')
    parser.add_argument('--arch', type=str, default='conv',
                        choices=['conv', 'mlp', 'gmlp'],
                        help='Network backbone: conv (default), mlp, or gmlp')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Training data ratio (val=0.15, test=0.05 - FIXED 5%% test set)')
    parser.add_argument('--early-stopping-patience', type=int, default=40,
                        help='Number of epochs to wait for improvement before stopping')
    parser.add_argument('--early-stopping-min-delta', type=float, default=1e-4,
                        help='Minimum change in validation loss to qualify as an improvement')
    
    # System parameters
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--save-dir', type=str, default='./save/swae_all_vars_5x5x5',
                        help='Directory to save models and logs')
    
    # Evaluation parameters
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='Evaluate reconstruction quality every N epochs')
    parser.add_argument('--save-interval', type=int, default=20,
                        help='Save model every N epochs')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Data folder: {args.data_folder}")
    print(f"Theoretical compression ratio: {calculate_compression_ratio():.1f}:1")
    
    # Mixed precision & compile optimizations
    torch.backends.cudnn.benchmark = True
    
    # TF32 optimization (PyTorch 1.7+)
    if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = True
        print("✓ TF32 optimization enabled")
    else:
        print(f"⚠️  TF32 not available in PyTorch {torch.__version__}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup logging
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))
    else:
        writer = None
    
    # Create datasets with proper train/val/test split
    print("Creating All Variables 5x5x5 datasets...")
    train_dataset, val_dataset, test_dataset = create_all_variables_5x5x5_datasets(
        data_folder=args.data_folder,
        train_ratio=args.train_split,
        val_ratio=0.15,  # 15% for validation during training
        normalize=args.normalize,
        normalize_method=args.normalize_method,
        exclude_vars=args.exclude_vars,
        scaleitup=args.scaleitup,
        scale_target=args.scale_target
    )
    
    print(f"🔄 Data splits: Train={len(train_dataset)} (80%), Val={len(val_dataset)} (15%), Test={len(test_dataset)} (5%)")
    print(f"⚠️  CRITICAL: Test set (5%) is held out for final unbiased evaluation - NEVER seen during training")
    print(f"🔒 Using deterministic split (seed=42) to ensure consistent test set")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("Creating SWAE 3D 5x5x5 model...")
    # Parse encoder channels from string to list of integers
    encoder_channels = [int(c) for c in args.encoder_channels.split(',')]
    print(f"Using encoder channels: {encoder_channels}")
    
    model = create_swae_3d_5x5x5_model(
        latent_dim=args.latent_dim,
        lambda_reg=args.lambda_reg,
        encoder_channels=encoder_channels,
        arch=args.arch
    ).to(device)
    
    # Apply compile optimization for faster training (PyTorch 2.0+)
    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("✓ Model compilation enabled (PyTorch 2.0+)")
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")
    else:
        if not hasattr(torch, 'compile'):
            print(f"⚠️  torch.compile() not available in PyTorch {torch.__version__} (requires 2.0+)")
        else:
            print("⚠️  CUDA not available, skipping model compilation")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model settings: latent_dim={args.latent_dim}, lambda_reg={args.lambda_reg}, channels={encoder_channels}, arch={args.arch}")
    print(f"Training settings: lr={args.lr}, batch_size={args.batch_size}")
    if args.early_stopping_patience:
        print(f"Early stopping: enabled (patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta})")
    else:
        print(f"Early stopping: disabled")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_state = None
    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, device, epoch, writer)
        
        # Validate
        val_losses = validate_epoch(model, val_loader, device, epoch, writer)
        
        # Update learning rate
        scheduler.step()
        
        # Early stopping check
        current_val_loss = val_losses["loss"]
        if current_val_loss < best_val_loss - args.early_stopping_min_delta:
            best_val_loss = current_val_loss
            early_stopping_counter = 0
            # Save best model state
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.early_stopping_patience:
                print(f'\nEarly stopping triggered after {epoch} epochs')
                # Restore best model
                model.load_state_dict(best_model_state)
                break
        
        # Log epoch results
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch} completed in {epoch_time:.1f}s')
        print(f'Train Loss: {train_losses["loss"]:.6f}, Val Loss: {val_losses["loss"]:.6f}')
        print(f'Early stopping counter: {early_stopping_counter}/{args.early_stopping_patience}')
        
        if writer:
            writer.add_scalar('Train/EpochLoss', train_losses['loss'], epoch)
            writer.add_scalar('Train/EpochReconLoss', train_losses['recon_loss'], epoch)
            writer.add_scalar('Train/EpochSWLoss', train_losses['sw_loss'], epoch)
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        # Save best model
        if val_losses['loss'] < best_val_loss:
            best_val_loss = val_losses['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_losses['loss'],
                'val_loss': val_losses['loss'],
                'args': args
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'New best model saved (val_loss: {best_val_loss:.6f})')
        
        # Evaluate reconstruction quality
        if epoch % args.eval_interval == 0:
            avg_psnr, avg_mse = evaluate_reconstruction_quality(model, val_dataset, device)
            if writer:
                writer.add_scalar('Eval/PSNR', avg_psnr, epoch)
                writer.add_scalar('Eval/MSE', avg_mse, epoch)
        
        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_losses['loss'],
                'val_loss': val_losses['loss'],
                'args': args
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
            print(f'Checkpoint saved at epoch {epoch}')
    
    # Final evaluation
    print("\nFinal evaluation...")
    avg_psnr, avg_mse = evaluate_reconstruction_quality(model, val_dataset, device, num_samples=50)
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_train_loss': train_losses['loss'],
        'final_val_loss': val_losses['loss'],
        'final_psnr': avg_psnr,
        'final_mse': avg_mse,
        'args': args
    }, os.path.join(args.save_dir, 'final_model.pth'))
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final PSNR: {avg_psnr:.2f} dB")
    print(f"Final MSE: {avg_mse:.6f}")
    print(f"Models saved in: {args.save_dir}")
    
    # Optional: Save INT8 quantized model for inference
    if args.arch in ['mlp', 'gmlp']:
        try:
            # Try different import paths for different PyTorch versions
            try:
                from torch.ao.quantization import quantize_dynamic
            except ImportError:
                from torch.quantization import quantize_dynamic
            
            print("\nCreating INT8 quantized model for inference...")
            int8_model = quantize_dynamic(
                model.cpu(),                          # must be on CPU
                {nn.Linear},                          # layers to quantise
                dtype=torch.qint8
            )
            int8_path = os.path.join(args.save_dir, 'int8_model.pth')
            torch.save({
                'model_state_dict': int8_model.state_dict(),
                'args': args
            }, int8_path)
            print(f"✓ INT8 quantized model saved to: {int8_path}")
        except ImportError as e:
            print(f"⚠️  INT8 quantization not available in PyTorch {torch.__version__}: {e}")
        except Exception as e:
            print(f"⚠️  Could not create INT8 model: {e}")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main() 