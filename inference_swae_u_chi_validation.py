#!/usr/bin/env python3
"""
SWAE U_CHI Validation Inference Script
Evaluates trained SWAE model on validation set of U_CHI data
"""

import os
import sys
import argparse
import numpy as np
import torch
import h5py
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# VTK imports (optional)
try:
    import vtk
    from vtk.util import numpy_support
    VTK_AVAILABLE = True
except ImportError:
    print("Warning: VTK not available, VTI files will not be saved")
    VTK_AVAILABLE = False

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.swae_pure_3d_7x7x7 import create_swae_3d_7x7x7_model
from datasets.u_chi_dataset import create_u_chi_datasets


def calculate_metrics(original, reconstructed):
    """Calculate reconstruction quality metrics"""
    # MSE
    mse = np.mean((original - reconstructed) ** 2)
    
    # PSNR
    value_range = np.max(original) - np.min(original)
    if value_range == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(value_range) - 10 * np.log10(mse + 1e-8)
    
    # MAE
    mae = np.mean(np.abs(original - reconstructed))
    
    # Correlation coefficient
    flat_orig = original.flatten()
    flat_recon = reconstructed.flatten()
    correlation = np.corrcoef(flat_orig, flat_recon)[0, 1]
    
    return {
        'mse': mse,
        'psnr': psnr,
        'mae': mae,
        'correlation': correlation,
        'min_error': np.min(original - reconstructed),
        'max_error': np.max(original - reconstructed),
        'mean_error': np.mean(original - reconstructed),
        'std_error': np.std(original - reconstructed)
    }


def save_vti_file(data, filename, spacing=(1.0, 1.0, 1.0)):
    """Save 3D numpy array as VTI file"""
    if not VTK_AVAILABLE:
        print(f"VTK not available, skipping VTI file: {filename}")
        return
    
    # Create VTK image data
    image_data = vtk.vtkImageData()
    dims = data.shape
    image_data.SetDimensions(dims[0], dims[1], dims[2])
    image_data.SetSpacing(spacing)
    image_data.SetOrigin(0.0, 0.0, 0.0)
    
    # Convert numpy array to VTK array
    vtk_array = numpy_support.numpy_to_vtk(data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)
    vtk_array.SetName("U_CHI")
    
    # Add array to image data
    image_data.GetPointData().SetScalars(vtk_array)
    
    # Write VTI file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image_data)
    writer.Write()
    print(f"Saved VTI file: {filename}")


def plot_comparison_slices(original, reconstructed, error, output_dir, sample_idx):
    """Plot comparison slices from different axes at 1/3 and 1/2 positions"""
    # Create figure with subplots
    fig, axes = plt.subplots(6, 3, figsize=(15, 24))
    fig.suptitle(f'Sample {sample_idx:03d} - Reconstruction Comparison', fontsize=16)
    
    # Get dimensions
    depth, height, width = original.shape
    
    # Define slice positions (1/3 and 1/2)
    slice_positions = {
        'z': [depth // 3, depth // 2],      # Z-axis (depth)
        'y': [height // 3, height // 2],    # Y-axis (height)  
        'x': [width // 3, width // 2]       # X-axis (width)
    }
    
    plot_row = 0
    
    for axis_name, positions in slice_positions.items():
        for pos in positions:
            # Extract slices based on axis
            if axis_name == 'z':
                orig_slice = original[pos, :, :]
                recon_slice = reconstructed[pos, :, :]
                error_slice = error[pos, :, :]
                title_suffix = f"Z={pos}/{depth}"
            elif axis_name == 'y':
                orig_slice = original[:, pos, :]
                recon_slice = reconstructed[:, pos, :]
                error_slice = error[:, pos, :]
                title_suffix = f"Y={pos}/{height}"
            else:  # x
                orig_slice = original[:, :, pos]
                recon_slice = reconstructed[:, :, pos]
                error_slice = error[:, :, pos]
                title_suffix = f"X={pos}/{width}"
            
            # Plot original
            im1 = axes[plot_row, 0].imshow(orig_slice, cmap='viridis', aspect='equal')
            axes[plot_row, 0].set_title(f'Original - {title_suffix}')
            axes[plot_row, 0].axis('off')
            plt.colorbar(im1, ax=axes[plot_row, 0], shrink=0.6)
            
            # Plot reconstructed
            im2 = axes[plot_row, 1].imshow(recon_slice, cmap='viridis', aspect='equal')
            axes[plot_row, 1].set_title(f'Reconstructed - {title_suffix}')
            axes[plot_row, 1].axis('off')
            plt.colorbar(im2, ax=axes[plot_row, 1], shrink=0.6)
            
            # Plot error
            im3 = axes[plot_row, 2].imshow(error_slice, cmap='RdBu_r', aspect='equal')
            axes[plot_row, 2].set_title(f'Error - {title_suffix}')
            axes[plot_row, 2].axis('off')
            plt.colorbar(im3, ax=axes[plot_row, 2], shrink=0.6)
            
            plot_row += 1
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = os.path.join(output_dir, f'sample_{sample_idx:03d}_comparison_slices.png')
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {plot_filename}")


def save_validation_results(original, reconstructed, error, metrics, output_dir, sample_idx, save_vti=False):
    """Save validation results to HDF5 file and optionally VTI files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save HDF5
    output_file = os.path.join(output_dir, f'sample_{sample_idx:03d}_validation.hdf5')
    
    with h5py.File(output_file, 'w') as f:
        # Save data
        f.create_dataset('original', data=original)
        f.create_dataset('reconstructed', data=reconstructed)
        f.create_dataset('error', data=error)
        
        # Save metrics
        metrics_group = f.create_group('metrics')
        for key, value in metrics.items():
            metrics_group.attrs[key] = value
        
        # Save metadata
        f.attrs['shape'] = original.shape
        f.attrs['sample_idx'] = sample_idx
    
    # Save VTI files if requested
    if save_vti:
        vti_dir = os.path.join(output_dir, 'vti_files')
        os.makedirs(vti_dir, exist_ok=True)
        
        save_vti_file(original, os.path.join(vti_dir, f'sample_{sample_idx:03d}_original.vti'))
        save_vti_file(reconstructed, os.path.join(vti_dir, f'sample_{sample_idx:03d}_reconstructed.vti'))
        save_vti_file(error, os.path.join(vti_dir, f'sample_{sample_idx:03d}_error.vti'))
        
        # Plot comparison slices
        plot_comparison_slices(original, reconstructed, error, output_dir, sample_idx)


def main():
    parser = argparse.ArgumentParser(description='SWAE U_CHI Validation Inference')
    
    # Data parameters
    parser.add_argument('--data-folder', type=str, 
                        default='/u/tawal/0620-NN-based-compression-thera/tt_q01/',
                        help='Path to folder containing HDF5 files')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output-dir', type=str, default='validation_u_chi_results',
                        help='Directory to save validation results')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of validation samples to process')
    parser.add_argument('--num-vti-samples', type=int, default=5,
                        help='Number of random samples to save as VTI files')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create datasets
    print("Loading U_CHI datasets...")
    _, val_dataset = create_u_chi_datasets(
        data_folder=args.data_folder,
        train_ratio=0.8,
        normalize=True,
        normalize_method='minmax'
    )
    
    # Create data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )
    
    # Load model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model = create_swae_3d_7x7x7_model(
        latent_dim=checkpoint['args'].latent_dim,
        lambda_reg=checkpoint['args'].lambda_reg
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully (epoch {checkpoint['epoch']})")
    print(f"Model parameters: latent_dim={checkpoint['args'].latent_dim}, lambda_reg={checkpoint['args'].lambda_reg}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select random samples for VTI output
    vti_sample_indices = random.sample(range(min(args.num_samples, len(val_dataset))), 
                                       min(args.num_vti_samples, args.num_samples, len(val_dataset)))
    print(f"Selected samples for VTI output: {vti_sample_indices}")
    
    # Run validation
    print("\nRunning validation inference...")
    all_metrics = []
    
    with torch.no_grad():
        for sample_idx in range(min(args.num_samples, len(val_dataset))):
            # Get sample
            sample, metadata = val_dataset[sample_idx]
            sample = sample.unsqueeze(0).to(device)  # Add batch dimension
            
            # Reconstruct
            x_recon, z = model(sample)
            
            # Convert to numpy and denormalize
            original = val_dataset.denormalize(sample.cpu().numpy().squeeze())
            reconstructed = val_dataset.denormalize(x_recon.cpu().numpy().squeeze())
            error = original - reconstructed
            
            # Calculate metrics
            metrics = calculate_metrics(original, reconstructed)
            all_metrics.append(metrics)
            
            # Determine if this sample should get VTI output
            save_vti = sample_idx in vti_sample_indices
            
            # Save results
            save_validation_results(
                original=original,
                reconstructed=reconstructed,
                error=error,
                metrics=metrics,
                output_dir=args.output_dir,
                sample_idx=sample_idx,
                save_vti=save_vti
            )
            
            # Print progress
            print(f"Sample {sample_idx + 1}/{args.num_samples}:")
            print(f"  Original range: [{original.min():.6f}, {original.max():.6f}]")
            print(f"  Reconstructed range: [{reconstructed.min():.6f}, {reconstructed.max():.6f}]")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  PSNR: {metrics['psnr']:.2f} dB")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  Correlation: {metrics['correlation']:.6f}")
            print(f"  Error range: [{metrics['min_error']:.6f}, {metrics['max_error']:.6f}]")
            if save_vti:
                print(f"  ✓ Saved VTI files and plots")
    
    # Calculate and save average metrics
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    print("\n" + "="*50)
    print("FINAL VALIDATION METRICS:")
    print("="*50)
    print(f"Average MSE: {avg_metrics['mse']:.6f}")
    print(f"Average PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"Average MAE: {avg_metrics['mae']:.6f}")
    print(f"Average Correlation: {avg_metrics['correlation']:.6f}")
    print(f"Average Error Range: [{avg_metrics['min_error']:.6f}, {avg_metrics['max_error']:.6f}]")
    
    # Calculate compression ratio
    original_size = 7 * 7 * 7  # 343
    compressed_size = checkpoint['args'].latent_dim  # 16
    compression_ratio = original_size / compressed_size
    print(f"Compression Ratio: {compression_ratio:.1f}:1")
    
    # Save average metrics
    metrics_file = os.path.join(args.output_dir, 'average_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Average Validation Metrics:\n")
        f.write("="*30 + "\n")
        for key, value in avg_metrics.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nCompression Ratio: {compression_ratio:.1f}:1\n")
        f.write(f"Model Parameters: latent_dim={checkpoint['args'].latent_dim}, lambda_reg={checkpoint['args'].lambda_reg}\n")
    
    print(f"\nValidation results saved in: {args.output_dir}")
    print(f"VTI files and plots saved for {len(vti_sample_indices)} random samples")


if __name__ == "__main__":
    main() 