#!/usr/bin/env python3
"""
Measure inference speed for SWAE models
"""
import torch
import numpy as np
import time
import os
import sys
from pathlib import Path

# Add the current directory to path to import our modules
sys.path.append('.')

try:
    from models.swae_pure_3d_7x7x7 import SWAE3D
    from utils.data_loader import create_data_loader
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def measure_inference_speed(model_path, latent_dim, data_folder, num_samples=50, batch_size=32, device='cuda'):
    """
    Measure inference speed for a given model
    """
    try:
        # Load model
        if not os.path.exists(model_path):
            return None, f"Model file not found: {model_path}"
        
        # Create model instance
        model = SWAE3D(latent_dim=latent_dim)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        # Create data loader
        data_loader = create_data_loader(
            data_folder, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            normalize_method='pos_log'
        )
        
        # Collect samples for speed testing
        samples = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if len(samples) >= num_samples:
                    break
                    
                batch = batch.to(device)
                for i in range(batch.size(0)):
                    if len(samples) < num_samples:
                        samples.append(batch[i:i+1])
                    else:
                        break
        
        if len(samples) == 0:
            return None, "No data samples found"
        
        print(f"Collected {len(samples)} samples for speed testing")
        
        # Warm up GPU
        with torch.no_grad():
            for _ in range(5):
                sample = samples[0]
                _ = model.encode(sample)
                _ = model.decode(model.encode(sample))
        
        # Measure compression speed
        compression_times = []
        decompression_times = []
        
        with torch.no_grad():
            for sample in samples:
                # Measure compression (encoding)
                torch.cuda.synchronize()
                start_time = time.time()
                encoded = model.encode(sample)
                torch.cuda.synchronize()
                compression_time = time.time() - start_time
                compression_times.append(compression_time)
                
                # Measure decompression (decoding)
                torch.cuda.synchronize()
                start_time = time.time()
                decoded = model.decode(encoded)
                torch.cuda.synchronize()
                decompression_time = time.time() - start_time
                decompression_times.append(decompression_time)
        
        # Calculate statistics
        avg_compression_time = np.mean(compression_times)
        avg_decompression_time = np.mean(decompression_times)
        
        # Calculate data size (7x7x7 = 343 float32 values)
        data_size_bytes = 343 * 4  # 4 bytes per float32
        data_size_mb = data_size_bytes / (1024 * 1024)
        
        # Calculate speeds in MBps
        compression_speed_mbps = data_size_mb / avg_compression_time
        decompression_speed_mbps = data_size_mb / avg_decompression_time
        
        # Calculate speeds in GBps
        compression_speed_gbps = compression_speed_mbps / 1000
        decompression_speed_gbps = decompression_speed_mbps / 1000
        
        # Calculate throughput (samples per second)
        compression_throughput = 1.0 / avg_compression_time
        decompression_throughput = 1.0 / avg_decompression_time
        
        results = {
            'compression_speed_mbps': compression_speed_mbps,
            'decompression_speed_mbps': decompression_speed_mbps,
            'compression_speed_gbps': compression_speed_gbps,
            'decompression_speed_gbps': decompression_speed_gbps,
            'compression_throughput': compression_throughput,
            'decompression_throughput': decompression_throughput,
            'avg_compression_time_ms': avg_compression_time * 1000,
            'avg_decompression_time_ms': avg_decompression_time * 1000,
            'num_samples': len(samples)
        }
        
        return results, "SUCCESS"
        
    except Exception as e:
        return None, f"Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python measure_inference_speed.py <model_path> <latent_dim> <data_folder>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    latent_dim = int(sys.argv[2])
    data_folder = sys.argv[3]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results, status = measure_inference_speed(model_path, latent_dim, data_folder, device=device)
    
    if results:
        print(f"SUCCESS,{results['compression_speed_mbps']:.2f},{results['decompression_speed_mbps']:.2f},{results['compression_speed_gbps']:.4f},{results['decompression_speed_gbps']:.4f},{results['compression_throughput']:.2f},{results['decompression_throughput']:.2f},{results['avg_compression_time_ms']:.2f},{results['avg_decompression_time_ms']:.2f},{results['num_samples']}")
    else:
        print(f"FAILED,0,0,0,0,0,0,0,0,0")
