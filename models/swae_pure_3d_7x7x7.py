#!/usr/bin/env python3
"""
SWAE 3D Model for 7x7x7 Blocks (U_CHI Dataset)
Adapted from the original 8x8x8 SWAE implementation
Based on "Exploring Autoencoder-based Error-bounded Compression for Scientific Data"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GDN(nn.Module):
    """Generalized Divisive Normalization layer as used in the paper"""
    def __init__(self, channels, inverse=False, beta_min=1e-6, gamma_init=0.1):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.channels = channels
        
        # Learnable parameters
        self.beta = nn.Parameter(torch.ones(channels))
        self.gamma = nn.Parameter(gamma_init * torch.eye(channels))
        
    def forward(self, x):
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        
        # Reshape for matrix operations
        x_flat = x.view(B, C, -1)  # (B, C, D*H*W)
        
        # Compute normalization with numerical stability
        beta = torch.clamp(self.beta, min=self.beta_min)
        
        if self.inverse:
            # iGDN: x * sqrt(beta + gamma @ x^2)
            x_sq = torch.clamp(x_flat ** 2, min=1e-12)  # Prevent zeros
            norm = torch.sqrt(beta.view(1, -1, 1) + torch.einsum('ij,bjk->bik', torch.abs(self.gamma), x_sq) + 1e-12)
            y = x_flat * norm
        else:
            # GDN: x / sqrt(beta + gamma @ x^2)
            x_sq = torch.clamp(x_flat ** 2, min=1e-12)  # Prevent zeros
            norm = torch.sqrt(beta.view(1, -1, 1) + torch.einsum('ij,bjk->bik', torch.abs(self.gamma), x_sq) + 1e-12)
            y = x_flat / (norm + 1e-12)  # Prevent division by zero
            
        return y.view(B, C, D, H, W)


class Conv3DBlock(nn.Module):
    """3D Convolutional block as specified in the paper"""
    def __init__(self, in_channels, out_channels, stride=2):
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # Use BatchNorm instead of GDN for better stability
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DeConv3DBlock(nn.Module):
    """3D Deconvolutional block as specified in the paper"""
    def __init__(self, in_channels, out_channels, stride=2, final_layer=False):
        super(DeConv3DBlock, self).__init__()
        self.deconv1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1)
        self.deconv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.final_layer = final_layer
        
        if not final_layer:
            # Use BatchNorm instead of iGDN for better stability
            self.bn = nn.BatchNorm3d(out_channels)
            self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        if not self.final_layer:
            x = self.bn(x)
            x = self.relu(x)
        return x


class SWAE3DEncoder7x7x7(nn.Module):
    """3D SWAE Encoder for 7x7x7 blocks"""
    def __init__(self, channels=[32, 64, 128], latent_dim=16):
        super(SWAE3DEncoder7x7x7, self).__init__()
        
        # Input: 7x7x7x1 -> Conv blocks
        self.conv_blocks = nn.ModuleList()
        
        # First block: 1 -> 32 channels
        self.conv_blocks.append(Conv3DBlock(1, channels[0], stride=2))  # 7x7x7 -> 4x4x4
        
        # Second block: 32 -> 64 channels  
        self.conv_blocks.append(Conv3DBlock(channels[0], channels[1], stride=2))  # 4x4x4 -> 2x2x2
        
        # Third block: 64 -> 128 channels
        self.conv_blocks.append(Conv3DBlock(channels[1], channels[2], stride=2))  # 2x2x2 -> 1x1x1
        
        # Fully connected layer to latent space
        self.fc = nn.Linear(channels[2], latent_dim)
        
    def forward(self, x):
        # x: (B, 1, 7, 7, 7)
        for block in self.conv_blocks:
            x = block(x)
        
        # x: (B, 128, 1, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)  # (B, latent_dim)
        
        return x


class SWAE3DDecoder7x7x7(nn.Module):
    """3D SWAE Decoder for 7x7x7 blocks"""
    def __init__(self, channels=[128, 64, 32], latent_dim=16):
        super(SWAE3DDecoder7x7x7, self).__init__()
        
        # Fully connected layer from latent space
        self.fc = nn.Linear(latent_dim, channels[0])
        
        # Deconv blocks (reverse of encoder)
        self.deconv_blocks = nn.ModuleList()
        
        # First deconv block: 128 -> 64 channels
        self.deconv_blocks.append(DeConv3DBlock(channels[0], channels[1], stride=2))  # 1x1x1 -> 2x2x2
        
        # Second deconv block: 64 -> 32 channels
        self.deconv_blocks.append(DeConv3DBlock(channels[1], channels[2], stride=2))  # 2x2x2 -> 4x4x4
        
        # Final deconv block: 32 -> 1 channel (no activation)
        # This will go from 4x4x4 to 8x8x8, then we crop to 7x7x7
        self.final_deconv = nn.ConvTranspose3d(channels[2], 1, kernel_size=3, stride=2, padding=1, output_padding=1)  # 4x4x4 -> 8x8x8
        
    def forward(self, x):
        # x: (B, latent_dim)
        x = self.fc(x)  # (B, 128)
        x = x.view(x.size(0), -1, 1, 1, 1)  # (B, 128, 1, 1, 1)
        
        # Apply first two deconv blocks
        for block in self.deconv_blocks:
            x = block(x)
        
        # Final deconv to get 8x8x8
        x = self.final_deconv(x)  # (B, 1, 8, 8, 8)
        
        # Crop to 7x7x7 by removing the last element in each dimension
        x = x[:, :, :7, :7, :7]  # (B, 1, 7, 7, 7)
            
        return x


def sliced_wasserstein_distance(encoded_samples, prior_samples, num_projections=50):
    """
    Compute sliced Wasserstein distance as specified in the paper
    O(n log n) complexity as mentioned in the paper
    """
    batch_size, latent_dim = encoded_samples.shape
    device = encoded_samples.device
    
    # Sample random projections from unit sphere (S^{d-1})
    projections = torch.randn(num_projections, latent_dim, device=device)
    projections = F.normalize(projections, dim=1)
    
    # Project samples onto random directions
    encoded_projections = torch.matmul(encoded_samples, projections.t())  # (batch_size, num_projections)
    prior_projections = torch.matmul(prior_samples, projections.t())      # (batch_size, num_projections)
    
    # Sort projections (this is the O(n log n) step mentioned in paper)
    encoded_projections, _ = torch.sort(encoded_projections, dim=0)
    prior_projections, _ = torch.sort(prior_projections, dim=0)
    
    # Compute L2 distance between sorted projections
    wasserstein_distance = torch.mean((encoded_projections - prior_projections) ** 2)
    
    return wasserstein_distance


class SWAE3D7x7x7(nn.Module):
    """
    Pure SWAE 3D Autoencoder for 7x7x7 U_CHI Data
    Based on "Exploring Autoencoder-based Error-bounded Compression for Scientific Data"
    """
    def __init__(self, channels=[32, 64, 128], latent_dim=16, lambda_reg=10.0):
        super(SWAE3D7x7x7, self).__init__()
        
        self.encoder = SWAE3DEncoder7x7x7(channels, latent_dim)
        self.decoder = SWAE3DDecoder7x7x7(list(reversed(channels)), latent_dim)
        self.lambda_reg = lambda_reg
        self.latent_dim = latent_dim
        
    def forward(self, x):
        # Encode
        z = self.encoder(x)
        
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon, z
    
    def loss_function(self, x, x_recon, z, prior_dist='normal'):
        """
        SWAE Loss function as specified in Equation (1) of the paper:
        L(φ, ψ) = (1/M) * Σ c(x_m, ψ(φ(x_m))) + (λ/LM) * Σ c(θ_l · z_i[m], θ_l · φ(x_j[m]))
        
        Where c(x, y) = ||x - y||^2 (Equation 2)
        """
        batch_size = x.size(0)
        
        # Reconstruction loss: ||x - x_recon||^2
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # Sample from prior distribution (standard normal as commonly used)
        if prior_dist == 'normal':
            prior_samples = torch.randn(batch_size, self.latent_dim, device=x.device)
        else:
            # Could implement other priors here
            prior_samples = torch.randn(batch_size, self.latent_dim, device=x.device)
        
        # Sliced Wasserstein distance between encoded samples and prior
        sw_distance = sliced_wasserstein_distance(z, prior_samples)
        
        # Total loss (Equation 3 in paper)
        total_loss = recon_loss + self.lambda_reg * sw_distance
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'sw_distance': sw_distance
        }
    
    def encode(self, x):
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent space"""
        return self.decoder(z)
    
    def reconstruct(self, x):
        """Full reconstruction pipeline"""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon


def create_swae_3d_7x7x7_model(latent_dim=16, lambda_reg=10.0):
    """
    Create SWAE 3D model for 7x7x7 blocks
    
    Args:
        latent_dim: Latent vector dimension (16 as per Table VI)
        lambda_reg: Regularization weight for SW distance (10.0 default)
    
    Returns:
        SWAE3D7x7x7 model
    """
    # Channels as specified in Table VI for 3D data: [32, 64, 128]
    channels = [32, 64, 128]
    
    model = SWAE3D7x7x7(
        channels=channels,
        latent_dim=latent_dim,
        lambda_reg=lambda_reg
    )
    
    return model


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_swae_3d_7x7x7_model().to(device)
    
    # Test with dummy data (batch of 7x7x7 blocks)
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 7, 7, 7).to(device)
    
    print(f"Input shape: {test_input.shape}")
    
    # Forward pass
    x_recon, z = model(test_input)
    
    print(f"Latent shape: {z.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    
    # Compute loss
    loss_dict = model.loss_function(test_input, x_recon, z)
    
    print(f"Total loss: {loss_dict['loss']:.6f}")
    print(f"Reconstruction loss: {loss_dict['recon_loss']:.6f}")
    print(f"SW distance: {loss_dict['sw_distance']:.6f}")
    
    print("\nSWAE 3D 7x7x7 model created successfully!") 