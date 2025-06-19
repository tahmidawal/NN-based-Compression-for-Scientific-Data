# Neural Network-Based Compression for Scientific Data

A deep learning compression system designed for 3D scientific simulation data using convolutional autoencoders.

## Overview

This project implements a neural network-based compression system specifically designed for scientific simulation data. The system uses 3D convolutional autoencoders to compress and decompress volumetric data blocks while preserving critical scientific information.

## Data Format

The system is designed to work with HDF5 files containing:
- `var_data`: Shape (30, 2218, 7, 7, 7) - 30 variables across 2218 spatial locations
- `centers`: Shape (2218, 3) - Spatial coordinates
- `levels`: Shape (2218,) - Refinement levels
- 30 physics variables including: U_ALPHA, U_CHI, U_K, U_GT0-GT2, U_BETA0-BETA2, U_B0-B2, etc.

## Architecture

### 3D Convolutional Autoencoder
- **Input**: 5×5×5 data blocks (core data extracted from 7×7×7 blocks)
- **Encoder**: Compresses 3D blocks to low-dimensional latent vectors
- **Decoder**: Reconstructs original blocks from latent representations
- **Loss Function**: Multi-component loss combining:
  - Weighted MSE for reconstruction accuracy
  - Latent regularization for compression efficiency
  - Physics-aware constraints for domain validity

## Project Structure

```
├── eda.ipynb              # Exploratory data analysis
├── eda.py                 # Data analysis utilities
├── requirements.txt       # Python dependencies
├── data_loader.py         # Data loading and preprocessing (planned)
├── model.py              # Autoencoder architecture (planned)
├── train.py              # Training script (planned)
├── compress.py           # Compression/decompression API (planned)
└── visualizations/       # Generated plots and analysis
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Data Analysis
```bash
python eda.py
```

### Training (Coming Soon)
```bash
python train.py --config config.yaml
```

### Compression/Decompression (Coming Soon)
```python
from compress import compress_data, decompress_data

# Compress data
compressed = compress_data(data_block)

# Decompress data
reconstructed = decompress_data(compressed)
```

## Features

- **High Compression Ratios**: Target 10-50x compression
- **Fast Processing**: Optimized for real-time compression/decompression
- **Scientific Accuracy**: Physics-aware loss functions preserve critical data properties
- **Flexible Architecture**: Configurable compression ratios and quality trade-offs

## Development Status

- [x] Data analysis and exploration
- [ ] Data preprocessing pipeline
- [ ] 3D Convolutional Autoencoder implementation
- [ ] Training pipeline
- [ ] Evaluation metrics and visualization
- [ ] Compression/decompression API
- [ ] Model optimization and quantization

## Dependencies

- Python 3.8+
- TensorFlow/PyTorch
- NumPy
- HDF5/h5py
- Matplotlib
- Seaborn

## Contributing

This is a research project focused on neural network compression for scientific data. Contributions and suggestions are welcome!

## License

[Add your license here]
