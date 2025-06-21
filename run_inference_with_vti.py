import os
import glob
import numpy as np
import h5py
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pyevtk.hl import imageToVTK

# --- Data Loading and Preprocessing Functions ---

def load_and_aggregate_data(pattern, target_variable_name=None):
    """Loads and aggregates data from multiple HDF5 files, optionally filtering for a single variable."""
    print("--- Starting load_and_aggregate_data ---")
    hdf5_files = sorted(glob.glob(pattern))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found matching pattern: {pattern}")
    print(f"Found {len(hdf5_files)} HDF5 files to process.")

    all_var_data = []
    var_names = None

    for i, file_path in enumerate(hdf5_files):
        try:
            with h5py.File(file_path, 'r') as hf:
                current_var_data = hf['var_data'][:]
                all_var_data.append(current_var_data)
                if var_names is None:
                    var_names = [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in hf['vars'][:]]
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            raise

    if not all_var_data:
        raise ValueError("No data loaded from HDF5 files.")

    aggregated_var_data = np.concatenate(all_var_data, axis=1)
    print(f"Initial aggregated var_data shape: {aggregated_var_data.shape}")

    if target_variable_name:
        print(f"Filtering for target variable: {target_variable_name}")
        try:
            target_idx = var_names.index(target_variable_name)
            aggregated_var_data = aggregated_var_data[[target_idx], ...]
            var_names = [var_names[target_idx]]
            print(f"Filtered var_data shape: {aggregated_var_data.shape}, Variable: {var_names}")
        except ValueError:
            print(f"ERROR - Target variable '{target_variable_name}' not found. Available: {var_names}")
            raise
    else:
        print("Using all variables.")

    print("--- Finished load_and_aggregate_data ---")
    return aggregated_var_data, var_names

def normalize_min_max(data, mins, maxs):
    """Normalizes data to [0, 1] range using pre-computed min and max values."""
    data_norm = np.copy(data)
    num_channels = data.shape[-1]
    for i in range(num_channels):
        channel_range = maxs[i] - mins[i]
        if channel_range > 1e-9:
            data_norm[..., i] = (data[..., i] - mins[i]) / channel_range
        else:
            data_norm[..., i] = 0.5
    return data_norm

def unnormalize_min_max(data_norm, mins, maxs):
    """Un-normalizes data from [0, 1] range back to its original scale."""
    data_unnorm = np.copy(data_norm)
    num_channels = data_unnorm.shape[-1]
    for i in range(num_channels):
        channel_range = maxs[i] - mins[i]
        if channel_range > 1e-9:
            data_unnorm[..., i] = (data_unnorm[..., i] * channel_range) + mins[i]
        else:
            data_unnorm[..., i] = mins[i]
    return data_unnorm

# --- Configuration ---
# Define which model and data to use
MODEL_TYPE = 'U_CHI'
NUM_SAMPLES_TO_PLOT = 5  # Number of samples to visualize from each dataset (val and test)

# Define paths based on the model type
if MODEL_TYPE == 'U_CHI':
    TARGET_VARIABLE_NAME_FILTER = 'U_CHI'
    MODEL_PATH = 'bssn_autoencoder_U_CHI_best.keras'
    NORM_PARAMS_PATH = 'normalization_params_U_CHI.npz'
    VIS_OUTPUT_DIR = 'visualizations/inference_results_U_CHI/'
    VTI_OUTPUT_DIR = 'visualizations/vti_files/'
else:
    # Configuration for a different model can be added here
    raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")

DATA_DIR = 'BSSN Extracted Data/tt_q01/'
HDF5_PATTERN = os.path.join(DATA_DIR, '*.hdf5')

def visualize_reconstructions_with_loss(
    model,
    dataset,
    norm_params,
    output_dir,
    dataset_name,
    num_samples=5,
    slice_idx=None
):
    """Generates and saves reconstruction plots with loss values in the title."""
    print(f"--- Generating visualizations for {dataset_name} set ---")
    os.makedirs(output_dir, exist_ok=True)

    # Select random samples from the dataset
    if num_samples > len(dataset):
        print(f"Warning: Requested {num_samples} samples, but dataset only has {len(dataset)}. Using all samples.")
        num_samples = len(dataset)
    
    sample_indices = np.random.choice(len(dataset), num_samples, replace=False)
    samples_norm = dataset[sample_indices]

    # Get model reconstructions
    reconstructions_norm = model.predict(samples_norm)

    # Un-normalize data
    mins, maxs = norm_params['mins'], norm_params['maxs']
    samples_unnorm = unnormalize_min_max(samples_norm, mins, maxs)
    reconstructions_unnorm = unnormalize_min_max(reconstructions_norm, mins, maxs)

    # Determine slice index (middle slice)
    if slice_idx is None:
        slice_idx = samples_unnorm.shape[2] // 2

    for i in range(num_samples):
        original_sample = samples_unnorm[i]
        reconstructed_sample = reconstructions_unnorm[i]

        # Calculate per-sample losses
        mse = np.mean((original_sample - reconstructed_sample)**2)
        mae = np.mean(np.abs(original_sample - reconstructed_sample))

        # Plotting (assuming single channel for U_CHI)
        original_slice = original_sample[:, :, slice_idx, 0]
        reconstructed_slice = reconstructed_sample[:, :, slice_idx, 0]

        # Determine shared color scale for original and reconstructed plots
        vmin = min(original_slice.min(), reconstructed_slice.min())
        vmax = max(original_slice.max(), reconstructed_slice.max())

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        title = f'{dataset_name} Sample (Index: {sample_indices[i]}) - Slice z={slice_idx}\nMSE: {mse:.6f} | MAE: {mae:.6f}'
        fig.suptitle(title, fontsize=16)

        # Original
        im1 = axes[0].imshow(original_slice, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title('Original')
        fig.colorbar(im1, ax=axes[0])

        # Reconstructed
        im2 = axes[1].imshow(reconstructed_slice, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title('Reconstructed')
        fig.colorbar(im2, ax=axes[1])

        # Absolute Error
        diff = np.abs(original_slice - reconstructed_slice)
        im3 = axes[2].imshow(diff, cmap='inferno')
        axes[2].set_title('Absolute Error')
        fig.colorbar(im3, ax=axes[2])

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f'{dataset_name}_reconstruction_sample_idx_{sample_indices[i]}.png')
        plt.savefig(save_path)
        plt.close(fig)

    print(f"Saved {num_samples} reconstruction plots to {output_dir}")

def save_to_vti(data, output_path, spacing=(1.0, 1.0, 1.0)):
    """
    Save a 3D numpy array to VTI format for ParaView visualization.
    
    Parameters:
    - data: 3D numpy array (D, H, W)
    - output_path: Path where to save the VTI file (without extension)
    - spacing: Tuple of (dx, dy, dz) spacing between points
    """
    # Ensure the data is 3D
    if len(data.shape) != 3:
        raise ValueError(f"Expected 3D data but got shape {data.shape}")
    
    # VTK uses Fortran order (z, y, x)
    # Convert from our (x, y, z) to (z, y, x)
    data_vtk = np.transpose(data, (2, 1, 0))
    
    # Create the VTI file
    imageToVTK(output_path, cellData={"data": data_vtk}, spacing=spacing)
    print(f"Saved VTI file to {output_path}.vti")

def generate_vti_files(
    model,
    dataset,
    norm_params,
    output_dir,
    dataset_name,
    num_samples=5
):
    """Generates and saves VTI files for original, reconstructed, and error volumes."""
    print(f"--- Generating VTI files for {dataset_name} set ---")
    os.makedirs(output_dir, exist_ok=True)

    # Select random samples from the dataset
    if num_samples > len(dataset):
        print(f"Warning: Requested {num_samples} samples, but dataset only has {len(dataset)}. Using all samples.")
        num_samples = len(dataset)
    
    sample_indices = np.random.choice(len(dataset), num_samples, replace=False)
    samples_norm = dataset[sample_indices]

    # Get model reconstructions
    reconstructions_norm = model.predict(samples_norm)

    # Un-normalize data
    mins, maxs = norm_params['mins'], norm_params['maxs']
    samples_unnorm = unnormalize_min_max(samples_norm, mins, maxs)
    reconstructions_unnorm = unnormalize_min_max(reconstructions_norm, mins, maxs)

    for i in range(num_samples):
        sample_idx = sample_indices[i]
        
        # Extract volumes (assuming single channel for U_CHI)
        original_volume = samples_unnorm[i, :, :, :, 0]
        reconstructed_volume = reconstructions_unnorm[i, :, :, :, 0]
        error_volume = np.abs(original_volume - reconstructed_volume)
        
        # Calculate per-sample losses for the filename
        mse = np.mean((original_volume - reconstructed_volume)**2)
        mae = np.mean(error_volume)
        
        # Save VTI files
        base_path = os.path.join(output_dir, f"{dataset_name}_sample_{sample_idx}")
        save_to_vti(original_volume, f"{base_path}_original", spacing=(1.0, 1.0, 1.0))
        save_to_vti(reconstructed_volume, f"{base_path}_reconstructed", spacing=(1.0, 1.0, 1.0))
        save_to_vti(error_volume, f"{base_path}_error", spacing=(1.0, 1.0, 1.0))
        
        # Save metadata file with loss values
        with open(f"{base_path}_metadata.txt", 'w') as f:
            f.write(f"Sample Index: {sample_idx}\n")
            f.write(f"MSE: {mse:.6f}\n")
            f.write(f"MAE: {mae:.6f}\n")
            f.write(f"Original Shape: {original_volume.shape}\n")
            f.write(f"Value Range: [{original_volume.min():.6f}, {original_volume.max():.6f}]\n")
    
    print(f"Saved {num_samples} sets of VTI files to {output_dir}")

def main():
    """Main function to run the inference and visualization pipeline."""
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Running on GPU: {gpus}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("Running on CPU")

    # 1. Load Model and Normalization Parameters
    print(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.summary()

    print(f"Loading normalization parameters from: {NORM_PARAMS_PATH}")
    norm_data = np.load(NORM_PARAMS_PATH, allow_pickle=True)
    mins = norm_data['mins']
    maxs = norm_data['maxs']
    var_names_model = norm_data['var_names']

    # 2. Load and Process Full Dataset
    print(f"Loading and aggregating data from: {HDF5_PATTERN}")
    aggregated_data, var_names_loaded = load_and_aggregate_data(HDF5_PATTERN, target_variable_name=TARGET_VARIABLE_NAME_FILTER)
    data_permuted = np.transpose(aggregated_data, (1, 2, 3, 4, 0))

    # 3. Split Data into Train, Validation, and Test sets
    print("Splitting data into training, validation, and test sets...")
    train_val_data, test_data = train_test_split(data_permuted, test_size=0.15, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=(0.15/0.85), random_state=42) # 0.15/0.85 gives correct validation proportion

    print(f"Test data shape: {test_data.shape}")
    print(f"Validation data shape: {val_data.shape}")

    # 4. Normalize Data
    val_data_norm = normalize_min_max(val_data, mins, maxs)
    test_data_norm = normalize_min_max(test_data, mins, maxs)

    # 5. Run Visualization
    # Create specific output directories for validation and test plots
    val_plot_dir = os.path.join(VIS_OUTPUT_DIR, 'validation_plots')
    test_plot_dir = os.path.join(VIS_OUTPUT_DIR, 'test_plots')
    
    # Create VTI output directories
    vti_val_dir = os.path.join(VTI_OUTPUT_DIR, 'validation')
    vti_test_dir = os.path.join(VTI_OUTPUT_DIR, 'test')
    os.makedirs(vti_val_dir, exist_ok=True)
    os.makedirs(vti_test_dir, exist_ok=True)

    # Visualize validation set
    visualize_reconstructions_with_loss(
        model=model,
        dataset=val_data_norm,
        norm_params=norm_data,
        output_dir=val_plot_dir,
        dataset_name='validation',
        num_samples=NUM_SAMPLES_TO_PLOT
    )

    # Visualize test set
    visualize_reconstructions_with_loss(
        model=model,
        dataset=test_data_norm,
        norm_params=norm_data,
        output_dir=test_plot_dir,
        dataset_name='test',
        num_samples=NUM_SAMPLES_TO_PLOT
    )
    
    # Generate VTI files for test set only (to save space)
    generate_vti_files(
        model=model,
        dataset=test_data_norm,
        norm_params=norm_data,
        output_dir=vti_test_dir,
        dataset_name='test',
        num_samples=NUM_SAMPLES_TO_PLOT
    )

    print("\nInference script finished.")

if __name__ == "__main__":
    main()
