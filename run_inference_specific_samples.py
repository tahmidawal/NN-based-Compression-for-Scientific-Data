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
# Specific sample indices requested by the user
SPECIFIC_SAMPLE_INDICES = [2706, 1446, 42]

# Define paths based on the model type
if MODEL_TYPE == 'U_CHI':
    TARGET_VARIABLE_NAME_FILTER = 'U_CHI'
    MODEL_PATH = 'bssn_autoencoder_U_CHI_best.keras'
    NORM_PARAMS_PATH = 'normalization_params_U_CHI.npz'
    VTI_OUTPUT_DIR = 'visualizations/vti_files/specific_samples/'
else:
    # Configuration for a different model can be added here
    raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")

DATA_DIR = 'BSSN Extracted Data/tt_q01/'
HDF5_PATTERN = os.path.join(DATA_DIR, '*.hdf5')

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

def generate_vti_files_for_specific_samples(
    model,
    dataset,
    norm_params,
    output_dir,
    sample_indices
):
    """Generates and saves VTI files for original, reconstructed, and error volumes for specific sample indices."""
    print(f"--- Generating VTI files for specific samples: {sample_indices} ---")
    os.makedirs(output_dir, exist_ok=True)

    # Check if the requested indices are valid
    valid_indices = [idx for idx in sample_indices if idx < len(dataset)]
    invalid_indices = [idx for idx in sample_indices if idx >= len(dataset)]
    
    if invalid_indices:
        print(f"Warning: The following indices are out of range (max index: {len(dataset)-1}): {invalid_indices}")
    
    if not valid_indices:
        print("Error: No valid sample indices provided.")
        return
    
    # Get the specific samples
    samples_norm = dataset[valid_indices]

    # Get model reconstructions
    reconstructions_norm = model.predict(samples_norm)

    # Un-normalize data
    mins, maxs = norm_params['mins'], norm_params['maxs']
    samples_unnorm = unnormalize_min_max(samples_norm, mins, maxs)
    reconstructions_unnorm = unnormalize_min_max(reconstructions_norm, mins, maxs)

    for i, sample_idx in enumerate(valid_indices):
        # Extract volumes (assuming single channel for U_CHI)
        original_volume = samples_unnorm[i, :, :, :, 0]
        reconstructed_volume = reconstructions_unnorm[i, :, :, :, 0]
        error_volume = np.abs(original_volume - reconstructed_volume)
        
        # Calculate per-sample losses for the filename
        mse = np.mean((original_volume - reconstructed_volume)**2)
        mae = np.mean(error_volume)
        
        # Save VTI files
        base_path = os.path.join(output_dir, f"test_sample_{sample_idx}")
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
    
    print(f"Saved VTI files for {len(valid_indices)} samples to {output_dir}")

def main():
    """Main function to run the inference and visualization pipeline for specific samples."""
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
    print(f"Requested sample indices: {SPECIFIC_SAMPLE_INDICES}")
    print(f"Test data has {len(test_data)} samples, valid range: 0-{len(test_data)-1}")

    # 4. Normalize Data
    test_data_norm = normalize_min_max(test_data, mins, maxs)

    # 5. Generate VTI files for specific samples
    os.makedirs(VTI_OUTPUT_DIR, exist_ok=True)
    
    generate_vti_files_for_specific_samples(
        model=model,
        dataset=test_data_norm,
        norm_params=norm_data,
        output_dir=VTI_OUTPUT_DIR,
        sample_indices=SPECIFIC_SAMPLE_INDICES
    )

    print("\nSpecific samples inference script finished.")

if __name__ == "__main__":
    main()
