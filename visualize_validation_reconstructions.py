"""
Script to visualize reconstructions on the validation dataset using the best trained autoencoder model.
Ensures correct Min-Max scaling is applied for preprocessing and postprocessing.
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

# Import necessary functions from the training script
from train_autoencoder import load_and_aggregate_data
from sklearn.model_selection import train_test_split

# --- Normalization Functions (to match the Min-Max model) ---
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
# This script is now intended for a specific model (e.g., U_CHI or all_vars)
# The TARGET_VARIABLE_NAME_FILTER determines which variable to *process* if the model handles multiple.
# For a single-variable model like U_CHI, this should match the variable the model was trained on.
MODEL_TYPE = 'U_CHI' # Options: 'U_CHI', 'ALL_VARS', or others if you train more specific models

DATA_DIR = 'BSSN Extracted Data/tt_q01/'
HDF5_PATTERN = os.path.join(DATA_DIR, '*.hdf5')

if MODEL_TYPE == 'U_CHI':
    TARGET_VARIABLE_NAME_FILTER = 'U_CHI' # This must match the variable the model was trained for
    MODEL_PATH = 'bssn_autoencoder_U_CHI_best.keras'
    NORM_PARAMS_PATH = 'normalization_params_U_CHI.npz'
    VIS_OUTPUT_DIR = 'visualizations/validation_reconstructions_U_CHI/'
    VIS_FILENAME_PREFIX = 'U_CHI_'
elif MODEL_TYPE == 'ALL_VARS':
    TARGET_VARIABLE_NAME_FILTER = None # Process all variables the model was trained on
    MODEL_PATH = 'bssn_autoencoder_all_vars_best.keras'
    NORM_PARAMS_PATH = 'normalization_params_all_vars.npz'
    VIS_OUTPUT_DIR = 'visualizations/validation_reconstructions_all_vars/'
    VIS_FILENAME_PREFIX = ''
else:
    raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}. Choose 'U_CHI' or 'ALL_VARS'.")

LOG_DIR_BASE = "logs/viz_val/"
LOG_DIR = LOG_DIR_BASE + f"{MODEL_TYPE}_" + datetime.now().strftime("%Y%m%d-%H%M%S")

NUM_SAMPLES_TO_PLOT = 3 # Number of validation samples to visualize

# --- Main Visualization Logic ---
def visualize_reconstruction(model, data_subset, dataset_name, norm_params_path, output_dir, vis_filename_prefix='', num_samples_to_plot=3, slice_idx=None):
    """
    Visualizes reconstructions for given validation samples.
    val_data_norm: Normalized validation data (samples, D, H, W, C)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Load normalization parameters
    try:
        norm_data = np.load(norm_params_path, allow_pickle=True)
        mins = norm_data['mins']
        maxs = norm_data['maxs']
        var_names = norm_data['var_names']
        constant_indices = norm_data.get('constant_indices', np.array([])) # Handle if not saved
    except FileNotFoundError:
        print(f"Error: Normalization parameters file not found at {norm_params_path}")
        print("Please ensure train_autoencoder.py has been run successfully first for the Min-Max model.")
        return

    if data_subset.shape[0] < num_samples_to_plot:
        print(f"Warning: Requested {num_samples_to_plot} samples, but validation set only has {data_subset.shape[0]}. Plotting all available.")
        num_samples_to_plot = data_subset.shape[0]

    selected_val_samples_norm = data_subset[:num_samples_to_plot]

    # Get reconstructions
    print(f"Generating reconstructions for {num_samples_to_plot} validation samples...")
    reconstructed_val_norm = model.predict(selected_val_samples_norm)

    # Un-normalize data for visualization
    original_val_unnorm = unnormalize_min_max(selected_val_samples_norm, mins, maxs)
    reconstructed_val_unnorm = unnormalize_min_max(reconstructed_val_norm, mins, maxs)

    # Determine slice index if not provided (e.g., middle slice of depth)
    if slice_idx is None:
        slice_idx = original_val_unnorm.shape[3] // 2 # D is at index 1, H at 2, W at 3 (0-indexed after sample dim)

    # Plot
    # Select a few non-constant channels to visualize (similar to train_autoencoder.py)
    channels_to_plot_names = ['U_CHI'] # Only U_CHI for this model
    channels_to_plot_indices = []
    for name in channels_to_plot_names:
        try:
            idx = list(var_names).index(name)
            if idx not in constant_indices:
                channels_to_plot_indices.append(idx)
        except ValueError:
            print(f"Warning: Variable '{name}' not found in var_names during validation plotting.")
    
    if not channels_to_plot_indices:
        print("No suitable non-constant channels found for plotting. Selecting first few available.")
        available_channels = min(original_val_unnorm.shape[-1], 5) # Plot up to 5 channels
        channels_to_plot_indices = [i for i in range(available_channels) if i not in constant_indices][:3]
        if not channels_to_plot_indices and available_channels > 0:
             channels_to_plot_indices = [0] # Fallback to first channel if all are constant somehow

    if not channels_to_plot_indices:
        print("Error: Could not find any channels to plot.")
        return

    print(f"Plotting reconstructions for variables: {[var_names[i] for i in channels_to_plot_indices]}")

    for sample_idx in range(num_samples_to_plot):
        for channel_idx in channels_to_plot_indices:
            original_slice = original_val_unnorm[sample_idx, :, :, slice_idx, channel_idx]
            reconstructed_slice = reconstructed_val_unnorm[sample_idx, :, :, slice_idx, channel_idx]
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            title = f"Validation Sample {sample_idx}, Var: {var_names[channel_idx]}, Slice z={slice_idx}"
            fig.suptitle(title, fontsize=16)
            
            # Original
            im1 = axes[0].imshow(original_slice, aspect='auto', origin='lower')
            axes[0].set_title("Original (Validation)")
            fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
            
            # Reconstructed
            im2 = axes[1].imshow(reconstructed_slice, aspect='auto', origin='lower')
            axes[1].set_title("Reconstructed")
            fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Absolute Error
            diff = np.abs(original_slice - reconstructed_slice)
            im3 = axes[2].imshow(diff, aspect='auto', origin='lower')
            axes[2].set_title("Absolute Error")
            fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(output_dir, f"{vis_filename_prefix}{dataset_name}_reconstruction_sample{sample_idx}_var_{var_names[channel_idx].replace('/', '_')}_slice{slice_idx}.png"))
            plt.close(fig)
            print(f"Saved: {output_dir}")

    print(f"Validation reconstruction visualizations saved to {output_dir}")


if __name__ == "__main__":
    print("Starting validation set reconstruction visualization script...")

    # --- GPU Configuration ---
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

    # --- 1. Load Normalization Parameters ---
    print(f"Loading normalization parameters from: {NORM_PARAMS_PATH}")
    try:
        norm_data = np.load(NORM_PARAMS_PATH, allow_pickle=True)
        mins = norm_data['mins']
        maxs = norm_data['maxs']
        var_names_model = norm_data['var_names']
    except FileNotFoundError:
        print(f"Error: Normalization parameters file not found at {NORM_PARAMS_PATH}")
        print("Please ensure train_autoencoder.py has been run successfully for the Min-Max model.")
        exit(1)

    # --- 2. Load and Process Data ---
    print(f"Loading and aggregating data using pattern: {HDF5_PATTERN}")
    try:
        aggregated_data, var_names_loaded = load_and_aggregate_data(HDF5_PATTERN, target_variable_name=TARGET_VARIABLE_NAME_FILTER)
        
        if not np.array_equal(var_names_model, var_names_loaded):
             print(f"Warning: Var names in norm file {var_names_model} do not match loaded data vars {var_names_loaded}.")

        # Permute from (n_vars, n_samples, D, H, W) to (n_samples, D, H, W, n_vars)
        data_permuted = np.transpose(aggregated_data, (1, 2, 3, 4, 0))
        
        # Split data to get the same validation set as in training (same random_state)
        _, val_test_data = train_test_split(data_permuted, test_size=0.3, random_state=42)
        val_data, _ = train_test_split(val_test_data, test_size=0.5, random_state=42)
        
        print(f"Validation data shape (original scale): {val_data.shape}")

        # Normalize the validation data using loaded parameters
        val_data_normalized = normalize_min_max(val_data, mins, maxs)
        print(f"Validation data shape (normalized): {val_data_normalized.shape}")

    except Exception as e:
        print(f"An unexpected error occurred during data processing: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # --- 3. Load Model ---
    print(f"Loading best trained model from: {MODEL_PATH}")
    try:
        # We don't need to compile the model for inference
        best_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        best_model.summary()
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # --- 4. Visualize Reconstructions ---
    dynamic_slice_idx = val_data_normalized.shape[2] // 2
    
    visualize_reconstruction(
        model=best_model,
        data_subset=val_data_normalized,
        dataset_name='validation',
        norm_params_path=NORM_PARAMS_PATH,
        output_dir=VIS_OUTPUT_DIR,
        vis_filename_prefix=VIS_FILENAME_PREFIX,
        num_samples_to_plot=NUM_SAMPLES_TO_PLOT,
        slice_idx=dynamic_slice_idx
    )
    
    print("Validation set reconstruction visualization script finished.")

