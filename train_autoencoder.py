import os
import glob
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Conv3DTranspose, BatchNormalization, LeakyReLU, Cropping3D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime

# --- Configuration ---
TARGET_VARIABLE_NAME = 'U_CHI' # Set to None or 'ALL' to use all variables, or specify a name like 'U_CHI'
DATA_DIR = 'BSSN Extracted Data/tt_q01/'
HDF5_PATTERN = os.path.join(DATA_DIR, '*.hdf5')

# Adjust save paths based on whether a single variable is targeted
# Adding _zscore to filenames to distinguish from min-max models
if TARGET_VARIABLE_NAME and TARGET_VARIABLE_NAME != 'ALL':
    MODEL_SAVE_PATH = f'bssn_autoencoder_{TARGET_VARIABLE_NAME}_zscore_best.keras'
    NORM_PARAMS_SAVE_PATH = f'normalization_params_{TARGET_VARIABLE_NAME}_zscore.npz'
    VIS_FILENAME_PREFIX = f'{TARGET_VARIABLE_NAME}_zscore_'
else:
    MODEL_SAVE_PATH = 'bssn_autoencoder_all_vars_zscore_best.keras'
    NORM_PARAMS_SAVE_PATH = 'normalization_params_all_vars_zscore.npz'
    VIS_FILENAME_PREFIX = 'zscore_'

LOG_DIR_BASE = "logs/fit/"
LOG_DIR = LOG_DIR_BASE + (f"{TARGET_VARIABLE_NAME}_") if TARGET_VARIABLE_NAME and TARGET_VARIABLE_NAME != 'ALL' else LOG_DIR_BASE + "all_vars_"
LOG_DIR += datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
NUM_EPOCHS = 1500 # Increased epochs, with early stopping
BATCH_SIZE = 32
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1 # Relative to remaining data after val_split, so 0.09 of total

# --- 1. Load and Aggregate Data ---
def load_and_aggregate_data(pattern):
    """Loads and aggregates data from multiple HDF5 files."""
    print("--- Starting load_and_aggregate_data ---")
    hdf5_files = sorted(glob.glob(pattern))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found matching pattern: {pattern}")
    print(f"Found {len(hdf5_files)} HDF5 files to process: {hdf5_files}")

    all_var_data = []
    var_names = None
    
    print("Starting loop through HDF5 files...")
    for i, file_path in enumerate(hdf5_files):
        print(f"Processing file {i+1}/{len(hdf5_files)}: {file_path}")
        try:
            with h5py.File(file_path, 'r') as hf:
                print(f"  Opened HDF5 file. Keys: {list(hf.keys())}")
                
                current_var_data = hf['var_data'][:]
                print(f"  Loaded 'var_data', shape: {current_var_data.shape}")
                all_var_data.append(current_var_data)

                if var_names is None:
                    # Handle both bytes and strings for compatibility
                    var_names = [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in hf['vars'][:]]
                    print(f"  Loaded variable names: {var_names}")
                else:
                    # Sanity check: ensure variable names are consistent across files
                    current_names = [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in hf['vars'][:]]
                    if current_names != var_names:
                        print(f"Warning: Inconsistent variable names in {file_path}. Using names from first file.")
                        # To be robust, we'll just use the first set of names and assume the order is the same.
                        # For a stricter check, you could raise an error:
                        # raise ValueError(f"Inconsistent variable names in {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            raise
    
    if not all_var_data:
        raise ValueError("No data loaded from HDF5 files.")
    print("Finished loop through HDF5 files.")

    print("Concatenating all loaded data...")
    # Resulting shape: (n_vars, total_n_blocks, D, H, W)
    aggregated_var_data = np.concatenate(all_var_data, axis=1)
    print(f"Initial aggregated var_data shape: {aggregated_var_data.shape}")

    if TARGET_VARIABLE_NAME and TARGET_VARIABLE_NAME != 'ALL':
        print(f"Filtering for target variable: {TARGET_VARIABLE_NAME}")
        try:
            target_idx = var_names.index(TARGET_VARIABLE_NAME)
            print(f"Found target variable '{TARGET_VARIABLE_NAME}' at index {target_idx}")
            aggregated_var_data = aggregated_var_data[[target_idx], ...]
            var_names = [var_names[target_idx]]
            print(f"Filtered var_data shape: {aggregated_var_data.shape}, Variable: {var_names}")
        except ValueError:
            print(f"ERROR - Target variable '{TARGET_VARIABLE_NAME}' not found. Available: {var_names}")
            raise
    else:
        print("Using all variables for training.")

    print("--- Finished load_and_aggregate_data ---")
    return aggregated_var_data, var_names

# --- 2. Preprocess Data ---
def normalize_z_score(data, means, stds):
    """Normalizes data using Z-score standardization."""
    # Add a small epsilon to std deviation to avoid division by zero
    return (data - means) / (stds + 1e-9)

def unnormalize_z_score(data_norm, means, stds):
    """Un-normalizes data from Z-score back to its original scale."""
    return (data_norm * (stds + 1e-9)) + means

def preprocess_data(data, var_names):
    """Permutes, splits, and normalizes the data using Z-score standardization."""
    # Permute from (n_vars, n_samples, D, H, W) to (n_samples, D, H, W, n_vars)
    data_permuted = np.transpose(data, (1, 2, 3, 4, 0))
    print(f"Permuted data shape: {data_permuted.shape}")

    # Split data into training, validation, and test sets
    print("Splitting data into training, validation, and test sets...")
    train_data, temp_data = train_test_split(
        data_permuted, test_size=(VAL_SPLIT + TEST_SPLIT), random_state=42, shuffle=True
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=(TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)), random_state=42, shuffle=True
    )
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Calculate mean and std for each channel from the TRAINING SET ONLY
    print("Calculating mean/std from training data for Z-score normalization...")
    means = np.mean(train_data, axis=(0, 1, 2, 3))
    stds = np.std(train_data, axis=(0, 1, 2, 3))

    # Identify channels with near-zero std deviation to handle them separately
    constant_indices = np.where(stds < 1e-9)[0]
    if len(constant_indices) > 0:
        print(f"Found {len(constant_indices)} constant channels (std dev < 1e-9):")
        for idx in constant_indices:
            print(f"  - Channel {idx} ('{var_names[idx]}') is constant with value {means[idx]}")

    # Save normalization parameters
    np.savez(NORM_PARAMS_SAVE_PATH, means=means, stds=stds, var_names=var_names, constant_indices=constant_indices)
    print(f"Normalization parameters (means, stds) saved to {NORM_PARAMS_SAVE_PATH}")

    # Normalize all datasets using the parameters from the training set
    print("Normalizing all datasets using Z-score...")
    train_data_norm = normalize_z_score(train_data, means, stds)
    val_data_norm = normalize_z_score(val_data, means, stds)
    test_data_norm = normalize_z_score(test_data, means, stds)
    
    return train_data_norm, val_data_norm, test_data_norm, var_names

# --- 3. Define Autoencoder Model ---
def build_autoencoder(input_shape):
    # input_shape is (D, H, W, N_vars)
    inputs = Input(shape=input_shape)
    
    # Encoder
    # Block 1: 7x7x7 -> 4x4x4 (approx, due to padding='same')
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)
    
    # Block 2: 4x4x4 -> 2x2x2
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    encoded = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x) # Bottleneck: (None, 2, 2, 2, 128)
    
    # Decoder
    # Block 3: 2x2x2 -> 4x4x4
    x = Conv3DTranspose(filters=128, kernel_size=(3, 3, 3), strides=(2,2,2), padding='same')(encoded)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Block 4: 4x4x4 -> 8x8x8. Then crop to 7x7x7.
    x = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3), strides=(2,2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    # Output of Conv3DTranspose with strides=2 and padding='same' might be D*2, H*2, W*2
    # If input D=7, first MaxPool gives 4 (ceil(7/2)). Second MaxPool gives 2 (ceil(4/2)).
    # First UpSample (Conv3DTranspose) gives 2*2=4. Second UpSample gives 4*2=8.
    # So we have 8x8x8, need to crop to 7x7x7.
    # Cropping: ((top, bottom), (left, right), (front, back))
    # We need to crop 1 from one side of each dimension if the original was odd.
    # Assuming input_shape[0,1,2] are D,H,W which are 7,7,7
    # Current shape is (None, 8, 8, 8, 64)
    # We need to calculate cropping based on original input_shape to be general
    # However, for 7->8, we crop 1 from the end of each of the first 3 dims.
    # target_dims = input_shape[:3] # (7,7,7)
    # current_dims = x.shape[1:4] # (8,8,8)
    # crop_needed = [(0, cd - td) for cd, td in zip(current_dims, target_dims)] # [(0,1), (0,1), (0,1)]
    x = Cropping3D(cropping=((0,1), (0,1), (0,1)))(x) # Crop 8x8x8 to 7x7x7

    # Output Layer
    # CRITICAL CHANGE: Use 'linear' activation for Z-scored data
    decoded = Conv3D(input_shape[-1], (3, 3, 3), activation='linear', padding='same')(x)
    
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    autoencoder.summary()
    return autoencoder

# --- 4. Visualization ---
def visualize_reconstruction(model, test_data, norm_params_path, vis_filename_prefix='', num_samples_to_plot=3, slice_idx=3):
    # Load normalization parameters
    norm_data = np.load(norm_params_path, allow_pickle=True)
    means = norm_data['means']
    stds = norm_data['stds']
    var_names = norm_data['var_names']

    # Get reconstructions
    reconstructed_data_norm = model.predict(test_data[:num_samples_to_plot])
    original_data_norm = test_data[:num_samples_to_plot]

    # Unnormalize the test data to get the original scale for comparison
    original_test_unnorm = unnormalize_z_score(test_data, means, stds)
    reconstructed_data_unnorm = unnormalize_z_score(reconstructed_data_norm, means, stds)

    # Plot
    # Select a few non-constant channels to visualize
    channels_to_plot_names = ['U_ALPHA', 'U_CHI', 'U_K', 'C_HAM', 'C_PSI4_REAL']
    channels_to_plot_indices = []
    for name in channels_to_plot_names:
        try:
            idx = list(var_names).index(name)
            if idx not in constant_indices:
                channels_to_plot_indices.append(idx)
        except ValueError:
            print(f"Warning: Variable '{name}' not found in var_names.")
    
    if not channels_to_plot_indices:
        print("No suitable non-constant channels found for plotting. Selecting first few available.")
        channels_to_plot_indices = [i for i in range(min(test_data.shape[-1], 3)) if i not in constant_indices][:3]

    for sample_idx in range(num_samples_to_plot):
        for channel_idx in channels_to_plot_indices:
            original_slice = original_data_unnorm[sample_idx, :, :, slice_idx, channel_idx]
            reconstructed_slice = reconstructed_data_unnorm[sample_idx, :, :, slice_idx, channel_idx]
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f"Sample {sample_idx + 1}, Var: {var_names[channel_idx]}, Slice z={slice_idx}", fontsize=16)
            
            # Original
            im1 = axes[0].imshow(original_slice, aspect='auto')
            axes[0].set_title("Original")
            fig.colorbar(im1, ax=axes[0])
            
            # Reconstructed
            im2 = axes[1].imshow(reconstructed_slice, aspect='auto')
            axes[1].set_title("Reconstructed")
            fig.colorbar(im2, ax=axes[1])
            
            # Absolute Error
            diff = np.abs(original_slice - reconstructed_slice)
            im3 = axes[2].imshow(diff, aspect='auto')
            axes[2].set_title("Absolute Error")
            fig.colorbar(im3, ax=axes[2])
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(f"{vis_filename_prefix}reconstruction_sample{sample_idx}_var_{var_names[channel_idx].replace('/', '_')}_slice{slice_idx}.png")
            plt.close(fig)
    print(f"Reconstruction visualizations saved for {num_samples_to_plot} samples and selected channels.")

# --- Main Execution ---
if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Running on GPU: {gpus}")
        except RuntimeError as e:
            print(e)
    else:
        print("Running on CPU")

    # 1. Load Data
    aggregated_data, var_names_list = load_and_aggregate_data(HDF5_PATTERN)
    
    # 2. Preprocess Data
    train_data, val_data, test_data, var_names = preprocess_data(aggregated_data, var_names_list)
    
    # 3. Build Model
    # Input shape for the model is (D, H, W, N_vars)
    model_input_shape = train_data.shape[1:] 
    autoencoder_model = build_autoencoder(model_input_shape)
    
    # 4. Train Model
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True), # Increased patience
        TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
    ]
    
    print("Starting model training...")
    history = autoencoder_model.fit(
        train_data, train_data, 
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(val_data, val_data),
        callbacks=callbacks
    )
    
    print("Model training finished.")
    
    # 5. Evaluate Model
    print("Evaluating model on test data...")
    # Load the best saved model for evaluation
    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    test_loss, test_mae = best_model.evaluate(test_data, test_data, verbose=1)
    print(f"Test Set Reconstruction Loss (MSE): {test_loss}")
    print(f"Test Set Mean Absolute Error (MAE): {test_mae}")
    
    # 6. Visualize Reconstruction
    print("Visualizing reconstructions...")
    visualize_reconstruction(best_model, test_data, NORM_PARAMS_SAVE_PATH, vis_filename_prefix=VIS_FILENAME_PREFIX, num_samples_to_plot=3, slice_idx=test_data.shape[2]//2)
    
    print("Script finished.")
