"""
Model Trainer Module
====================
Builds and trains Direct NN (feedforward neural network) models for surrogate modeling.
"""

import json
from pathlib import Path
import numpy as np
from datetime import datetime
from .direct_nn_model import DirectNNModel
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def analyze_dataset_for_training(dataset_dir):
    """
    Analyze dataset to determine optimal model architecture.

    Parameters
    ----------
    dataset_dir : Path
        Dataset directory

    Returns
    -------
    dict
        Analysis results with recommendations
    """
    analysis = {
        'dataset_found': False,
        'num_samples': 0,
        'input_dim': 0,
        'output_dim': 0,
        'output_size': 0,
        'recommended_latent_size': 0,
        'recommended_architecture': None,
        'errors': []
    }

    try:
        # Try to get input dimensions from model_setup.json
        setup_file = dataset_dir / "model_setup.json"
        if setup_file.exists():
            with open(setup_file, 'r') as f:
                setup_data = json.load(f)

            # Count DOE parameters
            doe_config = setup_data.get('doe_configuration', {})
            num_inputs = 0
            for params in doe_config.values():
                num_inputs += len([v for v in params.values() if v])

            analysis['input_dim'] = num_inputs
        else:
            # Fallback to CSV if setup file not found
            input_csv = dataset_dir / "inputs" / "input_variables.csv"
            if input_csv.exists():
                import csv
                with open(input_csv, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    analysis['input_dim'] = len(header) - 1  # Exclude simulation_id
            else:
                analysis['errors'].append("Could not determine input dimensions")

        # Check for output files
        outputs_dir = dataset_dir / "dataset"
        if outputs_dir.exists():
            output_files = list(outputs_dir.glob("sim_*.npz"))

            if output_files:
                # Count samples from number of output files
                analysis['num_samples'] = len(output_files)

                # Load first file to get dimensions
                try:
                    sample_data = np.load(output_files[0], allow_pickle=True)

                    # Concatenate all output arrays to get total output size
                    total_size = 0
                    for key in sample_data.files:
                        arr = sample_data[key]

                        # Handle different data formats
                        if arr.dtype == object:
                            # If it's an object array
                            if arr.ndim == 0:
                                # 0-d object array - extract the object
                                obj = arr.item()
                                if isinstance(obj, dict):
                                    # Dict contains arrays - count total values in all arrays
                                    for v in obj.values():
                                        if isinstance(v, np.ndarray):
                                            total_size += v.size
                                        elif isinstance(v, (list, tuple)):
                                            total_size += len(v)
                                        else:
                                            total_size += 1
                                elif isinstance(obj, (list, tuple)):
                                    total_size += len(obj)
                                else:
                                    total_size += 1
                            else:
                                total_size += arr.size
                        else:
                            total_size += arr.size

                    analysis['output_dim'] = len(sample_data.files)
                    analysis['output_size'] = total_size
                    analysis['dataset_found'] = True

                    # Direct NN doesn't use latent compression
                    analysis['recommended_latent_size'] = None
                except Exception as e:
                    analysis['errors'].append(f"Error loading output file: {e}")
            else:
                analysis['errors'].append("No simulation output files found")
        else:
            analysis['errors'].append("Outputs directory not found")

        # Recommend Direct NN hyperparameters
        if analysis['dataset_found']:
            input_dim = analysis['input_dim']

            # Adaptive hyperparameters based on dataset characteristics
            # All parameters scale smoothly with dataset characteristics

            n_samples = analysis['num_samples']
            n_outputs = analysis['output_size']

            # Learning rate: logarithmic decay with sample count
            # Small datasets (10-50): 0.0003 - 0.0005
            # Medium datasets (50-200): 0.0005 - 0.0008
            # Large datasets (200+): 0.0008 - 0.001
            lr_base = 0.0003
            lr_scale = min(1.0, n_samples / 200.0)  # Scale from 0 to 1 over 0-200 samples
            learning_rate = lr_base + (0.0007 * lr_scale)  # Range: 0.0003 to 0.001

            # Batch size: conservative for small datasets
            # Aim for 8-12 batches per epoch minimum for stable gradients
            # For 49 samples: ~12-16 batch size → 3-4 batches (too few!)
            # Better: use 16-20 → ensures at least 2-3 batches
            if n_samples < 50:
                batch_size = max(12, min(20, n_samples // 3))
            elif n_samples < 200:
                batch_size = max(16, min(24, n_samples // 8))
            else:
                batch_size = max(24, min(32, n_samples // 10))

            # Epochs: inverse relationship with sample count
            # Small datasets need more iterations to learn
            # Large datasets converge faster
            epochs_base = 500
            epochs_scale = max(1.0, 100.0 / n_samples)  # Inverse scaling
            epochs = int(epochs_base * epochs_scale)
            epochs = min(max(epochs, 500), 3000)  # Clamp to [500, 3000]

            # Early stopping patience: scales with epochs
            # ~5-10% of total epochs
            early_stop_patience = max(20, int(epochs * 0.05))

            # LR scheduler patience: ~2-3% of total epochs
            lr_patience = max(10, int(epochs * 0.02))

            analysis['recommended_architecture'] = {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs,
                'validation_split': 0.2,
                'early_stop_patience': early_stop_patience,
                'lr_patience': lr_patience
            }

    except Exception as e:
        analysis['errors'].append(str(e))

    return analysis




def model_training_menu(dataset_dir, ui_helpers):
    """
    Menu for model training configuration and execution.

    Parameters
    ----------
    dataset_dir : Path
        Dataset directory
    ui_helpers : module
        UI helpers module
    """
    # Analyze dataset
    print("Analyzing dataset...")
    analysis = analyze_dataset_for_training(dataset_dir)

    if not analysis['dataset_found']:
        ui_helpers.clear_screen()
        ui_helpers.print_header("DATASET NOT READY")
        print("\n[X] Dataset is not complete. Please run simulations first.")
        if analysis['errors']:
            print("\nErrors:")
            for error in analysis['errors']:
                print(f"  - {error}")
        ui_helpers.pause()
        return

    # Default configuration (Direct NN only)
    config = analysis['recommended_architecture'].copy()

    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("MODEL TRAINING")

        print(f"\nDataset: {dataset_dir.name}")
        print(f"Samples: {analysis['num_samples']}")
        print(f"Input Dimensions: {analysis['input_dim']}")
        print(f"Output Size: {analysis['output_size']:,} values")

        print("\n" + "="*70)
        print(f"DIRECT NN ARCHITECTURE")
        print("="*70)

        # Direct NN - auto-calculated architecture
        print(f"  Type: Direct Feedforward Neural Network")
        print(f"  Architecture: Auto-calculated based on output size")
        if analysis['output_size'] < 500:
            print(f"  Hidden layers: [64, 128, 64]")
        elif analysis['output_size'] < 2000:
            print(f"  Hidden layers: [128, 256, 512, 256, 128]")
        elif analysis['output_size'] < 5000:
            print(f"  Hidden layers: [256, 512, 1024, 512, 256]")
        else:
            print(f"  Hidden layers: [512, 1024, 2048, 1024, 512]")

        print(f"\n  Training Settings:")
        print(f"    Learning Rate: {config['learning_rate']}")
        print(f"    Batch Size: {config['batch_size']}")
        print(f"    Epochs: {config['epochs']}")
        print(f"    Validation Split: {config['validation_split']*100:.0f}%")

        print(f"\n{'='*70}")
        print("  [1] Modify Training Settings")
        print("  [2] Start Training")
        print("  [3] Load Existing Model")
        print("  [0] Back")
        print("="*70)

        choice = ui_helpers.get_choice(3)

        if choice == 0:
            return
        elif choice == 1:
            config = modify_model_config(config, analysis, ui_helpers)
        elif choice == 2:
            train_model(dataset_dir, config, analysis, ui_helpers)
        elif choice == 3:
            load_existing_model(dataset_dir, ui_helpers)


def modify_model_config(config, analysis, ui_helpers):
    """
    Interactive menu to modify training settings.

    Parameters
    ----------
    config : dict
        Current configuration
    analysis : dict
        Dataset analysis
    ui_helpers : module
        UI helpers module

    Returns
    -------
    dict
        Updated configuration
    """
    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("MODIFY TRAINING SETTINGS")

        print("\n[1] Learning Rate:", config['learning_rate'])
        print("[2] Batch Size:", config['batch_size'])
        print("[3] Epochs:", config['epochs'])
        print("[4] Validation Split:", f"{config['validation_split']*100:.0f}%")
        print("\n[5] Reset to Recommended")
        print("[0] Done")

        choice = input("\nSelect option: ").strip()

        if choice == '0':
            return config
        elif choice == '1':
            try:
                lr = float(input("Enter learning rate (e.g., 0.001): "))
                if 0 < lr < 1:
                    config['learning_rate'] = lr
            except:
                pass
        elif choice == '2':
            try:
                batch = int(input("Enter batch size: "))
                if batch > 0:
                    config['batch_size'] = batch
            except:
                pass
        elif choice == '3':
            try:
                epochs = int(input("Enter epochs: "))
                if epochs > 0:
                    config['epochs'] = epochs
            except:
                pass
        elif choice == '4':
            try:
                split = float(input("Enter validation split (0.1-0.5): "))
                if 0.1 <= split <= 0.5:
                    config['validation_split'] = split
            except:
                pass
        elif choice == '5':
            config = analysis['recommended_architecture'].copy()

    return config


def load_dataset(dataset_dir, analysis):
    """
    Load all simulation data from dataset.

    Parameters
    ----------
    dataset_dir : Path
        Dataset directory
    analysis : dict
        Dataset analysis

    Returns
    -------
    tuple
        (X_inputs, Y_outputs) as numpy arrays
    """
    print("\nLoading simulation data...")

    # Load model setup to get DOE configuration
    setup_file = dataset_dir / "model_setup.json"
    with open(setup_file, 'r') as f:
        setup_data = json.load(f)

    doe_config = setup_data.get('doe_configuration', {})

    # Get output files
    outputs_dir = dataset_dir / "dataset"
    output_files = sorted(outputs_dir.glob("sim_*.npz"))

    if not output_files:
        raise ValueError("No simulation output files found")

    print(f"  Found {len(output_files)} simulation files")

    # Load first file to determine structure
    sample_data = np.load(output_files[0], allow_pickle=True)
    output_keys = list(sample_data.files)

    # Determine total output size
    total_output_size = 0
    for key in output_keys:
        arr = sample_data[key]

        # Handle different data formats
        if arr.dtype == object:
            if arr.ndim == 0:
                # 0-d object array - extract the object
                obj = arr.item()
                if isinstance(obj, dict):
                    # Dict contains arrays - count total values in all arrays
                    for v in obj.values():
                        if isinstance(v, np.ndarray):
                            total_output_size += v.size
                        elif isinstance(v, (list, tuple)):
                            total_output_size += len(v)
                        else:
                            total_output_size += 1
                elif isinstance(obj, (list, tuple)):
                    total_output_size += len(obj)
                else:
                    total_output_size += 1
            else:
                total_output_size += arr.size
        else:
            total_output_size += arr.size

    print(f"  Output fields: {len(output_keys)}")
    print(f"  Total output size: {total_output_size:,} values")

    # Prepare arrays
    n_samples = len(output_files)
    X_inputs = []
    Y_outputs = np.zeros((n_samples, total_output_size))

    # Load each simulation
    for i, output_file in enumerate(output_files):
        # Load output data
        sim_data = np.load(output_file, allow_pickle=True)

        # Concatenate all output fields into single vector
        output_vector = []
        for key in output_keys:
            arr = sim_data[key]

            # Handle different data formats
            if arr.dtype == object:
                # If it's an object array (might contain dict or list)
                if arr.ndim == 0:
                    # 0-d object array - extract the object
                    obj = arr.item()
                    if isinstance(obj, dict):
                        # Extract arrays from dict values and concatenate
                        for v in obj.values():
                            if isinstance(v, np.ndarray):
                                output_vector.append(v.flatten())
                            elif isinstance(v, (list, tuple)):
                                output_vector.append(np.array(v, dtype=float).flatten())
                            else:
                                output_vector.append(np.array([float(v)]))
                        continue  # Skip the append at the end since we already added
                    elif isinstance(obj, (list, tuple)):
                        arr = np.array(obj, dtype=float).flatten()
                    else:
                        # Try to convert to float array
                        arr = np.array(obj, dtype=float).flatten()
                else:
                    # Multi-dimensional object array - flatten and convert
                    arr = np.array([float(x) if not isinstance(x, dict) else list(x.values())[0]
                                   for x in arr.flatten()], dtype=float)
            else:
                # Regular numeric array - just flatten
                arr = arr.flatten()

            output_vector.append(arr)

        Y_outputs[i, :] = np.concatenate(output_vector)

        # Extract input parameters from filename or metadata
        # Simulation files are named sim_XXXX.npz where XXXX is simulation ID
        sim_id = int(output_file.stem.split('_')[1])

        # Get input parameters from DOE configuration
        # This requires reconstructing the DOE grid
        input_params = extract_input_params_from_doe(doe_config, sim_id)
        X_inputs.append(input_params)

        if (i + 1) % 10 == 0 or i == n_samples - 1:
            print(f"  Loaded {i+1}/{n_samples} simulations", end='\r')

    print()  # New line after loading
    X_inputs = np.array(X_inputs)

    print(f"\n[OK] Dataset loaded successfully")
    print(f"  Input shape: {X_inputs.shape}")
    print(f"  Output shape: {Y_outputs.shape}")

    return X_inputs, Y_outputs


def extract_input_params_from_doe(doe_config, sim_id):
    """
    Extract input parameters for a given simulation ID from DOE configuration.

    Parameters
    ----------
    doe_config : dict
        DOE configuration from model_setup.json
    sim_id : int
        Simulation ID (0-based index into DOE grid)

    Returns
    -------
    list
        List of input parameter values
    """
    # Flatten DOE configuration into ordered list of parameters
    all_params = []
    param_ranges = []

    for bc_name, params in sorted(doe_config.items()):
        for param_name, values in sorted(params.items()):
            if values:  # Non-empty list
                all_params.append((bc_name, param_name))
                param_ranges.append(values)

    # DOE grid is full factorial - convert linear index to multi-index
    if not param_ranges:
        return []

    # Calculate multi-index from linear sim_id
    multi_index = []
    remaining = sim_id

    for param_values in reversed(param_ranges):
        n_values = len(param_values)
        multi_index.append(remaining % n_values)
        remaining //= n_values

    multi_index = list(reversed(multi_index))

    # Extract actual values
    input_values = []
    for idx, param_values in zip(multi_index, param_ranges):
        input_values.append(param_values[idx])

    return input_values

def _create_scalar_plots(Y_test, Y_pred, output_metadata, save_dir):
    """Create scatter plots for scalar outputs."""
    n_samples, n_outputs = Y_test.shape

    # Create subplots
    n_cols = min(4, n_outputs)
    n_rows = (n_outputs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_outputs == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle('Scalar Output Predictions vs. Actual', fontsize=14, fontweight='bold')

    for idx in range(n_outputs):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        y_true = Y_test[:, idx]
        y_pred = Y_pred[:, idx]

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)

        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

        # Calculate R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Get output name if available
        output_name = f"Output {idx+1}"
        if output_metadata and 'names' in output_metadata:
            output_name = output_metadata['names'][idx]

        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{output_name} (R²={r2:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_outputs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_dir:
        plt.savefig(Path(save_dir) / 'scalar_predictions.png', dpi=150)

    plt.show(block=False)
    print("[OK] Scalar output plots displayed")

def _create_2d_field_plots(Y_test, Y_pred, output_metadata, save_dir):
    """Create visualizations for 2D field data."""
    n_samples, n_outputs = Y_test.shape

    # Create figure with multiple views
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('2D Field Data: Predictions vs. Actual', fontsize=14, fontweight='bold')

    # 1. Overall scatter plot (sample of all field points)
    ax1 = fig.add_subplot(gs[0, :])
    sample_size = min(10000, Y_test.size)
    indices = np.random.choice(Y_test.size, sample_size, replace=False)
    y_true_flat = Y_test.flatten()[indices]
    y_pred_flat = Y_pred.flatten()[indices]

    ax1.scatter(y_true_flat, y_pred_flat, alpha=0.3, s=10, c='steelblue')

    # Perfect prediction line
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

    # Overall R²
    ss_res = np.sum((Y_test - Y_pred) ** 2)
    ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)
    r2_overall = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    ax1.set_xlabel('Actual Field Values')
    ax1.set_ylabel('Predicted Field Values')
    ax1.set_title(f'All Field Points (R²={r2_overall:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Per-sample RMSE
    ax2 = fig.add_subplot(gs[1, :])
    sample_errors = np.sqrt(np.mean((Y_test - Y_pred) ** 2, axis=1))

    ax2.bar(range(n_samples), sample_errors, color='steelblue', alpha=0.7, edgecolor='k')
    ax2.axhline(np.mean(sample_errors), color='r', linestyle='--',
                linewidth=2, label=f'Mean RMSE={np.mean(sample_errors):.4e}')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Prediction Error per Sample')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Sample field comparisons (show 3 samples if available)
    n_show = min(3, n_samples)
    for i in range(n_show):
        ax = fig.add_subplot(gs[2, i])

        sample_idx = i * (n_samples // n_show) if n_samples > 3 else i
        error = np.abs(Y_test[sample_idx] - Y_pred[sample_idx])

        ax.plot(Y_test[sample_idx], 'b-', label='Actual', alpha=0.7, linewidth=2)
        ax.plot(Y_pred[sample_idx], 'r--', label='Predicted', alpha=0.7, linewidth=2)
        ax.fill_between(range(n_outputs), Y_test[sample_idx] - error,
                        Y_test[sample_idx] + error, alpha=0.2, color='red')

        ax.set_xlabel('Field Point Index')
        ax.set_ylabel('Field Value')
        ax.set_title(f'Sample {sample_idx+1} (RMSE={sample_errors[sample_idx]:.4e})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        plt.savefig(Path(save_dir) / '2d_field_predictions.png', dpi=150)

    plt.show(block=False)
    print("[OK] 2D field data plots displayed")

def _create_3d_field_plots(Y_test, Y_pred, output_metadata, save_dir):
    """Create visualizations for 3D field data."""
    n_samples, n_outputs = Y_test.shape

    # Create figure with multiple views
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    fig.suptitle('3D Volume Data: Predictions vs. Actual', fontsize=14, fontweight='bold')

    # 1. Overall scatter plot (sample of all field points)
    ax1 = fig.add_subplot(gs[0, 0])
    sample_size = min(15000, Y_test.size)
    indices = np.random.choice(Y_test.size, sample_size, replace=False)
    y_true_flat = Y_test.flatten()[indices]
    y_pred_flat = Y_pred.flatten()[indices]

    ax1.scatter(y_true_flat, y_pred_flat, alpha=0.2, s=5, c='steelblue')

    # Perfect prediction line
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

    # Overall R²
    ss_res = np.sum((Y_test - Y_pred) ** 2)
    ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)
    r2_overall = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    ax1.set_xlabel('Actual Field Values')
    ax1.set_ylabel('Predicted Field Values')
    ax1.set_title(f'All Volume Points (R²={r2_overall:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Per-sample RMSE
    ax2 = fig.add_subplot(gs[0, 1])
    sample_errors = np.sqrt(np.mean((Y_test - Y_pred) ** 2, axis=1))

    ax2.bar(range(n_samples), sample_errors, color='steelblue', alpha=0.7, edgecolor='k')
    ax2.axhline(np.mean(sample_errors), color='r', linestyle='--',
                linewidth=2, label=f'Mean RMSE={np.mean(sample_errors):.4e}')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Prediction Error per Sample')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Error distribution histogram
    ax3 = fig.add_subplot(gs[1, 0])
    all_errors = np.abs(Y_test - Y_pred).flatten()

    ax3.hist(all_errors, bins=50, color='steelblue', alpha=0.7, edgecolor='k')
    ax3.axvline(np.median(all_errors), color='r', linestyle='--',
                linewidth=2, label=f'Median={np.median(all_errors):.4e}')
    ax3.set_xlabel('Absolute Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Field statistics comparison
    ax4 = fig.add_subplot(gs[1, 1])

    stats_names = ['Mean', 'Std', 'Min', 'Max']
    actual_stats = [
        np.mean(Y_test, axis=1),
        np.std(Y_test, axis=1),
        np.min(Y_test, axis=1),
        np.max(Y_test, axis=1)
    ]
    pred_stats = [
        np.mean(Y_pred, axis=1),
        np.std(Y_pred, axis=1),
        np.min(Y_pred, axis=1),
        np.max(Y_pred, axis=1)
    ]

    x = np.arange(len(stats_names))
    width = 0.35

    for i in range(n_samples):
        actual_vals = [stat[i] for stat in actual_stats]
        pred_vals = [stat[i] for stat in pred_stats]

        ax4.bar(x - width/2 + i*0.1, actual_vals, width/n_samples,
                label=f'Actual S{i+1}' if i == 0 else '', alpha=0.7, color='blue')
        ax4.bar(x + width/2 + i*0.1, pred_vals, width/n_samples,
                label=f'Predicted S{i+1}' if i == 0 else '', alpha=0.7, color='red')

    ax4.set_xlabel('Statistic')
    ax4.set_ylabel('Value')
    ax4.set_title('Field Statistics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(stats_names)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_dir:
        plt.savefig(Path(save_dir) / '3d_volume_predictions.png', dpi=150)

    plt.show(block=False)
    print("[OK] 3D volume data plots displayed")

def train_model(dataset_dir, config, analysis, ui_helpers):
    """
    Train the Direct NN model.

    Parameters
    ----------
    dataset_dir : Path
        Dataset directory
    config : dict
        Model configuration
    analysis : dict
        Dataset analysis
    ui_helpers : module
        UI helpers module
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("TRAINING MODEL")

    print(f"\nDataset: {dataset_dir.name}")
    print(f"Architecture: DIRECT NN")
    print(f"Samples: {analysis['num_samples']}")
    print(f"Epochs: {config['epochs']}")

    try:
        # Load dataset
        X_inputs, Y_outputs = load_dataset(dataset_dir, analysis)

        # Create Direct NN configuration
        model_config = {
            'input_dim': X_inputs.shape[1],
            'output_dim': Y_outputs.shape[1],
            'hidden_dims': None,  # Auto-calculated
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'epochs': config['epochs'],
            'validation_split': config['validation_split'],
            'lr_patience': config.get('lr_patience', 10)
        }

        # Create model
        print(f"\nCreating Direct NN model...")
        model = DirectNNModel(model_config)

        # Train model
        print(f"\nStarting training...")
        print(f"  Training will open a real-time plot window.")
        print(f"  Press Ctrl+C to stop training early.\n")

        input("\nPress Enter to start training...")

        # Use dynamically calculated patience values
        early_stop_patience = config.get('early_stop_patience', 30)

        print(f"\n  Training configuration:")
        print(f"    Early stopping patience: {early_stop_patience} epochs")
        print(f"    LR scheduler patience: {config.get('lr_patience', 10)} epochs\n")

        model.train(
            X_inputs,
            Y_outputs,
            validation_split=config['validation_split'],
            early_stopping_patience=early_stop_patience
        )

        # Evaluate on full dataset (for demonstration)
        print("\nEvaluating model on training data...")
        results = evaluate_model(model, X_inputs, Y_outputs)

        # Generate adaptive visualizations
        print("\nGenerating adaptive prediction visualizations...")
        create_adaptive_visualizations(
            Y_outputs,
            results['predictions']
        )

        # Ask to save model
        print(f"\n{'='*70}")
        save_choice = input("\nSave trained model? (y/n): ").strip().lower()

        if save_choice == 'y':
            model_name = input("Enter model name (default: timestamp): ").strip()
            if not model_name:
                model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Create model save directory inside the case folder
            models_dir = dataset_dir / "models"
            models_dir.mkdir(exist_ok=True)

            model_dir = models_dir / model_name
            model_dir.mkdir(exist_ok=True)

            # Save model
            model.save(model_dir)

            # Save evaluation results
            eval_results = {
                'r2_score': results['r2_score'],
                'mae': results['mae'],
                'rmse': results['rmse'],
                'max_error': results['max_error'],
                'per_output_r2': results['per_output_r2'],
                'n_test_samples': results['n_test_samples'],
                'dataset_name': dataset_dir.name,
                'training_date': datetime.now().isoformat()
            }

            with open(model_dir / 'evaluation_results.json', 'w') as f:
                json.dump(eval_results, f, indent=2)

            print(f"\n[OK] Model saved to: {model_dir}")
            print(f"  Model files: direct_nn.pth")
            print(f"  Metadata: model_info.json, evaluation_results.json")
        else:
            print("\n[X] Model not saved")

    except KeyboardInterrupt:
        print("\n\n[STOP] Training cancelled by user")
    except Exception as e:
        print(f"\n[X] Error during training: {e}")
        import traceback
        traceback.print_exc()

    ui_helpers.pause()

def load_existing_model(dataset_dir, ui_helpers):
    """Load and test a previously trained model."""
    ui_helpers.clear_screen()
    ui_helpers.print_header("LOAD EXISTING MODEL")

    # Look in case's models directory
    models_dir = dataset_dir / "models"

    if not models_dir.exists():
        print("\n[X] No models directory found for this case.")
        print(f"  Expected location: {models_dir}")
        ui_helpers.pause()
        return

    # Find model directories (contain model_info.json)
    model_dirs = [d for d in models_dir.iterdir()
                  if d.is_dir() and (d / "model_info.json").exists()]

    if not model_dirs:
        print("\n[X] No trained models found in this case.")
        print(f"  Searched in: {models_dir}")
        ui_helpers.pause()
        return

    # Display available models
    print("\nAvailable Models:")
    print("="*70)

    for i, model_dir in enumerate(model_dirs, 1):
        # Load model info
        with open(model_dir / "model_info.json", 'r') as f:
            model_info = json.load(f)

        config = model_info['config']
        timestamp = model_info.get('timestamp', 'Unknown')

        print(f"\n[{i}] {model_dir.name}")
        print(f"    Architecture: Direct NN")
        print(f"    Created: {timestamp}")
        print(f"    Input dim: {config['input_dim']}")
        print(f"    Output dim: {config['output_dim']}")

        # Show evaluation results if available
        eval_file = model_dir / "evaluation_results.json"
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                eval_results = json.load(f)
            print(f"    R² Score: {eval_results['r2_score']:.6f}")
            print(f"    RMSE: {eval_results['rmse']:.6e}")

    print("\n" + "="*70)
    print("[0] Back")
    print("="*70)

    choice = ui_helpers.get_choice(len(model_dirs))

    if choice == 0:
        return

    # Load selected model
    selected_model_dir = model_dirs[choice - 1]

    try:
        print(f"\nLoading model from: {selected_model_dir.name}")

        # Load Direct NN model
        model = DirectNNModel.load(selected_model_dir)

        print("\n[OK] Model loaded successfully!")

        # Offer to test on dataset
        print("\n" + "="*70)
        print("Model Testing Options:")
        print("="*70)
        print("[1] Test on current dataset")
        print("[2] Make single prediction")
        print("[0] Back")
        print("="*70)

        test_choice = ui_helpers.get_choice(2)

        if test_choice == 1:
            # Test on dataset
            print("\nLoading dataset...")
            analysis = analyze_dataset_for_training(dataset_dir)
            X_inputs, Y_outputs = load_dataset(dataset_dir, analysis)

            print("\nEvaluating model...")
            results = evaluate_model(model, X_inputs, Y_outputs)

            # Generate adaptive visualizations
            create_adaptive_visualizations(
                Y_outputs,
                results['predictions']
            )

        elif test_choice == 2:
            # Make single prediction
            print("\nEnter input parameters:")
            with open(selected_model_dir / "model_info.json", 'r') as f:
                model_info = json.load(f)

            input_dim = model_info['config']['input_dim']
            input_values = []

            for i in range(input_dim):
                while True:
                    try:
                        val = float(input(f"  Parameter {i+1}: "))
                        input_values.append(val)
                        break
                    except ValueError:
                        print("    Invalid number, try again.")

            # Make prediction
            X_test = np.array([input_values])
            Y_pred = model.predict(X_test)

            print(f"\nPrediction shape: {Y_pred.shape}")
            print(f"Output values (first 10): {Y_pred[0, :10]}")
            print(f"Output range: [{Y_pred.min():.6e}, {Y_pred.max():.6e}]")

    except Exception as e:
        print(f"\n[X] Error loading model: {e}")
        import traceback
        traceback.print_exc()

    ui_helpers.pause()

def evaluate_model(model, X_test, Y_test):
    """
    Evaluate trained model on test data.

    Parameters
    ----------
    model : DirectNNModel
        Trained model
    X_test : ndarray
        Test inputs (n_samples, input_dim)
    Y_test : ndarray
        Test outputs (n_samples, output_dim)

    Returns
    -------
    dict
        Dictionary containing evaluation metrics:
        - r2_score: R-squared score
        - mae: Mean absolute error
        - rmse: Root mean squared error
        - max_error: Maximum absolute error
        - predictions: Model predictions
    """
    print(f"\n{'='*70}")
    print("EVALUATING MODEL")
    print(f"{'='*70}")

    # Get predictions
    Y_pred = model.predict(X_test)

    # Calculate metrics
    # R² score
    ss_res = np.sum((Y_test - Y_pred) ** 2)
    ss_tot = np.sum((Y_test - np.mean(Y_test, axis=0)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # MAE
    mae = np.mean(np.abs(Y_test - Y_pred))

    # RMSE
    rmse = np.sqrt(np.mean((Y_test - Y_pred) ** 2))

    # Max error
    max_error = np.max(np.abs(Y_test - Y_pred))

    # Per-output metrics (if multiple outputs)
    if Y_test.shape[1] > 1:
        per_output_r2 = []
        for i in range(Y_test.shape[1]):
            ss_res_i = np.sum((Y_test[:, i] - Y_pred[:, i]) ** 2)
            ss_tot_i = np.sum((Y_test[:, i] - np.mean(Y_test[:, i])) ** 2)
            r2_i = 1 - (ss_res_i / ss_tot_i) if ss_tot_i > 0 else 0
            per_output_r2.append(r2_i)
    else:
        per_output_r2 = [r2]

    results = {
        'r2_score': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'max_error': float(max_error),
        'per_output_r2': per_output_r2,
        'predictions': Y_pred,
        'n_test_samples': len(X_test)
    }

    # Print results
    print(f"\nTest Set Size: {len(X_test)} samples")
    print(f"\nOverall Metrics:")
    print(f"  R² Score:    {r2:.6f}")
    print(f"  MAE:         {mae:.6e}")
    print(f"  RMSE:        {rmse:.6e}")
    print(f"  Max Error:   {max_error:.6e}")

    if len(per_output_r2) > 1:
        print(f"\nPer-Output R² Scores:")
        for i, r2_i in enumerate(per_output_r2):
            print(f"  Output {i+1}: {r2_i:.6f}")

    print(f"{'='*70}\n")

    return results

def create_adaptive_visualizations(Y_test, Y_pred, output_metadata=None, save_dir=None):
    """
    Create adaptive visualizations based on output data type.
    Automatically detects whether outputs are:
    - Scalars (report definitions)
    - 2D field data (surfaces)
    - 3D field data (volumes)
    And creates appropriate visualizations.

    Parameters
    ----------
    Y_test : ndarray
        Actual test outputs (n_samples, n_outputs)
    Y_pred : ndarray
        Predicted outputs (n_samples, n_outputs)
    output_metadata : dict, optional
        Metadata about outputs including 'type', 'name', 'dimensions', etc.
    save_dir : Path, optional
        Directory to save plots
    """
    n_samples, n_outputs = Y_test.shape

    print(f"\n{'='*70}")
    print("CREATING ADAPTIVE VISUALIZATIONS")
    print(f"{'='*70}")
    print(f"Output shape: {Y_test.shape}")
    print(f"  Samples: {n_samples}")
    print(f"  Output dimension: {n_outputs}")

    # Determine output type
    if n_outputs <= 10:
        # Likely scalar outputs (report definitions)
        output_type = 'scalar'
    elif n_outputs < 1000:
        # Likely 2D surface data
        output_type = '2d_field'
    else:
        # Likely 3D volume data
        output_type = '3d_field'

    print(f"  Detected type: {output_type}")
    print(f"{'='*70}\n")

    if output_type == 'scalar':
        _create_scalar_plots(Y_test, Y_pred, output_metadata, save_dir)
    elif output_type == '2d_field':
        _create_2d_field_plots(Y_test, Y_pred, output_metadata, save_dir)
    else:  # 3d_field
        _create_3d_field_plots(Y_test, Y_pred, output_metadata, save_dir)
