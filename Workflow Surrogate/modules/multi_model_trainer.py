"""
Multi-Model Trainer Module
===========================
Trains specialized neural network models (1D, 2D, 3D) for each output parameter.
Automatically detects output dimensionality and uses appropriate model type.
"""

import json
from pathlib import Path
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF INFO and WARNING messages
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', message='.*tf.function retracing.*')

from .scalar_nn_model import ScalarNNModel
from .field_nn_model import FieldNNModel
from .volume_nn_model import VolumeNNModel


def detect_output_type(data_shape):
    """
    Detect if output is 1D (scalar), 2D (field), or 3D (volume).

    Parameters
    ----------
    data_shape : tuple
        Shape of the output data for a single sample

    Returns
    -------
    str
        Output type: '1D', '2D', or '3D'
    int
        Suggested number of POD modes (if applicable)
    """
    n_points = data_shape[0] if len(data_shape) > 0 else 1

    if n_points <= 100:
        # Small number of points - scalar/point data
        return '1D', 0
    elif n_points <= 10000:
        # Medium size - likely 2D cut plane
        return '2D', min(10, n_points // 100)
    else:
        # Large size - likely 3D volume
        return '3D', min(20, n_points // 1000)


def load_training_data(dataset_dir):
    """
    Load all simulation data from case directory.

    Parameters
    ----------
    dataset_dir : Path
        Case directory

    Returns
    -------
    dict
        Dictionary with:
        - 'parameters': Input parameters array (n_samples, n_params)
        - 'outputs': Dict of output arrays by location name
        - 'output_info': Dict with metadata for each output
    """
    print(f"\nLoading training data from: {dataset_dir.name}")

    # Load model setup to get DOE configuration
    setup_file = dataset_dir / "model_setup.json"
    with open(setup_file, 'r') as f:
        setup_data = json.load(f)

    # Load output parameters configuration
    output_params_file = dataset_dir / "output_parameters.json"
    with open(output_params_file, 'r') as f:
        output_params = json.load(f)

    # Get simulation output files
    dataset_output_dir = dataset_dir / "dataset"
    output_files = sorted(dataset_output_dir.glob("sim_*.npz"))

    if not output_files:
        raise ValueError(f"No simulation files found in {dataset_output_dir}")

    print(f"  Found {len(output_files)} simulation files")

    # Reconstruct input parameters from DOE
    doe_config = setup_data.get('doe_configuration', {})
    param_names = []
    param_values = []

    for bc_name, params in doe_config.items():
        for param_name, values in params.items():
            param_names.append(f"{bc_name}.{param_name}")
            param_values.append(values)

    # Generate all combinations
    import itertools
    param_combinations = list(itertools.product(*param_values))
    X_params = np.array(param_combinations[:len(output_files)])

    print(f"  Input parameters: {X_params.shape}")

    # Load outputs
    output_data = {}
    output_info = {}

    # First pass: determine structure
    sample_file = np.load(output_files[0], allow_pickle=True)

    for output_location, field_list in output_params.items():
        for field_name in field_list:
            # Create unique key matching the NPZ file format: "location|field"
            npz_key = f"{output_location}|{field_name}"
            model_key = f"{output_location}_{field_name}"

            # Check if this field exists in the data
            if npz_key in sample_file.files:
                # Load data directly from NPZ (already flattened)
                sample_values = sample_file[npz_key]

                # Detect output type
                output_type, n_modes = detect_output_type(sample_values.shape)

                output_info[model_key] = {
                    'location': output_location,
                    'field': field_name,
                    'npz_key': npz_key,  # Store NPZ key for loading
                    'type': output_type,
                    'n_modes': n_modes,
                    'n_points': len(sample_values)
                }

                # Removed redundant print - output info shown in training summary
                pass
            else:
                print(f"  ⚠ Warning: Key '{npz_key}' not found in simulation data")

    # Second pass: load all data
    for model_key, info in output_info.items():
        output_arrays = []
        npz_key = info['npz_key']

        for output_file in output_files:
            data = np.load(output_file, allow_pickle=True)

            # Load data directly using NPZ key
            values = data[npz_key]
            output_arrays.append(values)

        output_data[model_key] = np.array(output_arrays)

    return {
        'parameters': X_params,
        'outputs': output_data,
        'output_info': output_info,
        'param_names': param_names
    }


def train_all_models(dataset_dir, ui_helpers, test_size=0.2, epochs=500):
    """
    Train models for all outputs in a case.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    ui_helpers : module
        UI helpers module
    test_size : float
        Fraction of data for testing
    epochs : int
        Training epochs
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("TRAIN SURROGATE MODELS")

    try:
        # Load data
        data = load_training_data(dataset_dir)
        X_params = data['parameters']
        outputs = data['outputs']
        output_info = data['output_info']

        print(f"\n{'='*70}")
        print(f"Training Configuration")
        print(f"{'='*70}")
        print(f"  Input samples: {len(X_params)}")
        print(f"  Input parameters: {X_params.shape[1]}")
        print(f"  Outputs to train: {len(outputs)}")
        print(f"  Test split: {test_size*100:.0f}%")
        print(f"  Epochs: {epochs}")

        # Split data
        train_idx, test_idx = train_test_split(
            np.arange(len(X_params)),
            test_size=test_size,
            random_state=42
        )

        print(f"  Train samples: {len(train_idx)}")
        print(f"  Test samples: {len(test_idx)}")

        # Create models directory
        models_dir = dataset_dir / "models"
        models_dir.mkdir(exist_ok=True)

        # Track model counts for naming
        model_counts = {'1D': {}, '2D': {}, '3D': {}}
        trained_models = []

        # Train each output
        for output_key, output_data in outputs.items():
            info = output_info[output_key]
            output_type = info['type']
            field_name = info['field']

            # Determine model name with index
            if field_name not in model_counts[output_type]:
                model_counts[output_type][field_name] = 0
            model_counts[output_type][field_name] += 1
            index = model_counts[output_type][field_name]

            model_name = f"{output_type}_{field_name}_{index}"

            print(f"\n{'='*70}")
            print(f"Training model: {model_name}")
            print(f"{'='*70}")

            # Create appropriate model
            if output_type == '1D':
                model = ScalarNNModel(field_name=output_key)
            elif output_type == '2D':
                model = FieldNNModel(n_modes=info['n_modes'], field_name=output_key)
            else:  # 3D
                model = VolumeNNModel(n_modes=info['n_modes'], field_name=output_key)

            # Train
            model.fit(
                X_params[train_idx],
                output_data[train_idx],
                validation_split=0.2,
                epochs=epochs,
                verbose=0
            )

            # Evaluate
            print(f"\nEvaluating on train set...")
            train_metrics = model.evaluate(X_params[train_idx], output_data[train_idx], 'train')

            print(f"Evaluating on test set...")
            test_metrics = model.evaluate(X_params[test_idx], output_data[test_idx], 'test')

            print(f"\n  Train R²: {train_metrics['r2']:.4f}")
            print(f"  Test R²:  {test_metrics['r2']:.4f}")
            print(f"  Test MAE: {test_metrics['mae']:.4f}")

            # Save model
            model_path = models_dir / model_name
            model.save(model_path)

            # Save metadata
            metadata = {
                'model_name': model_name,
                'output_key': output_key,
                'output_type': output_type,
                'field_name': field_name,
                'location': info['location'],
                'n_points': info['n_points'],
                'train_metrics': {k: float(v) if not isinstance(v, list) else v
                                  for k, v in train_metrics.items()},
                'test_metrics': {k: float(v) if not isinstance(v, list) else v
                                 for k, v in test_metrics.items()},
                'trained_date': datetime.now().isoformat(),
                'n_train_samples': len(train_idx),
                'n_test_samples': len(test_idx)
            }

            if output_type in ['2D', '3D']:
                metadata['n_modes'] = info['n_modes']
                metadata['variance_explained'] = float(model.variance_explained.sum())

            with open(models_dir / f"{model_name}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            trained_models.append(metadata)

        # Save training summary
        summary = {
            'case_name': dataset_dir.name,
            'trained_date': datetime.now().isoformat(),
            'n_models': len(trained_models),
            'n_train_samples': len(train_idx),
            'n_test_samples': len(test_idx),
            'test_split': test_size,
            'epochs': epochs,
            'models': trained_models
        }

        with open(models_dir / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"\nTrained {len(trained_models)} models:")
        for model_meta in trained_models:
            print(f"  - {model_meta['model_name']:30s} (Test R²: {model_meta['test_metrics']['r2']:.4f})")
        print(f"\nModels saved to: {models_dir}")

    except Exception as e:
        print(f"\n[X] Error during training: {e}")
        import traceback
        traceback.print_exc()

    ui_helpers.pause()


def train_model_menu(dataset_dir, ui_helpers):
    """
    Interactive menu for model training.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    ui_helpers : module
        UI helpers module
    """
    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("MODEL TRAINING")

        print(f"\nCase: {dataset_dir.name}")

        # Check if data exists
        dataset_output_dir = dataset_dir / "dataset"
        if not dataset_output_dir.exists():
            print("\n[X] No simulation data found. Run simulations first.")
            ui_helpers.pause()
            return

        output_files = list(dataset_output_dir.glob("sim_*.npz"))
        if not output_files:
            print("\n[X] No simulation files found. Run simulations first.")
            ui_helpers.pause()
            return

        print(f"Simulation files: {len(output_files)}")

        # Check for existing models
        models_dir = dataset_dir / "models"
        existing_models = []
        if models_dir.exists():
            existing_models = list(models_dir.glob("*_metadata.json"))

        if existing_models:
            print(f"Existing models: {len(existing_models)}")

        print(f"\n{'='*70}")
        print("  [1] Train New Models (All Outputs)")
        print("  [2] View Existing Models")
        print("  [0] Back")
        print("="*70)

        choice = ui_helpers.get_choice(2)

        if choice == 0:
            return
        elif choice == 1:
            # Get training parameters
            print("\nTraining Parameters:")

            try:
                test_size_input = input("  Test split (0-1, default 0.2): ").strip()
                test_size = float(test_size_input) if test_size_input else 0.2
                test_size = max(0.1, min(0.5, test_size))

                epochs_input = input("  Epochs (default 500): ").strip()
                epochs = int(epochs_input) if epochs_input else 500
                epochs = max(10, min(2000, epochs))

            except ValueError:
                print("\n[X] Invalid input. Using defaults.")
                test_size = 0.2
                epochs = 500
                ui_helpers.pause()

            # Confirm
            confirm = input(f"\nTrain models with test_size={test_size}, epochs={epochs}? [y/N]: ").strip().lower()
            if confirm == 'y':
                train_all_models(dataset_dir, ui_helpers, test_size=test_size, epochs=epochs)

        elif choice == 2:
            view_existing_models(dataset_dir, ui_helpers)


def view_existing_models(dataset_dir, ui_helpers):
    """
    View information about existing trained models.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    ui_helpers : module
        UI helpers module
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("EXISTING MODELS")

    models_dir = dataset_dir / "models"

    if not models_dir.exists():
        print("\n[X] No models directory found.")
        ui_helpers.pause()
        return

    # Look for training summary
    summary_file = models_dir / "training_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        print(f"\nCase: {summary['case_name']}")
        print(f"Trained: {summary['trained_date']}")
        print(f"Models: {summary['n_models']}")
        print(f"Training samples: {summary['n_train_samples']}")
        print(f"Test samples: {summary['n_test_samples']}")

        print(f"\n{'='*70}")
        print(f"{'Model Name':<35s} {'Type':<6s} {'Test R²':<10s} {'Test MAE':<10s}")
        print(f"{'='*70}")

        for model_meta in summary['models']:
            model_name = model_meta['model_name']
            model_type = model_meta['output_type']
            test_r2 = model_meta['test_metrics']['r2']
            test_mae = model_meta['test_metrics']['mae']

            print(f"{model_name:<35s} {model_type:<6s} {test_r2:>9.4f} {test_mae:>9.4f}")
    else:
        print("\n[X] No training summary found.")
        print("Looking for individual model metadata files...")

        metadata_files = list(models_dir.glob("*_metadata.json"))
        if metadata_files:
            print(f"\nFound {len(metadata_files)} models:")
            for meta_file in metadata_files:
                print(f"  - {meta_file.stem}")
        else:
            print("\nNo models found.")

    ui_helpers.pause()
