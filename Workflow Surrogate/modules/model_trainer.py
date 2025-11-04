"""
Model Trainer Module
====================
Builds and trains neural network models with automatic architecture selection.
Supports both Bottleneck NN (autoencoder) and Direct NN architectures.
"""

import json
from pathlib import Path
import numpy as np
from datetime import datetime
from .autoencoder_model import AutoencoderModel, evaluate_model, create_adaptive_visualizations
from .direct_nn_model import DirectNNModel


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
        outputs_dir = dataset_dir / "outputs"
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

                    # Recommend latent size with adaptive formula
                    # For small datasets: use more latent dimensions to avoid extreme compression
                    # For large datasets: use sample-based approach

                    sample_based = max(8, analysis['num_samples'] // 2)  # At least 8, up to half of samples

                    # Also consider output size - don't compress more than 100:1
                    max_compression_ratio = 100
                    size_based = max(10, analysis['output_size'] // max_compression_ratio)

                    # Use the larger of the two (more conservative)
                    analysis['recommended_latent_size'] = min(
                        max(sample_based, size_based),
                        100  # Cap at 100
                    )
                except Exception as e:
                    analysis['errors'].append(f"Error loading output file: {e}")
            else:
                analysis['errors'].append("No simulation output files found")
        else:
            analysis['errors'].append("Outputs directory not found")

        # Recommend neural network architecture
        if analysis['dataset_found']:
            input_dim = analysis['input_dim']
            latent_size = analysis['recommended_latent_size']

            # Encoder: input -> latent representation
            encoder_h1 = max(32, int(2 * input_dim))
            encoder_h2 = max(16, encoder_h1 // 2)

            # Decoder: latent representation -> output
            decoder_h1 = max(64, int(2 * latent_size))
            decoder_h2 = max(32, decoder_h1 // 2)

            # Adaptive hyperparameters based on dynamic curves
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

            # Batch size: logarithmic growth with sample count
            # Ensure at least 3-5 batches per epoch for good gradient estimates
            batch_size = max(2, min(32, int(n_samples ** 0.7 / 2)))

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
                'encoder': [input_dim, encoder_h1, encoder_h2, latent_size],
                'decoder': [latent_size, decoder_h1, decoder_h2, analysis['output_size']],
                'activation': 'relu',
                'output_activation': 'linear',
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


def select_architecture_type(ui_helpers):
    """
    Let user select between Bottleneck NN and Direct NN.

    Parameters
    ----------
    ui_helpers : module
        UI helpers module

    Returns
    -------
    str or None
        'bottleneck' or 'direct', or None if cancelled
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("SELECT ARCHITECTURE TYPE")

    print("\nChoose neural network architecture:\n")
    print("  [1] Bottleneck NN (Autoencoder)")
    print("      - Compresses data through latent space")
    print("      - Good for dimensionality reduction")
    print("      - Learns compressed feature representation")
    print("      - Requires sufficient data for compression ratio")
    print()
    print("  [2] Direct NN (Feedforward)")
    print("      - Direct mapping from inputs to outputs")
    print("      - No compression bottleneck")
    print("      - Generally higher accuracy")
    print("      - Larger model size")
    print()
    print("  [0] Back")
    print()

    choice = ui_helpers.get_choice(2)

    if choice == 0:
        return None
    elif choice == 1:
        return 'bottleneck'
    elif choice == 2:
        return 'direct'


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

    # Select architecture type
    architecture_type = select_architecture_type(ui_helpers)
    if architecture_type is None:
        return

    # Default configuration
    config = analysis['recommended_architecture'].copy()
    config['latent_dim'] = analysis['recommended_latent_size']
    config['architecture_type'] = architecture_type

    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("MODEL TRAINING")

        print(f"\nDataset: {dataset_dir.name}")
        print(f"Samples: {analysis['num_samples']}")
        print(f"Input Dimensions: {analysis['input_dim']}")
        print(f"Output Size: {analysis['output_size']:,} values")

        print("\n" + "="*70)
        print(f"ARCHITECTURE: {architecture_type.upper()}")
        print("="*70)

        if architecture_type == 'bottleneck':
            print(f"  Latent Size: {config['latent_dim']}")
            print(f"  Encoder: {' -> '.join(map(str, config['encoder']))}")
            print(f"  Decoder: {' -> '.join(map(str, config['decoder']))}")
            compression_ratio = analysis['output_size'] / config['latent_dim']
            print(f"  Compression Ratio: {compression_ratio:.1f}:1")
        else:
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
            print(f"  No bottleneck compression")

        print(f"\n  Training Settings:")
        print(f"    Learning Rate: {config['learning_rate']}")
        print(f"    Batch Size: {config['batch_size']}")
        print(f"    Epochs: {config['epochs']}")
        print(f"    Validation Split: {config['validation_split']*100:.0f}%")

        print(f"\n{'='*70}")
        if architecture_type == 'bottleneck':
            print("  [1] Modify Model Configuration")
        else:
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
    Interactive menu to modify model configuration.

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
        ui_helpers.print_header("MODIFY MODEL CONFIGURATION")

        print("\n[1] Latent Size:", config['latent_dim'])
        print(f"    (Recommended: {analysis['recommended_latent_size']})")
        print("\n[2] Learning Rate:", config['learning_rate'])
        print("[3] Batch Size:", config['batch_size'])
        print("[4] Epochs:", config['epochs'])
        print("[5] Validation Split:", f"{config['validation_split']*100:.0f}%")
        print("\n[6] Reset to Recommended")
        print("[0] Done")

        choice = input("\nSelect option: ").strip()

        if choice == '0':
            return config
        elif choice == '1':
            try:
                latent_size = int(input(f"Enter latent size (1-100): "))
                if 1 <= latent_size <= 100:
                    config['latent_dim'] = latent_size
                    # Update encoder/decoder
                    config['encoder'][-1] = latent_size
                    config['decoder'][0] = latent_size
            except:
                pass
        elif choice == '2':
            try:
                lr = float(input("Enter learning rate (e.g., 0.001): "))
                if 0 < lr < 1:
                    config['learning_rate'] = lr
            except:
                pass
        elif choice == '3':
            try:
                batch = int(input("Enter batch size: "))
                if batch > 0:
                    config['batch_size'] = batch
            except:
                pass
        elif choice == '4':
            try:
                epochs = int(input("Enter epochs: "))
                if epochs > 0:
                    config['epochs'] = epochs
            except:
                pass
        elif choice == '5':
            try:
                split = float(input("Enter validation split (0.1-0.5): "))
                if 0.1 <= split <= 0.5:
                    config['validation_split'] = split
            except:
                pass
        elif choice == '6':
            config = analysis['recommended_architecture'].copy()
            config['latent_dim'] = analysis['recommended_latent_size']

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
    outputs_dir = dataset_dir / "outputs"
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


def train_model(dataset_dir, config, analysis, ui_helpers):
    """
    Train the neural network model (Bottleneck NN or Direct NN).

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

    architecture_type = config.get('architecture_type', 'bottleneck')

    print(f"\nDataset: {dataset_dir.name}")
    print(f"Architecture: {architecture_type.upper()}")
    print(f"Samples: {analysis['num_samples']}")
    if architecture_type == 'bottleneck':
        print(f"Latent Size: {config['latent_dim']}")
    print(f"Epochs: {config['epochs']}")

    try:
        # Load dataset
        X_inputs, Y_outputs = load_dataset(dataset_dir, analysis)

        if architecture_type == 'bottleneck':
            # Create Bottleneck NN (autoencoder) configuration
            model_config = {
                'input_dim': X_inputs.shape[1],
                'encoder_hidden_layers': config['encoder'][1:-1],  # Middle layers
                'latent_dim': config['latent_dim'],
                'decoder_hidden_layers': config['decoder'][1:-1],  # Middle layers
                'output_dim': Y_outputs.shape[1],
                'activation': config['activation'],
                'learning_rate': config['learning_rate'],
                'batch_size': config['batch_size'],
                'epochs': config['epochs'],
                'validation_split': config['validation_split'],
                'lr_patience': config.get('lr_patience', 10)
            }

            # Create model
            print(f"\nCreating Bottleneck NN model...")
            model = AutoencoderModel(model_config)

        else:
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

            # Create model save directory
            trained_models_dir = dataset_dir.parent.parent / "trained_models"
            trained_models_dir.mkdir(exist_ok=True)

            model_dir = trained_models_dir / model_name
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
            if architecture_type == 'bottleneck':
                print(f"  Model files: encoder.pth, decoder.pth")
            else:
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

    # Look in project's trained_models directory
    trained_models_dir = dataset_dir.parent.parent / "trained_models"

    if not trained_models_dir.exists():
        print("\n[X] No trained models directory found.")
        print(f"  Expected location: {trained_models_dir}")
        ui_helpers.pause()
        return

    # Find model directories (contain model_info.json)
    model_dirs = [d for d in trained_models_dir.iterdir()
                  if d.is_dir() and (d / "model_info.json").exists()]

    if not model_dirs:
        print("\n[X] No trained models found.")
        print(f"  Searched in: {trained_models_dir}")
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

        # Detect architecture type
        is_direct_nn = (model_dir / "direct_nn.pth").exists()
        arch_type = "Direct NN" if is_direct_nn else "Bottleneck NN"

        print(f"\n[{i}] {model_dir.name}")
        print(f"    Architecture: {arch_type}")
        print(f"    Created: {timestamp}")
        print(f"    Input dim: {config['input_dim']}")
        if not is_direct_nn:
            print(f"    Latent size: {config.get('latent_dim', config.get('pod_modes', 'N/A'))}")
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

        # Detect architecture type
        is_direct_nn = (selected_model_dir / "direct_nn.pth").exists()

        if is_direct_nn:
            model = DirectNNModel.load(selected_model_dir)
        else:
            model = AutoencoderModel.load(selected_model_dir)

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
