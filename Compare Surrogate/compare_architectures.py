"""
Architecture Comparison Tool
=============================
Compare Bottleneck NN vs Direct NN performance on simulation datasets.
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tkinter import Tk, filedialog
import time

# Add parent directory to path to import modules
parent_dir = Path(__file__).parent.parent / "Workflow Surrogate"
sys.path.insert(0, str(parent_dir))

from modules.project_system import WorkflowProject
from modules.autoencoder_model import AutoencoderModel
from direct_nn_model import DirectNNModel


def clear_screen():
    """Clear the console screen."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def select_project():
    """
    Select a project folder.

    Returns
    -------
    WorkflowProject or None
        Loaded project
    """
    print("\nSelect project folder...")
    Tk().withdraw()

    project_folder = filedialog.askdirectory(
        title="Select Project Folder"
    )

    if not project_folder:
        print("\n[X] No folder selected")
        return None

    project = WorkflowProject(Path(project_folder))

    if project.load():
        print(f"\n[OK] Project loaded: {project.info['project_name']}")
        return project
    else:
        print(f"\n[X] Invalid project folder")
        return None


def select_dataset(project):
    """
    Select a simulation setup from project.

    Parameters
    ----------
    project : WorkflowProject
        Loaded project

    Returns
    -------
    dict or None
        Selected setup info
    """
    if not project.datasets:
        print("\n[X] No simulation setups found in project")
        return None

    print_header("SELECT SETUP")
    print("\nAvailable Simulation Setups:\n")

    for i, dataset in enumerate(project.datasets, 1):
        print(f"  {i}. {dataset['name']}")
        print(f"     Inputs: {dataset['num_inputs']} | "
              f"Outputs: {dataset['num_outputs']} | "
              f"Simulations: {dataset['num_simulations']}/{dataset['total_required']} "
              f"({dataset['completeness']:.1f}%)")

    print("\n  0. Cancel")

    while True:
        try:
            choice = input("\nSelect dataset [0]: ").strip()
            if not choice or choice == '0':
                return None

            choice = int(choice)
            if 1 <= choice <= len(project.datasets):
                return project.datasets[choice - 1]
            else:
                print("[X] Invalid selection")
        except ValueError:
            print("[X] Please enter a number")


def load_dataset_data(dataset):
    """
    Load simulation data from setup.

    Parameters
    ----------
    dataset : dict
        Setup information

    Returns
    -------
    tuple
        (X_data, Y_data, setup_data, output_params)
    """
    dataset_path = dataset['path']

    # Load setup
    with open(dataset['setup_file'], 'r') as f:
        setup_data = json.load(f)

    # Load output parameters
    output_params_file = dataset_path / "output_parameters.json"
    with open(output_params_file, 'r') as f:
        output_params = json.load(f)

    # Get DOE configuration
    doe_config = setup_data.get('doe_configuration', {})

    # Load simulation data
    outputs_dir = dataset_path / "outputs"
    sim_files = sorted(outputs_dir.glob("sim_*.npz"))

    if not sim_files:
        raise ValueError("No simulation files found")

    print(f"\nLoading {len(sim_files)} simulations...")

    # Load first file to determine output structure
    sample_data = np.load(sim_files[0], allow_pickle=True)
    output_keys = list(sample_data.files)

    # Determine total output size
    total_output_size = 0
    for key in output_keys:
        arr = sample_data[key]
        if arr.dtype == object:
            if arr.ndim == 0:
                obj = arr.item()
                if isinstance(obj, dict):
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

    # Prepare arrays
    n_samples = len(sim_files)
    X_inputs = []
    Y_outputs = np.zeros((n_samples, total_output_size))

    # Load each simulation
    for i, sim_file in enumerate(sim_files):
        # Load output data
        sim_data = np.load(sim_file, allow_pickle=True)

        # Concatenate all output fields into single vector
        output_vector = []
        for key in output_keys:
            arr = sim_data[key]

            # Handle different data formats
            if arr.dtype == object:
                if arr.ndim == 0:
                    obj = arr.item()
                    if isinstance(obj, dict):
                        for v in obj.values():
                            if isinstance(v, np.ndarray):
                                output_vector.append(v.flatten())
                            elif isinstance(v, (list, tuple)):
                                output_vector.append(np.array(v, dtype=float).flatten())
                            else:
                                output_vector.append(np.array([float(v)]))
                        continue
                    elif isinstance(obj, (list, tuple)):
                        arr = np.array(obj, dtype=float).flatten()
                    else:
                        arr = np.array(obj, dtype=float).flatten()
                else:
                    arr = np.array([float(x) if not isinstance(x, dict) else list(x.values())[0]
                                   for x in arr.flatten()], dtype=float)
            else:
                arr = arr.flatten()

            output_vector.append(arr)

        Y_outputs[i, :] = np.concatenate(output_vector)

        # Extract input parameters from DOE configuration
        sim_id = int(sim_file.stem.split('_')[1]) - 1  # Convert to 0-based
        input_params = extract_input_params_from_doe(doe_config, sim_id)
        X_inputs.append(input_params)

        if (i + 1) % 10 == 0 or i == n_samples - 1:
            print(f"  Loaded {i+1}/{n_samples} simulations", end='\r')

    print()  # New line
    X_data = np.array(X_inputs)
    Y_data = Y_outputs

    print(f"[OK] Data loaded: {X_data.shape} -> {Y_data.shape}")

    return X_data, Y_data, setup_data, output_params


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


def calculate_hyperparameters(n_samples, output_size):
    """
    Calculate dynamic hyperparameters for training.

    Parameters
    ----------
    n_samples : int
        Number of training samples
    output_size : int
        Output dimension size

    Returns
    -------
    dict
        Hyperparameters
    """
    # Learning rate: logarithmic decay with sample count
    lr_base = 0.0003
    lr_scale = min(1.0, n_samples / 200.0)
    learning_rate = lr_base + (0.0007 * lr_scale)

    # Batch size: logarithmic growth
    batch_size = max(2, min(32, int(n_samples ** 0.7 / 2)))

    # Epochs: inverse relationship with sample count
    epochs_scale = max(1.0, 100.0 / n_samples)
    epochs = int(500 * epochs_scale)
    epochs = min(max(epochs, 500), 3000)

    # Early stopping patience: ~5% of epochs
    early_stop_patience = max(20, int(epochs * 0.05))

    # LR scheduler patience: ~2% of epochs
    lr_patience = max(10, int(epochs * 0.02))

    return {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'early_stop_patience': early_stop_patience,
        'lr_patience': lr_patience
    }


def train_bottleneck_nn(X_train, Y_train, X_val, Y_val, hyperparams):
    """
    Train Bottleneck NN (current autoencoder architecture).

    Parameters
    ----------
    X_train, Y_train : np.ndarray
        Training data
    X_val, Y_val : np.ndarray
        Validation data
    hyperparams : dict
        Training hyperparameters

    Returns
    -------
    tuple
        (model, training_time, history)
    """
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]

    # Calculate latent dimension (dual constraint)
    n_samples = len(X_train)
    sample_based = max(8, n_samples // 2)
    size_based = max(10, output_dim // 100)  # 100:1 max compression
    latent_dim = min(max(sample_based, size_based), 100)

    print(f"\nBottleneck NN Configuration:")
    print(f"  Architecture: {input_dim} -> {latent_dim} -> {output_dim}")
    print(f"  Compression ratio: {output_dim / latent_dim:.1f}:1")

    # Create model configuration
    encoder_h1 = max(32, int(2 * input_dim))
    encoder_h2 = max(16, encoder_h1 // 2)
    decoder_h1 = max(64, int(2 * latent_dim))
    decoder_h2 = max(32, decoder_h1 // 2)

    model_config = {
        'input_dim': input_dim,
        'encoder_hidden_layers': [encoder_h1, encoder_h2],
        'latent_dim': latent_dim,
        'decoder_hidden_layers': [decoder_h1, decoder_h2],
        'output_dim': output_dim,
        'activation': 'relu',
        'learning_rate': hyperparams['learning_rate'],
        'batch_size': hyperparams['batch_size'],
        'epochs': hyperparams['epochs'],
        'validation_split': 0.15,  # Use 15% for validation
        'lr_patience': hyperparams['lr_patience']
    }

    # Create and train model
    model = AutoencoderModel(model_config)

    start_time = time.time()

    # Combine train and val for the model's internal split
    X_combined = np.vstack([X_train, X_val])
    Y_combined = np.vstack([Y_train, Y_val])

    model.train(
        X_combined,
        Y_combined,
        validation_split=0.15,
        early_stopping_patience=hyperparams['early_stop_patience']
    )

    training_time = time.time() - start_time

    # Extract history
    history = {
        'train_losses': model.history['train_loss'],
        'val_losses': model.history['val_loss'],
        'best_val_loss': min(model.history['val_loss']),
        'final_epoch': len(model.history['train_loss'])
    }

    return model, training_time, history


def train_direct_nn(X_train, Y_train, X_val, Y_val, hyperparams):
    """
    Train Direct NN (no bottleneck).

    Parameters
    ----------
    X_train, Y_train : np.ndarray
        Training data
    X_val, Y_val : np.ndarray
        Validation data
    hyperparams : dict
        Training hyperparameters

    Returns
    -------
    tuple
        (model, training_time, history)
    """
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]

    print(f"\nDirect NN Configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Output dimension: {output_dim}")

    # Create and train model
    model = DirectNNModel(input_dim, output_dim)

    start_time = time.time()

    history = model.train(
        X_train, Y_train, X_val, Y_val,
        epochs=hyperparams['epochs'],
        batch_size=hyperparams['batch_size'],
        learning_rate=hyperparams['learning_rate'],
        early_stop_patience=hyperparams['early_stop_patience'],
        lr_patience=hyperparams['lr_patience']
    )

    training_time = time.time() - start_time

    return model, training_time, history


def evaluate_model(model, X_test, Y_test, model_name):
    """
    Evaluate model performance.

    Parameters
    ----------
    model : Model
        Trained model
    X_test, Y_test : np.ndarray
        Test data
    model_name : str
        Model name for display

    Returns
    -------
    dict
        Evaluation metrics
    """
    print(f"\nEvaluating {model_name}...")

    # Make predictions
    Y_pred = model.predict(X_test)

    # Calculate metrics
    mse = np.mean((Y_pred - Y_test) ** 2)
    mae = np.mean(np.abs(Y_pred - Y_test))
    rmse = np.sqrt(mse)

    # R² score
    ss_res = np.sum((Y_test - Y_pred) ** 2)
    ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Per-sample errors
    per_sample_mse = np.mean((Y_pred - Y_test) ** 2, axis=1)

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'per_sample_mse': per_sample_mse,
        'predictions': Y_pred
    }


def create_comparison_plots(bottleneck_history, direct_history,
                            bottleneck_eval, direct_eval,
                            bottleneck_time, direct_time,
                            save_dir):
    """
    Create comparison visualization plots.

    Parameters
    ----------
    bottleneck_history : dict
        Bottleneck NN training history
    direct_history : dict
        Direct NN training history
    bottleneck_eval : dict
        Bottleneck NN evaluation metrics
    direct_eval : dict
        Direct NN evaluation metrics
    bottleneck_time : float
        Bottleneck NN training time (seconds)
    direct_time : float
        Direct NN training time (seconds)
    save_dir : Path
        Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))

    # 1. Training Loss Comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(bottleneck_history['train_losses'], label='Bottleneck Train', alpha=0.7)
    ax1.plot(bottleneck_history['val_losses'], label='Bottleneck Val', alpha=0.7)
    ax1.plot(direct_history['train_losses'], label='Direct Train', alpha=0.7, linestyle='--')
    ax1.plot(direct_history['val_losses'], label='Direct Val', alpha=0.7, linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 2. Final Loss Comparison (Bar chart)
    ax2 = plt.subplot(2, 3, 2)
    models = ['Bottleneck NN', 'Direct NN']
    final_losses = [
        bottleneck_history['val_losses'][-1],
        direct_history['val_losses'][-1]
    ]
    colors = ['#3498db', '#e74c3c']
    bars = ax2.bar(models, final_losses, color=colors, alpha=0.7)
    ax2.set_ylabel('Final Validation Loss')
    ax2.set_title('Final Loss Comparison')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}',
                ha='center', va='bottom')

    # 3. Test Set Performance (Bar chart)
    ax3 = plt.subplot(2, 3, 3)
    metrics = ['MSE', 'MAE', 'RMSE']
    bottleneck_metrics = [
        bottleneck_eval['mse'],
        bottleneck_eval['mae'],
        bottleneck_eval['rmse']
    ]
    direct_metrics = [
        direct_eval['mse'],
        direct_eval['mae'],
        direct_eval['rmse']
    ]

    x = np.arange(len(metrics))
    width = 0.35

    ax3.bar(x - width/2, bottleneck_metrics, width, label='Bottleneck NN',
            color='#3498db', alpha=0.7)
    ax3.bar(x + width/2, direct_metrics, width, label='Direct NN',
            color='#e74c3c', alpha=0.7)

    ax3.set_ylabel('Error')
    ax3.set_title('Test Set Error Metrics')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. R² Score Comparison
    ax4 = plt.subplot(2, 3, 4)
    r2_scores = [bottleneck_eval['r2'], direct_eval['r2']]
    bars = ax4.bar(models, r2_scores, color=colors, alpha=0.7)
    ax4.set_ylabel('R² Score')
    ax4.set_title('R² Score Comparison (higher is better)')
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')

    # 5. Training Time Comparison
    ax5 = plt.subplot(2, 3, 5)
    times = [bottleneck_time, direct_time]
    bars = ax5.bar(models, times, color=colors, alpha=0.7)
    ax5.set_ylabel('Time (seconds)')
    ax5.set_title('Training Time Comparison')
    ax5.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom')

    # 6. Per-Sample Error Distribution
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(bottleneck_eval['per_sample_mse'], bins=30, alpha=0.5,
             label='Bottleneck NN', color='#3498db')
    ax6.hist(direct_eval['per_sample_mse'], bins=30, alpha=0.5,
             label='Direct NN', color='#e74c3c')
    ax6.set_xlabel('Per-Sample MSE')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Error Distribution on Test Set')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[OK] Comparison plots saved to: {save_dir / 'architecture_comparison.png'}")


def save_comparison_report(dataset_info, bottleneck_eval, direct_eval,
                           bottleneck_time, direct_time,
                           bottleneck_history, direct_history,
                           hyperparams, save_dir):
    """
    Save detailed comparison report.

    Parameters
    ----------
    dataset_info : dict
        Dataset information
    bottleneck_eval : dict
        Bottleneck NN evaluation
    direct_eval : dict
        Direct NN evaluation
    bottleneck_time : float
        Bottleneck training time
    direct_time : float
        Direct training time
    bottleneck_history : dict
        Bottleneck training history
    direct_history : dict
        Direct training history
    hyperparams : dict
        Training hyperparameters
    save_dir : Path
        Save directory
    """
    save_dir = Path(save_dir)
    report_file = save_dir / 'comparison_report.txt'

    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ARCHITECTURE COMPARISON REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        f.write("\n" + "-"*70 + "\n")
        f.write("DATASET INFORMATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Dataset: {dataset_info['name']}\n")
        f.write(f"Input Parameters: {dataset_info['num_inputs']}\n")
        f.write(f"Output Parameters: {dataset_info['num_outputs']}\n")
        f.write(f"Total Simulations: {dataset_info['num_simulations']}\n")

        f.write("\n" + "-"*70 + "\n")
        f.write("TRAINING CONFIGURATION\n")
        f.write("-"*70 + "\n")
        for key, value in hyperparams.items():
            f.write(f"{key:25s}: {value}\n")

        f.write("\n" + "-"*70 + "\n")
        f.write("BOTTLENECK NN RESULTS\n")
        f.write("-"*70 + "\n")
        f.write(f"Training Time:           {bottleneck_time:.2f} seconds\n")
        f.write(f"Final Training Loss:     {bottleneck_history['train_losses'][-1]:.8f}\n")
        f.write(f"Final Validation Loss:   {bottleneck_history['val_losses'][-1]:.8f}\n")
        f.write(f"Epochs Completed:        {bottleneck_history['final_epoch']}\n")
        f.write(f"\nTest Set Performance:\n")
        f.write(f"  MSE:                   {bottleneck_eval['mse']:.8f}\n")
        f.write(f"  MAE:                   {bottleneck_eval['mae']:.8f}\n")
        f.write(f"  RMSE:                  {bottleneck_eval['rmse']:.8f}\n")
        f.write(f"  R² Score:              {bottleneck_eval['r2']:.6f}\n")

        f.write("\n" + "-"*70 + "\n")
        f.write("DIRECT NN RESULTS\n")
        f.write("-"*70 + "\n")
        f.write(f"Training Time:           {direct_time:.2f} seconds\n")
        f.write(f"Final Training Loss:     {direct_history['train_losses'][-1]:.8f}\n")
        f.write(f"Final Validation Loss:   {direct_history['val_losses'][-1]:.8f}\n")
        f.write(f"Epochs Completed:        {direct_history['final_epoch']}\n")
        f.write(f"\nTest Set Performance:\n")
        f.write(f"  MSE:                   {direct_eval['mse']:.8f}\n")
        f.write(f"  MAE:                   {direct_eval['mae']:.8f}\n")
        f.write(f"  RMSE:                  {direct_eval['rmse']:.8f}\n")
        f.write(f"  R² Score:              {direct_eval['r2']:.6f}\n")

        f.write("\n" + "-"*70 + "\n")
        f.write("COMPARISON SUMMARY\n")
        f.write("-"*70 + "\n")

        # Determine winner for each metric
        better_mse = "Bottleneck NN" if bottleneck_eval['mse'] < direct_eval['mse'] else "Direct NN"
        better_r2 = "Bottleneck NN" if bottleneck_eval['r2'] > direct_eval['r2'] else "Direct NN"
        better_time = "Bottleneck NN" if bottleneck_time < direct_time else "Direct NN"

        mse_diff = abs(bottleneck_eval['mse'] - direct_eval['mse'])
        mse_pct = (mse_diff / min(bottleneck_eval['mse'], direct_eval['mse'])) * 100

        f.write(f"\nMSE Winner:              {better_mse}\n")
        f.write(f"  Difference:            {mse_diff:.8f} ({mse_pct:.2f}%)\n")

        f.write(f"\nR² Winner:               {better_r2}\n")
        f.write(f"  Bottleneck R²:         {bottleneck_eval['r2']:.6f}\n")
        f.write(f"  Direct R²:             {direct_eval['r2']:.6f}\n")

        f.write(f"\nSpeed Winner:            {better_time}\n")
        f.write(f"  Time Difference:       {abs(bottleneck_time - direct_time):.2f} seconds\n")

        f.write("\n" + "="*70 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*70 + "\n")

        # Overall winner based on test MSE
        if bottleneck_eval['mse'] < direct_eval['mse']:
            improvement = ((direct_eval['mse'] - bottleneck_eval['mse']) / direct_eval['mse']) * 100
            f.write(f"\n[OK] Bottleneck NN performs {improvement:.2f}% better on test set MSE.\n")
        else:
            improvement = ((bottleneck_eval['mse'] - direct_eval['mse']) / bottleneck_eval['mse']) * 100
            f.write(f"\n[OK] Direct NN performs {improvement:.2f}% better on test set MSE.\n")

        f.write("\n" + "="*70 + "\n")

    print(f"[OK] Comparison report saved to: {report_file}")


def main():
    """Main comparison workflow."""
    clear_screen()
    print_header("ARCHITECTURE COMPARISON TOOL")
    print("\nCompare Bottleneck NN vs Direct NN performance")

    # Select project
    project = select_project()
    if not project:
        return

    # Select dataset
    dataset = select_dataset(project)
    if not dataset:
        return

    # Load data
    print_header("LOADING DATA")
    X_data, Y_data, setup_data, output_params = load_dataset_data(dataset)

    # Split data
    n_samples = len(X_data)
    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train, Y_train = X_data[train_idx], Y_data[train_idx]
    X_val, Y_val = X_data[val_idx], Y_data[val_idx]
    X_test, Y_test = X_data[test_idx], Y_data[test_idx]

    print(f"\nData split:")
    print(f"  Training:   {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test:       {len(X_test)} samples")

    # Calculate hyperparameters
    hyperparams = calculate_hyperparameters(len(X_train), Y_train.shape[1])

    print_header("HYPERPARAMETERS")
    for key, value in hyperparams.items():
        print(f"  {key:25s}: {value}")

    # Train Bottleneck NN
    print_header("TRAINING BOTTLENECK NN")
    bottleneck_model, bottleneck_time, bottleneck_history = train_bottleneck_nn(
        X_train, Y_train, X_val, Y_val, hyperparams
    )
    print(f"\n[OK] Bottleneck NN training completed in {bottleneck_time:.2f} seconds")

    # Train Direct NN
    print_header("TRAINING DIRECT NN")
    direct_model, direct_time, direct_history = train_direct_nn(
        X_train, Y_train, X_val, Y_val, hyperparams
    )
    print(f"\n[OK] Direct NN training completed in {direct_time:.2f} seconds")

    # Evaluate both models
    print_header("EVALUATION")
    bottleneck_eval = evaluate_model(bottleneck_model, X_test, Y_test, "Bottleneck NN")
    direct_eval = evaluate_model(direct_model, X_test, Y_test, "Direct NN")

    # Print comparison
    print_header("RESULTS COMPARISON")
    print(f"\n{'Metric':<20} {'Bottleneck NN':<20} {'Direct NN':<20} {'Winner':<15}")
    print("-" * 75)
    print(f"{'MSE':<20} {bottleneck_eval['mse']:<20.8f} {direct_eval['mse']:<20.8f} "
          f"{'Bottleneck' if bottleneck_eval['mse'] < direct_eval['mse'] else 'Direct':<15}")
    print(f"{'MAE':<20} {bottleneck_eval['mae']:<20.8f} {direct_eval['mae']:<20.8f} "
          f"{'Bottleneck' if bottleneck_eval['mae'] < direct_eval['mae'] else 'Direct':<15}")
    print(f"{'RMSE':<20} {bottleneck_eval['rmse']:<20.8f} {direct_eval['rmse']:<20.8f} "
          f"{'Bottleneck' if bottleneck_eval['rmse'] < direct_eval['rmse'] else 'Direct':<15}")
    print(f"{'R² Score':<20} {bottleneck_eval['r2']:<20.6f} {direct_eval['r2']:<20.6f} "
          f"{'Bottleneck' if bottleneck_eval['r2'] > direct_eval['r2'] else 'Direct':<15}")
    print(f"{'Training Time (s)':<20} {bottleneck_time:<20.2f} {direct_time:<20.2f} "
          f"{'Bottleneck' if bottleneck_time < direct_time else 'Direct':<15}")

    # Save results
    print_header("SAVING RESULTS")
    results_dir = Path(f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    create_comparison_plots(
        bottleneck_history, direct_history,
        bottleneck_eval, direct_eval,
        bottleneck_time, direct_time,
        results_dir
    )

    save_comparison_report(
        dataset, bottleneck_eval, direct_eval,
        bottleneck_time, direct_time,
        bottleneck_history, direct_history,
        hyperparams, results_dir
    )

    print(f"\n[OK] All results saved to: {results_dir.absolute()}")

    print_header("COMPARISON COMPLETE")
    input("\nPress Enter to exit...")


if __name__ == '__main__':
    main()
