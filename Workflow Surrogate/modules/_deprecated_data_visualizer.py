"""
Data Visualizer Module
=======================
Visualizes training results, comparison plots, and model performance.
Supports dynamic visualization based on data dimensionality (scalars, 2D fields, 3D fields).
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def data_visualization_menu(training_dir, ui_helpers):
    """
    Main menu for data visualization.

    Parameters
    ----------
    training_dir : Path
        Path to trained model directory
    ui_helpers : module
        UI helpers module
    """

    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("DATA VISUALIZATION")

        print(f"\nModel: {training_dir.name}")
        print(f"Location: {training_dir}")

        print(f"\n{'='*70}")
        print("  [1] View Training/Validation Curves")
        print("  [2] View Model Performance Metrics")
        print("  [3] Compare Predictions vs Actual")
        print("  [4] Error Distribution Analysis")
        print("  [0] Back to Main Menu")
        print("="*70)

        choice = ui_helpers.get_choice(4)

        if choice == 0:
            return

        elif choice == 1:
            view_training_curves(training_dir, ui_helpers)
        elif choice == 2:
            view_performance_metrics(training_dir, ui_helpers)
        elif choice == 3:
            view_comparison_plots(training_dir, ui_helpers)
        elif choice == 4:
            view_error_distribution(training_dir, ui_helpers)


def view_training_curves(training_dir, ui_helpers):
    """
    Display training and validation loss curves.

    Parameters
    ----------
    training_dir : Path
        Training directory
    ui_helpers : module
        UI helpers module
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("TRAINING/VALIDATION CURVES")

    print(f"\nModel: {training_dir.name}\n")

    model_info_file = training_dir / "model_info.json"

    if not model_info_file.exists():
        print("\n[X] Training history not found")
        ui_helpers.pause()
        return

    try:
        with open(model_info_file, 'r') as f:
            model_info = json.load(f)

        history = model_info.get('history', {})

        if not history or 'train_loss' not in history:
            print("\n[X] No training history available")
            ui_helpers.pause()
            return

        train_loss = history['train_loss']
        val_loss = history['val_loss']
        learning_rates = history.get('learning_rate', [])
        epochs = list(range(1, len(train_loss) + 1))

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle(f'Training History: {training_dir.name}', fontsize=14, fontweight='bold')

        # Loss plot
        ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training and Validation Loss')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Find best epoch
        best_epoch = np.argmin(val_loss) + 1
        best_val_loss = min(val_loss)
        ax1.axvline(best_epoch, color='g', linestyle='--', alpha=0.7,
                    label=f'Best (Epoch {best_epoch})')
        ax1.legend()

        # Learning rate plot
        if learning_rates:
            ax2.plot(epochs, learning_rates, 'g-', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)

            # Mark LR reductions
            for i in range(1, len(learning_rates)):
                if learning_rates[i] < learning_rates[i-1]:
                    ax2.axvline(i+1, color='r', linestyle=':', alpha=0.5)
        else:
            ax2.text(0.5, 0.5, 'Learning rate history not available',
                    ha='center', va='center', transform=ax2.transAxes)

        plt.tight_layout()

        print("[OK] Displaying training curves...")
        print(f"  Total epochs: {len(train_loss)}")
        print(f"  Best validation loss: {best_val_loss:.6e} (epoch {best_epoch})")
        print(f"  Final training loss: {train_loss[-1]:.6e}")
        print(f"  Final validation loss: {val_loss[-1]:.6e}")

        plt.show()

    except Exception as e:
        print(f"\n[X] Error loading training history: {e}")
        import traceback
        traceback.print_exc()

    ui_helpers.pause()


def view_performance_metrics(training_dir, ui_helpers):
    """
    Display model performance metrics.

    Parameters
    ----------
    training_dir : Path
        Training directory
    ui_helpers : module
        UI helpers module
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("MODEL PERFORMANCE METRICS")

    model_info_file = training_dir / "model_info.json"
    eval_results_file = training_dir / "evaluation_results.json"

    if not model_info_file.exists():
        print("\n[X] Model info file not found")
        ui_helpers.pause()
        return

    try:
        with open(model_info_file, 'r') as f:
            model_info = json.load(f)

        config = model_info.get('config', {})
        history = model_info.get('history', {})

        print(f"\nModel: {training_dir.name}")
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE:")
        print("="*70)
        print(f"  Type: Direct NN (Feedforward)")
        print(f"  Input Dimension: {config.get('input_dim', 'N/A')}")
        print(f"  Output Dimension: {config.get('output_dim', 'N/A')}")
        if 'hidden_dims' in model_info:
            hidden_dims = model_info['hidden_dims']
            print(f"  Hidden Layers: {' -> '.join(map(str, hidden_dims))}")

        print("\n" + "="*70)
        print("TRAINING CONFIGURATION:")
        print("="*70)
        print(f"  Learning Rate: {config.get('learning_rate', 'N/A')}")
        print(f"  Batch Size: {config.get('batch_size', 'N/A')}")
        print(f"  Epochs Configured: {config.get('epochs', 'N/A')}")
        if history and 'train_loss' in history:
            print(f"  Epochs Trained: {len(history['train_loss'])}")

        # Load evaluation results if available
        if eval_results_file.exists():
            with open(eval_results_file, 'r') as f:
                eval_results = json.load(f)

            print("\n" + "="*70)
            print("PERFORMANCE METRICS:")
            print("="*70)
            print(f"  R² Score: {eval_results.get('r2_score', 'N/A'):.6f}")
            print(f"  Mean Absolute Error: {eval_results.get('mae', 'N/A'):.6e}")
            print(f"  Root Mean Square Error: {eval_results.get('rmse', 'N/A'):.6e}")
            print(f"  Maximum Error: {eval_results.get('max_error', 'N/A'):.6e}")
            print(f"  Test Samples: {eval_results.get('n_test_samples', 'N/A')}")

            if 'dataset_name' in eval_results:
                print(f"  Source Dataset: {eval_results['dataset_name']}")

        print("\n" + "="*70)

    except Exception as e:
        print(f"\n[X] Error loading model info: {e}")

    ui_helpers.pause()


def view_comparison_plots(training_dir, ui_helpers):
    """
    Display prediction vs actual comparison plots.
    Loads model and dataset to generate interactive visualizations.

    Parameters
    ----------
    training_dir : Path
        Training directory
    ui_helpers : module
        UI helpers module
    """
    from .direct_nn_model import DirectNNModel
    from pathlib import Path

    ui_helpers.clear_screen()
    ui_helpers.print_header("PREDICTIONS VS ACTUAL")

    print(f"\nModel: {training_dir.name}\n")

    # Load model
    try:
        print("Loading model...")
        model = DirectNNModel.load(training_dir)
        print("[OK] Model loaded successfully!")
    except Exception as e:
        print(f"\n[X] Error loading model: {e}")
        ui_helpers.pause()
        return

    # Get source dataset from evaluation results
    eval_file = training_dir / "evaluation_results.json"
    if not eval_file.exists():
        print("\n[X] No evaluation results found. Cannot determine source dataset.")
        ui_helpers.pause()
        return

    with open(eval_file, 'r') as f:
        eval_results = json.load(f)

    dataset_name = eval_results.get('dataset_name', None)
    if not dataset_name:
        print("\n[X] Source dataset not recorded in evaluation results.")
        ui_helpers.pause()
        return

    # Find dataset directory
    project_dir = training_dir.parent.parent
    dataset_dir = project_dir / "cases" / dataset_name

    if not dataset_dir.exists():
        print(f"\n[X] Case directory not found: {dataset_dir}")
        ui_helpers.pause()
        return

    print(f"Case: {dataset_name}\n")

    # Load dataset output files to get available variables
    outputs_dir = dataset_dir / "dataset"
    if not outputs_dir.exists():
        print(f"\n[X] Dataset directory not found")
        ui_helpers.pause()
        return

    # Get output variable names from first file
    output_files = sorted(outputs_dir.glob("sim_*.npz"))
    if not output_files:
        print(f"\n[X] No simulation output files found")
        ui_helpers.pause()
        return

    sample_data = np.load(output_files[0], allow_pickle=True)
    output_vars = [key for key in sample_data.files if not key.startswith('_')]

    if not output_vars:
        print(f"\n[X] No output variables found in dataset")
        ui_helpers.pause()
        return

    # Display menu for variable selection
    print("="*70)
    print("SELECT OUTPUT VARIABLE TO VISUALIZE:")
    print("="*70)
    for i, var in enumerate(output_vars, 1):
        print(f"  [{i}] {var}")
    print(f"  [0] Back")
    print("="*70)

    choice = ui_helpers.get_choice(len(output_vars))
    if choice == 0:
        return

    selected_var = output_vars[choice - 1]

    # Load and visualize
    try:
        visualize_fluent_vs_nn(
            model=model,
            dataset_dir=dataset_dir,
            output_var=selected_var,
            ui_helpers=ui_helpers
        )
    except Exception as e:
        print(f"\n[X] Error during visualization: {e}")
        import traceback
        traceback.print_exc()

    ui_helpers.pause()


def view_error_distribution(training_dir, ui_helpers):
    """
    Display error distribution analysis from evaluation results.

    Parameters
    ----------
    training_dir : Path
        Training directory
    ui_helpers : module
        UI helpers module
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("ERROR DISTRIBUTION ANALYSIS")

    print(f"\nModel: {training_dir.name}")

    eval_results_file = training_dir / "evaluation_results.json"

    if not eval_results_file.exists():
        print("\n[X] Evaluation results not found")
        print("\nTo generate evaluation results:")
        print("  1. Go to 'Model Setup & Training' menu")
        print("  2. Load this model")
        print("  3. Test on a dataset")
        ui_helpers.pause()
        return

    try:
        with open(eval_results_file, 'r') as f:
            eval_results = json.load(f)

        print("\n" + "="*70)
        print("ERROR STATISTICS:")
        print("="*70)
        print(f"  R² Score: {eval_results.get('r2_score', 'N/A'):.6f}")
        print(f"  Mean Absolute Error: {eval_results.get('mae', 'N/A'):.6e}")
        print(f"  Root Mean Square Error: {eval_results.get('rmse', 'N/A'):.6e}")
        print(f"  Maximum Error: {eval_results.get('max_error', 'N/A'):.6e}")

        # Interpretation guide
        r2 = eval_results.get('r2_score', 0)
        print("\n" + "="*70)
        print("INTERPRETATION:")
        print("="*70)
        if r2 > 0.99:
            print("  [OK] Excellent fit - model captures nearly all variance")
        elif r2 > 0.95:
            print("  [OK] Very good fit - model is highly accurate")
        elif r2 > 0.90:
            print("  [OK] Good fit - model is reasonably accurate")
        elif r2 > 0.80:
            print("  [WARNING] Moderate fit - consider more training data or tuning")
        else:
            print("  [X] Poor fit - model may need redesign or more data")

        rmse = eval_results.get('rmse', 0)
        print(f"\n  Average prediction error: ~{rmse:.4f} (RMSE)")
        print(f"  Worst case error: {eval_results.get('max_error', 0):.4f}")

        # Per-output R² if available
        if 'per_output_r2' in eval_results:
            per_output_r2 = eval_results['per_output_r2']
            print("\n" + "="*70)
            print(f"PER-OUTPUT R² SCORES: ({len(per_output_r2)} outputs)")
            print("="*70)

            # Show summary statistics
            r2_array = np.array(per_output_r2)
            print(f"  Mean R²: {np.mean(r2_array):.6f}")
            print(f"  Min R²: {np.min(r2_array):.6f}")
            print(f"  Max R²: {np.max(r2_array):.6f}")
            print(f"  Std R²: {np.std(r2_array):.6f}")

            # Visualize if we have multiple outputs
            if len(per_output_r2) > 1:
                print("\nGenerating R² distribution plot...")

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(f'Per-Output R² Analysis: {training_dir.name}',
                            fontsize=14, fontweight='bold')

                # Histogram
                ax1.hist(per_output_r2, bins=min(50, len(per_output_r2)//2),
                        edgecolor='black', alpha=0.7)
                ax1.axvline(np.mean(per_output_r2), color='r', linestyle='--',
                           linewidth=2, label=f'Mean: {np.mean(per_output_r2):.4f}')
                ax1.set_xlabel('R² Score')
                ax1.set_ylabel('Frequency')
                ax1.set_title('R² Distribution Across Outputs')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Line plot
                ax2.plot(per_output_r2, 'b-', alpha=0.5, linewidth=1)
                ax2.axhline(np.mean(per_output_r2), color='r', linestyle='--',
                           linewidth=2, label='Mean')
                ax2.axhline(0.95, color='g', linestyle=':', alpha=0.5, label='R²=0.95')
                ax2.set_xlabel('Output Index')
                ax2.set_ylabel('R² Score')
                ax2.set_title('R² by Output Index')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()

    except Exception as e:
        print(f"\n[X] Error loading evaluation results: {e}")
        import traceback
        traceback.print_exc()

    ui_helpers.pause()


def visualize_fluent_vs_nn(model, dataset_dir, output_var, ui_helpers):
    """
    Create side-by-side comparison of Fluent results vs NN predictions.
    Automatically detects if data is 2D or 3D and creates appropriate plots.

    Parameters
    ----------
    model : DirectNNModel
        Trained neural network model
    dataset_dir : Path
        Dataset directory
    output_var : str
        Output variable name to visualize
    ui_helpers : module
        UI helpers module
    """
    print(f"\nLoading data for variable: {output_var}...")

    # Load inputs
    inputs_file = dataset_dir / "inputs" / "input_variables.csv"
    if not inputs_file.exists():
        print(f"[X] Input file not found: {inputs_file}")
        return

    import pandas as pd
    inputs_df = pd.read_csv(inputs_file)
    X_inputs = inputs_df.values

    # Load outputs for this variable
    outputs_dir = dataset_dir / "dataset"
    output_files = sorted(outputs_dir.glob("sim_*.npz"))

    Y_outputs = []
    coordinates = None

    for output_file in output_files:
        data = np.load(output_file, allow_pickle=True)

        # Handle nested structure
        if data[output_var].ndim == 0:  # Object array
            field_data = data[output_var].item()
            if isinstance(field_data, dict):
                # Multi-field output, concatenate all fields
                values = np.concatenate([v.flatten() for v in field_data.values()])
            else:
                values = field_data.flatten()
        else:
            values = data[output_var].flatten()

        Y_outputs.append(values)

        # Get coordinates from first file
        if coordinates is None and '_coordinates' in data.files:
            coord_data = data['_coordinates']
            if coord_data.ndim == 0:
                coordinates = coord_data.item()
            else:
                coordinates = coord_data

    Y_outputs = np.array(Y_outputs)

    print(f"  Loaded {len(X_inputs)} simulations")
    print(f"  Output shape: {Y_outputs.shape}")
    print(f"  Points per simulation: {Y_outputs.shape[1]}")

    # Make predictions
    print("\nGenerating NN predictions...")
    Y_pred = model.predict(X_inputs)

    # Select simulation
    print("\n" + "="*70)
    print("SELECT SIMULATION TO VISUALIZE:")
    print("="*70)
    print(f"  [1] Random simulation")
    print(f"  [2] Specific simulation number")
    print(f"  [0] Back")
    print("="*70)

    choice = ui_helpers.get_choice(2)
    if choice == 0:
        return
    elif choice == 1:
        sim_idx = np.random.randint(0, len(X_inputs))
    else:
        try:
            sim_idx = int(input(f"\nEnter simulation number (0-{len(X_inputs)-1}): ").strip())
            if sim_idx < 0 or sim_idx >= len(X_inputs):
                print(f"[X] Invalid simulation number")
                return
        except:
            print(f"[X] Invalid input")
            return

    # Get data for selected simulation
    fluent_data = Y_outputs[sim_idx]
    nn_data = Y_pred[sim_idx]
    params = X_inputs[sim_idx]

    # Detect dimensionality
    n_points = len(fluent_data)

    if n_points <= 10:
        plot_type = 'scalar'
    elif n_points <= 1000:
        plot_type = '2d'
    else:
        plot_type = '3d'

    print(f"\nSimulation {sim_idx}")
    print(f"  Parameters: {params}")
    print(f"  Data points: {n_points}")
    print(f"  Detected type: {plot_type}")

    # Calculate error metrics
    error = nn_data - fluent_data
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))

    from sklearn.metrics import r2_score
    r2 = r2_score(fluent_data, nn_data)

    print(f"\n  R² Score: {r2:.6f}")
    print(f"  MAE: {mae:.6e}")
    print(f"  RMSE: {rmse:.6e}")
    print(f"  Max Error: {max_error:.6e}")

    # Create visualization
    if plot_type == 'scalar':
        _plot_scalar_comparison(fluent_data, nn_data, output_var, params, r2, mae, rmse)
    elif plot_type == '2d':
        _plot_2d_comparison(fluent_data, nn_data, coordinates, output_var, params, r2, mae, rmse)
    else:
        _plot_3d_comparison(fluent_data, nn_data, coordinates, output_var, params, r2, mae, rmse)


def _plot_scalar_comparison(fluent_data, nn_data, var_name, params, r2, mae, rmse):
    """Plot scalar output comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar comparison
    x = np.arange(len(fluent_data))
    width = 0.35

    ax1.bar(x - width/2, fluent_data, width, label='Fluent', alpha=0.8, color='steelblue')
    ax1.bar(x + width/2, nn_data, width, label='NN Prediction', alpha=0.8, color='coral')
    ax1.set_xlabel('Output Index')
    ax1.set_ylabel('Value')
    ax1.set_title(f'{var_name} - Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Parity plot
    ax2.scatter(fluent_data, nn_data, alpha=0.6, s=100)
    ax2.plot([fluent_data.min(), fluent_data.max()],
             [fluent_data.min(), fluent_data.max()],
             'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Fluent')
    ax2.set_ylabel('NN Prediction')
    ax2.set_title(f'Parity Plot (R²={r2:.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Parameters: {params}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def _plot_2d_comparison(fluent_data, nn_data, coordinates, var_name, params, r2, mae, rmse):
    """Plot 2D field comparison."""
    if coordinates is None or len(coordinates) != len(fluent_data):
        print("[X] Coordinates not available or dimension mismatch")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Fluent result
    scatter1 = axes[0].scatter(coordinates[:, 0], coordinates[:, 1],
                               c=fluent_data, cmap='hot', s=20, edgecolors='none')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('Fluent Result')
    axes[0].set_aspect('equal')
    plt.colorbar(scatter1, ax=axes[0], label=var_name)

    # NN prediction
    scatter2 = axes[1].scatter(coordinates[:, 0], coordinates[:, 1],
                               c=nn_data, cmap='hot', s=20, edgecolors='none')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title(f'NN Prediction (R²={r2:.4f})')
    axes[1].set_aspect('equal')
    plt.colorbar(scatter2, ax=axes[1], label=var_name)

    # Match color scales
    vmin = min(fluent_data.min(), nn_data.min())
    vmax = max(fluent_data.max(), nn_data.max())
    scatter1.set_clim(vmin, vmax)
    scatter2.set_clim(vmin, vmax)

    # Error plot
    error = nn_data - fluent_data
    from matplotlib.colors import TwoSlopeNorm
    error_max = max(abs(error.min()), abs(error.max()))
    norm = TwoSlopeNorm(vmin=-error_max, vcenter=0, vmax=error_max)

    scatter3 = axes[2].scatter(coordinates[:, 0], coordinates[:, 1],
                               c=error, cmap='RdBu_r', s=20, edgecolors='none', norm=norm)
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].set_title(f'Error (RMSE={rmse:.4e})')
    axes[2].set_aspect('equal')
    plt.colorbar(scatter3, ax=axes[2], label='Error')

    plt.suptitle(f'{var_name} | Parameters: {params} | MAE: {mae:.4e}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def _plot_3d_comparison(fluent_data, nn_data, coordinates, var_name, params, r2, mae, rmse):
    """Plot 3D field comparison."""
    if coordinates is None or len(coordinates) != len(fluent_data):
        print("[X] Coordinates not available or dimension mismatch")
        return

    # Subsample for visualization performance
    n_points = len(fluent_data)
    step = max(1, n_points // 3000)

    coords_sub = coordinates[::step]
    fluent_sub = fluent_data[::step]
    nn_sub = nn_data[::step]

    print(f"  Plotting {len(coords_sub)} of {n_points} points")

    fig = plt.figure(figsize=(18, 5))

    # Fluent result
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(coords_sub[:, 0], coords_sub[:, 1], coords_sub[:, 2],
                           c=fluent_sub, cmap='hot', s=8, alpha=0.6, edgecolors='none')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Fluent Result')
    plt.colorbar(scatter1, ax=ax1, shrink=0.8, label=var_name)

    # NN prediction
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(coords_sub[:, 0], coords_sub[:, 1], coords_sub[:, 2],
                           c=nn_sub, cmap='hot', s=8, alpha=0.6, edgecolors='none')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'NN Prediction (R²={r2:.4f})')
    plt.colorbar(scatter2, ax=ax2, shrink=0.8, label=var_name)

    # Match color scales
    vmin = min(fluent_data.min(), nn_data.min())
    vmax = max(fluent_data.max(), nn_data.max())
    scatter1.set_clim(vmin, vmax)
    scatter2.set_clim(vmin, vmax)

    # Error plot
    error_sub = nn_sub - fluent_sub
    from matplotlib.colors import TwoSlopeNorm
    error_max = max(abs(error_sub.min()), abs(error_sub.max()))
    norm = TwoSlopeNorm(vmin=-error_max, vcenter=0, vmax=error_max)

    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = ax3.scatter(coords_sub[:, 0], coords_sub[:, 1], coords_sub[:, 2],
                           c=error_sub, cmap='RdBu_r', s=8, alpha=0.6,
                           edgecolors='none', norm=norm)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title(f'Error (RMSE={rmse:.4e})')
    plt.colorbar(scatter3, ax=ax3, shrink=0.8, label='Error')

    plt.suptitle(f'{var_name} | Parameters: {params} | MAE: {mae:.4e}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
