"""
Multi-Model Visualizer Module
===============================
Visualizes predictions from multiple trained models (1D, 2D, 3D).
Supports comparison plots, error analysis, and custom parameter prediction.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D

from .scalar_nn_model import ScalarNNModel
from .field_nn_model import FieldNNModel
from .volume_nn_model import VolumeNNModel


def visualization_menu(dataset_dir, ui_helpers):
    """
    Main visualization menu for trained models.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    ui_helpers : module
        UI helpers module
    """
    while True:
        ui_helpers.clear_screen()
        ui_helpers.print_header("DATA VISUALIZATION")

        print(f"\nCase: {dataset_dir.name}")

        # Check for models
        models_dir = dataset_dir / "models"
        if not models_dir.exists() or not list(models_dir.glob("*_metadata.json")):
            print("\n[X] No trained models found. Train models first.")
            ui_helpers.pause()
            return

        # Load training summary
        summary_file = models_dir / "training_summary.json"
        if not summary_file.exists():
            print("\n[X] Training summary not found.")
            ui_helpers.pause()
            return

        with open(summary_file, 'r') as f:
            summary = json.load(f)

        print(f"Trained models: {summary['n_models']}")
        print(f"Trained: {summary['trained_date']}")

        print(f"\n{'='*70}")
        print("  [1] View Model Performance Summary")
        print("  [2] Compare Predictions vs Ground Truth")
        print("  [3] Predict with Custom Parameters")
        print("  [0] Back")
        print("="*70)

        choice = ui_helpers.get_choice(3)

        if choice == 0:
            return
        elif choice == 1:
            view_model_summary(dataset_dir, summary, ui_helpers)
        elif choice == 2:
            compare_predictions(dataset_dir, summary, ui_helpers)
        elif choice == 3:
            predict_custom_parameters(dataset_dir, summary, ui_helpers)


def view_model_summary(dataset_dir, summary, ui_helpers):
    """
    Display summary of all trained models.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    summary : dict
        Training summary data
    ui_helpers : module
        UI helpers module
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("MODEL PERFORMANCE SUMMARY")

    print(f"\nCase: {summary['case_name']}")
    print(f"Training Date: {summary['trained_date']}")
    print(f"Total Models: {summary['n_models']}")
    print(f"Train Samples: {summary['n_train_samples']}")
    print(f"Test Samples: {summary['n_test_samples']}")

    print(f"\n{'='*80}")
    print(f"{'Model Name':<35s} {'Type':<6s} {'R² (Test)':<12s} {'MAE':<12s} {'RMSE':<12s}")
    print(f"{'='*80}")

    for model_meta in summary['models']:
        name = model_meta['model_name']
        mtype = model_meta['output_type']
        r2 = model_meta['test_metrics']['r2']
        mae = model_meta['test_metrics']['mae']
        rmse = model_meta['test_metrics']['rmse']

        print(f"{name:<35s} {mtype:<6s} {r2:>11.4f} {mae:>11.4e} {rmse:>11.4e}")

    print(f"{'='*80}")

    ui_helpers.pause()


def compare_predictions(dataset_dir, summary, ui_helpers):
    """
    Compare model predictions with ground truth from dataset.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    summary : dict
        Training summary data
    ui_helpers : module
        UI helpers module
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("COMPARE PREDICTIONS")

    # Select model
    print("\nAvailable Models:")
    for i, model_meta in enumerate(summary['models'], 1):
        print(f"  [{i}] {model_meta['model_name']} ({model_meta['output_type']})")

    print(f"  [0] Cancel")

    choice = ui_helpers.get_choice(len(summary['models']))
    if choice == 0:
        return

    model_meta = summary['models'][choice - 1]
    model_name = model_meta['model_name']
    output_type = model_meta['output_type']

    print(f"\nLoading model: {model_name}...")

    # Load model
    models_dir = dataset_dir / "models"
    model_path = models_dir / model_name

    try:
        if output_type == '1D':
            model = ScalarNNModel.load(model_path)
        elif output_type == '2D':
            model = FieldNNModel.load(model_path)
        else:  # 3D
            model = VolumeNNModel.load(model_path)
    except Exception as e:
        print(f"\n[X] Error loading model: {e}")
        ui_helpers.pause()
        return

    # Load ground truth data
    print(f"Loading ground truth data...")
    try:
        # Load simulation data
        from .multi_model_trainer import load_training_data
        data = load_training_data(dataset_dir)

        output_key = model_meta['output_key']
        if output_key not in data['outputs']:
            print(f"\n[X] Output '{output_key}' not found in data.")
            ui_helpers.pause()
            return

        X_params = data['parameters']
        Y_true = data['outputs'][output_key]

        # Load coordinates if available (for 2D/3D visualization)
        coordinates = None
        if output_type in ['2D', '3D']:
            # Try to load coordinates from first simulation file
            dataset_output_dir = dataset_dir / "dataset"
            output_files = sorted(dataset_output_dir.glob("sim_*.npz"))
            if output_files:
                # Get location from output_key (format: "location_field")
                location = model_meta['location']
                coord_key = f"{location}|coordinates"

                try:
                    sample_file = np.load(output_files[0], allow_pickle=True)
                    if coord_key in sample_file.files:
                        coordinates = sample_file[coord_key]
                        print(f"  Loaded coordinates: {coordinates.shape}")
                    else:
                        print(f"  ⚠ Coordinates not found (key: '{coord_key}')")
                        print(f"    Re-run simulations to extract coordinates")
                except Exception as e:
                    print(f"  ⚠ Could not load coordinates: {e}")

        # For 1D outputs, plot all samples. For 2D/3D, select a specific sample.
        if output_type == '1D':
            # Predict for all samples
            print(f"\nPredicting for all {len(X_params)} samples...")
            Y_pred_all = model.predict(X_params)

            # Compute metrics
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            r2 = r2_score(Y_true, Y_pred_all)
            mae = mean_absolute_error(Y_true, Y_pred_all)
            rmse = np.sqrt(mean_squared_error(Y_true, Y_pred_all))

            print(f"\nOverall Performance (all {len(X_params)} samples):")
            print(f"  R² Score: {r2:.4f}")
            print(f"  MAE: {mae:.4e}")
            print(f"  RMSE: {rmse:.4e}")

            # Plot comparison for all samples
            plot_comparison(Y_true, Y_pred_all, output_type, model_name, r2, mae, rmse, coordinates)
        else:
            # For 2D/3D, select a specific sample to visualize
            print(f"\nSelect sample to visualize (1-{len(X_params)}) or 0 for random: ", end='')
            sample_choice = input().strip()

            if sample_choice == '0' or not sample_choice:
                sample_idx = np.random.randint(0, len(X_params))
            else:
                sample_idx = int(sample_choice) - 1
                sample_idx = max(0, min(len(X_params) - 1, sample_idx))

            # Predict
            X_sample = X_params[sample_idx:sample_idx+1]
            Y_pred = model.predict(X_sample)[0]
            Y_ground_truth = Y_true[sample_idx]

            # Compute metrics
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            r2 = r2_score(Y_ground_truth, Y_pred)
            mae = mean_absolute_error(Y_ground_truth, Y_pred)
            rmse = np.sqrt(mean_squared_error(Y_ground_truth, Y_pred))

            print(f"\nSample {sample_idx + 1}/{len(X_params)}")
            print(f"  R² Score: {r2:.4f}")
            print(f"  MAE: {mae:.4e}")
            print(f"  RMSE: {rmse:.4e}")

            # Plot comparison
            plot_comparison(Y_ground_truth, Y_pred, output_type, model_name, r2, mae, rmse, coordinates)

    except Exception as e:
        print(f"\n[X] Error during visualization: {e}")
        import traceback
        traceback.print_exc()

    ui_helpers.pause()


def plot_comparison(Y_true, Y_pred, output_type, model_name, r2, mae, rmse, coordinates=None):
    """
    Plot comparison between ground truth and prediction.

    Parameters
    ----------
    Y_true : np.ndarray
        Ground truth data
    Y_pred : np.ndarray
        Predicted data
    output_type : str
        '1D', '2D', or '3D'
    model_name : str
        Model name for title
    r2 : float
        R² score
    mae : float
        Mean absolute error
    rmse : float
        Root mean squared error
    coordinates : np.ndarray, optional
        Physical coordinates (n_points, 3) for 2D/3D data
    """
    if output_type == '1D':
        # Scatter plot for 1D scalar data
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter plot
        ax = axes[0]
        ax.scatter(Y_true, Y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.plot([Y_true.min(), Y_true.max()], [Y_true.min(), Y_true.max()],
                'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Ground Truth', fontsize=12)
        ax.set_ylabel('Prediction', fontsize=12)
        ax.set_title(f'{model_name}\nR²={r2:.4f}', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Error plot
        ax = axes[1]
        error = Y_pred - Y_true
        ax.hist(error, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Prediction Error', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Error Distribution\nMAE={mae:.4e}, RMSE={rmse:.4e}',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    else:
        # 2D/3D field data - show three columns: Ground Truth | Prediction | Error
        # Common colormap limits for Ground Truth and Prediction
        vmin = min(Y_true.min(), Y_pred.min())
        vmax = max(Y_true.max(), Y_pred.max())

        # Determine if this is 3D volume data or 2D surface data
        is_3d_volume = (output_type == '3D')

        # Use physical coordinates if available, otherwise fall back to point indices
        if coordinates is not None and len(coordinates) == len(Y_true):
            x = coordinates[:, 0]  # X coordinates
            y = coordinates[:, 1]  # Y coordinates
            z = coordinates[:, 2] if is_3d_volume else None  # Z coordinates for 3D
            xlabel = 'X (m)'
            ylabel = 'Y (m)'
            zlabel = 'Z (m)'
            plot_type = '3d_spatial' if is_3d_volume else '2d_spatial'
        else:
            # Fallback: use point indices
            x = np.arange(len(Y_true))
            y = Y_true  # Will be overridden per subplot
            z = None
            xlabel = 'Point Index'
            ylabel = 'Value'
            zlabel = 'Value'
            plot_type = '1d_line'

        if plot_type == '3d_spatial':
            # 3D volume plots with isometric view and interactive rotation
            fig = plt.figure(figsize=(20, 7))

            # Subsample for visualization performance (max ~3000 points)
            n_points = len(Y_true)
            step = max(1, n_points // 3000)
            x_plot = x[::step]
            y_plot = y[::step]
            z_plot = z[::step]
            Y_true_plot = Y_true[::step]
            Y_pred_plot = Y_pred[::step]
            error_plot = Y_pred_plot - Y_true_plot

            print(f"\n  Plotting {len(x_plot)} points (subsampled from {n_points})")

            # Calculate equal aspect ratio limits
            max_range = np.array([
                x_plot.max() - x_plot.min(),
                y_plot.max() - y_plot.min(),
                z_plot.max() - z_plot.min()
            ]).max() / 2.0

            mid_x = (x_plot.max() + x_plot.min()) * 0.5
            mid_y = (y_plot.max() + y_plot.min()) * 0.5
            mid_z = (z_plot.max() + z_plot.min()) * 0.5

            # Column 1: Ground Truth
            ax1 = fig.add_subplot(1, 3, 1, projection='3d')
            scatter1 = ax1.scatter(x_plot, y_plot, z_plot, c=Y_true_plot,
                                  cmap='viridis', s=8, alpha=0.6,
                                  edgecolors='none', vmin=vmin, vmax=vmax)
            ax1.set_xlabel(xlabel, fontsize=9)
            ax1.set_ylabel(ylabel, fontsize=9)
            ax1.set_zlabel(zlabel, fontsize=9)
            ax1.set_title('Ground Truth', fontsize=10, fontweight='bold')
            ax1.view_init(elev=30, azim=45)  # Isometric view
            # Equal aspect ratio
            ax1.set_xlim(mid_x - max_range, mid_x + max_range)
            ax1.set_ylim(mid_y - max_range, mid_y + max_range)
            ax1.set_zlim(mid_z - max_range, mid_z + max_range)
            cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.1, shrink=0.7)
            cbar1.set_label('Value', fontsize=8)

            # Column 2: Prediction
            ax2 = fig.add_subplot(1, 3, 2, projection='3d')
            scatter2 = ax2.scatter(x_plot, y_plot, z_plot, c=Y_pred_plot,
                                  cmap='viridis', s=8, alpha=0.6,
                                  edgecolors='none', vmin=vmin, vmax=vmax)
            ax2.set_xlabel(xlabel, fontsize=9)
            ax2.set_ylabel(ylabel, fontsize=9)
            ax2.set_zlabel(zlabel, fontsize=9)
            ax2.set_title(f'Prediction\nR²={r2:.4f}', fontsize=10, fontweight='bold')
            ax2.view_init(elev=30, azim=45)  # Isometric view
            # Equal aspect ratio
            ax2.set_xlim(mid_x - max_range, mid_x + max_range)
            ax2.set_ylim(mid_y - max_range, mid_y + max_range)
            ax2.set_zlim(mid_z - max_range, mid_z + max_range)
            cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.1, shrink=0.7)
            cbar2.set_label('Value', fontsize=8)

            # Column 3: Error (diverging colormap centered at zero)
            ax3 = fig.add_subplot(1, 3, 3, projection='3d')
            error_max = max(abs(error_plot.min()), abs(error_plot.max()))
            if error_max == 0:
                error_max = 1e-10
            norm = TwoSlopeNorm(vmin=-error_max, vcenter=0, vmax=error_max)
            scatter3 = ax3.scatter(x_plot, y_plot, z_plot, c=error_plot,
                                  cmap='RdBu_r', s=8, alpha=0.6,
                                  edgecolors='none', norm=norm)
            ax3.set_xlabel(xlabel, fontsize=9)
            ax3.set_ylabel(ylabel, fontsize=9)
            ax3.set_zlabel(zlabel, fontsize=9)
            ax3.set_title(f'Error (Pred - Truth)\nMAE={mae:.4e}, RMSE={rmse:.4e}',
                         fontsize=10, fontweight='bold')
            ax3.view_init(elev=30, azim=45)  # Isometric view
            # Equal aspect ratio
            ax3.set_xlim(mid_x - max_range, mid_x + max_range)
            ax3.set_ylim(mid_y - max_range, mid_y + max_range)
            ax3.set_zlim(mid_z - max_range, mid_z + max_range)
            cbar3 = plt.colorbar(scatter3, ax=ax3, pad=0.1, shrink=0.7)
            cbar3.set_label('Error', fontsize=8)

            fig.suptitle(f'{model_name} - Interactive 3D (Click and drag to rotate)',
                        fontsize=13, fontweight='bold')

        elif plot_type == '2d_spatial':
            # 2D surface plots
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Column 1: Ground Truth
            ax = axes[0]
            scatter1 = ax.scatter(x, y, c=Y_true, cmap='viridis', s=12, alpha=0.8,
                                 edgecolors='none', vmin=vmin, vmax=vmax)
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title('Ground Truth', fontsize=13, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            cbar1 = plt.colorbar(scatter1, ax=ax)
            cbar1.set_label('Value', fontsize=10)

            # Column 2: Prediction
            ax = axes[1]
            scatter2 = ax.scatter(x, y, c=Y_pred, cmap='viridis', s=12, alpha=0.8,
                                 edgecolors='none', vmin=vmin, vmax=vmax)
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'Prediction\nR²={r2:.4f}', fontsize=13, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            cbar2 = plt.colorbar(scatter2, ax=ax)
            cbar2.set_label('Value', fontsize=10)

            # Column 3: Error (diverging colormap centered at zero)
            ax = axes[2]
            error = Y_pred - Y_true
            error_max = max(abs(error.min()), abs(error.max()))
            if error_max == 0:
                error_max = 1e-10

            norm = TwoSlopeNorm(vmin=-error_max, vcenter=0, vmax=error_max)
            scatter3 = ax.scatter(x, y, c=error, cmap='RdBu_r', s=12, alpha=0.8,
                                 edgecolors='none', norm=norm)
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'Error (Pred - Truth)\nMAE={mae:.4e}, RMSE={rmse:.4e}',
                         fontsize=13, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            cbar3 = plt.colorbar(scatter3, ax=ax)
            cbar3.set_label('Error', fontsize=10)

            fig.suptitle(model_name, fontsize=15, fontweight='bold', y=0.98)

        else:
            # Fallback: line plots with point indices
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Column 1: Ground Truth
            ax = axes[0]
            scatter1 = ax.scatter(x, Y_true, c=Y_true, cmap='viridis', s=5, alpha=0.8,
                                 edgecolors='none', vmin=vmin, vmax=vmax)
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title('Ground Truth', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            cbar1 = plt.colorbar(scatter1, ax=ax)
            cbar1.set_label('Value', fontsize=10)

            # Column 2: Prediction
            ax = axes[1]
            scatter2 = ax.scatter(x, Y_pred, c=Y_pred, cmap='viridis', s=5, alpha=0.8,
                                 edgecolors='none', vmin=vmin, vmax=vmax)
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'Prediction\nR²={r2:.4f}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            cbar2 = plt.colorbar(scatter2, ax=ax)
            cbar2.set_label('Value', fontsize=10)

            # Column 3: Error
            ax = axes[2]
            error = Y_pred - Y_true
            error_max = max(abs(error.min()), abs(error.max()))
            if error_max == 0:
                error_max = 1e-10

            norm = TwoSlopeNorm(vmin=-error_max, vcenter=0, vmax=error_max)
            scatter3 = ax.scatter(x, error, c=error, cmap='RdBu_r', s=5, alpha=0.8,
                                 edgecolors='none', norm=norm)
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel('Error', fontsize=11)
            ax.set_title(f'Error (Pred - Truth)\nMAE={mae:.4e}, RMSE={rmse:.4e}',
                         fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
            cbar3 = plt.colorbar(scatter3, ax=ax)
            cbar3.set_label('Error', fontsize=10)

            fig.suptitle(model_name, fontsize=15, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.show()


def predict_custom_parameters(dataset_dir, summary, ui_helpers):
    """
    Run Fluent simulation with custom parameters, then compare with model predictions.

    Parameters
    ----------
    dataset_dir : Path
        Case directory
    summary : dict
        Training summary data
    ui_helpers : module
        UI helpers module
    """
    ui_helpers.clear_screen()
    ui_helpers.print_header("PREDICT WITH CUSTOM PARAMETERS")

    print("\nThis feature will:")
    print("  1. Get custom parameter values from you")
    print("  2. Run Fluent simulation with those parameters")
    print("  3. Run model predictions")
    print("  4. Compare Fluent vs Model with plots")

    # Load model setup to get parameter info
    setup_file = dataset_dir / "model_setup.json"
    with open(setup_file, 'r') as f:
        setup_data = json.load(f)

    # Get parameter names and ranges
    doe_config = setup_data.get('doe_configuration', {})
    param_info = []
    bc_values_dict = {}  # For Fluent simulation

    # Build parameter info from model_inputs
    for input_item in setup_data['model_inputs']:
        bc_name = input_item['name']
        bc_type = input_item['category']
        doe_params = doe_config.get(bc_name, {})

        for param_name, values in doe_params.items():
            if values:
                param_info.append({
                    'bc_name': bc_name,
                    'bc_type': bc_type,
                    'param_name': param_name,
                    'param_path': param_name,
                    'full_name': f"{bc_name}.{param_name}",
                    'min': min(values),
                    'max': max(values)
                })

    print(f"\nInput Parameters ({len(param_info)}):")
    for i, info in enumerate(param_info, 1):
        print(f"  {i}. {info['full_name']}: [{info['min']:.3f}, {info['max']:.3f}]")

    # Get user input
    print(f"\nEnter parameter values (or press Enter for random):")
    custom_params = []

    for info in param_info:
        user_input = input(f"  {info['full_name']}: ").strip()

        if user_input:
            try:
                value = float(user_input)
            except ValueError:
                print(f"    Invalid input, using random value")
                value = np.random.uniform(info['min'], info['max'])
        else:
            value = np.random.uniform(info['min'], info['max'])
            print(f"    Using: {value:.3f}")

        custom_params.append(value)

        # Format for apply_boundary_conditions
        bc_key = f"{info['bc_name']}|{info['param_name']}"
        bc_values_dict[bc_key] = {
            'bc_name': info['bc_name'],
            'bc_type': info['bc_type'],
            'param_name': info['param_name'],
            'param_path': info['param_path'],
            'value': value
        }

    X_custom = np.array([custom_params])

    # Run Fluent simulation
    print(f"\n{'='*70}")
    print("RUNNING FLUENT SIMULATION")
    print(f"{'='*70}")

    try:
        from . import simulation_runner
        import ansys.fluent.core as pyfluent
        from tkinter import Tk, filedialog

        # Load case file
        case_file = setup_data.get('case_file', '')

        # Check if case file exists
        if not case_file or not Path(case_file).exists():
            print(f"\n[!] Case file not found in model_setup.json")
            if case_file:
                print(f"    Expected: {case_file}")

            # Try to find .cas files in project directory
            project_dir = dataset_dir.parent.parent
            cas_files = list(project_dir.glob("**/*.cas"))
            cas_files.extend(list(project_dir.glob("**/*.cas.h5")))
            cas_files.extend(list(project_dir.glob("**/*.cas.gz")))

            if cas_files:
                print(f"\n  Found {len(cas_files)} case file(s) in project:")
                for i, cf in enumerate(cas_files[:10], 1):  # Show max 10
                    print(f"    [{i}] {cf.name}")
                    print(f"        {cf.parent}")

                if len(cas_files) > 10:
                    print(f"    ... and {len(cas_files) - 10} more")

                print(f"\n  [Number] Select case file")
                print(f"  [B] Browse for case file")
                print(f"  [0] Cancel")

                choice = input("\nEnter choice: ").strip()

                if choice == '0':
                    return
                elif choice.upper() == 'B':
                    # Open file browser
                    print("\nOpening file browser...")
                    Tk().withdraw()
                    case_file = filedialog.askopenfilename(
                        title="Select Fluent Case File",
                        filetypes=[
                            ("Fluent Case Files", "*.cas *.cas.h5 *.cas.gz"),
                            ("All Files", "*.*")
                        ],
                        initialdir=str(project_dir)
                    )
                    if not case_file:
                        print("\n[X] No file selected")
                        ui_helpers.pause()
                        return
                    case_file = Path(case_file)
                elif choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(cas_files):
                        case_file = cas_files[idx]
                    else:
                        print("\n[X] Invalid selection")
                        ui_helpers.pause()
                        return
                else:
                    print("\n[X] Invalid choice")
                    ui_helpers.pause()
                    return
            else:
                # No case files found - open file browser
                print(f"\n  No case files found in project directory.")
                print(f"  Opening file browser...")
                Tk().withdraw()
                case_file = filedialog.askopenfilename(
                    title="Select Fluent Case File",
                    filetypes=[
                        ("Fluent Case Files", "*.cas *.cas.h5 *.cas.gz"),
                        ("All Files", "*.*")
                    ],
                    initialdir=str(project_dir)
                )
                if not case_file:
                    print("\n[X] No file selected")
                    ui_helpers.pause()
                    return
                case_file = Path(case_file)

            # Save case file to model_setup.json for future use
            setup_data['case_file'] = str(case_file)
            setup_file = dataset_dir / "model_setup.json"
            with open(setup_file, 'w') as f:
                json.dump(setup_data, f, indent=2)
            print(f"\n  ✓ Case file saved to model_setup.json")
        else:
            case_file = Path(case_file)

        print(f"\n  Using case file: {Path(case_file).name}")
        print(f"  Full path: {case_file}")

        print(f"\nStarting Fluent...")
        from ansys.fluent.core.launcher.launcher import UIMode
        solver = pyfluent.launch_fluent(
            precision='double',
            processor_count=4,
            mode='solver',
            ui_mode=UIMode.HIDDEN_GUI
        )

        print(f"Loading case file...")
        solver.settings.file.read_case(file_name=str(case_file))

        # Apply parameter values
        print(f"\nApplying parameter values...")
        if not simulation_runner.apply_boundary_conditions(solver, bc_values_dict):
            print("\n[X] Failed to apply boundary conditions")
            solver.exit()
            ui_helpers.pause()
            return

        # Run simulation
        print(f"\nRunning simulation...")
        solver.settings.solution.initialization.hybrid_initialize()
        solver.settings.solution.run_calculation.iterate(iter_count=100)

        # Extract outputs
        print(f"\nExtracting outputs...")
        fluent_results = simulation_runner.extract_field_data(
            solver,
            setup_data,
            dataset_dir
        )

        # Close Fluent
        print(f"\nClosing Fluent...")
        solver.exit()

    except Exception as e:
        print(f"\n[X] Error running Fluent simulation: {e}")
        import traceback
        traceback.print_exc()
        ui_helpers.pause()
        return

    # Run model predictions
    print(f"\n{'='*70}")
    print("RUNNING MODEL PREDICTIONS")
    print(f"{'='*70}")

    models_dir = dataset_dir / "models"
    model_results = {}

    for model_meta in summary['models']:
        model_name = model_meta['model_name']
        output_key = model_meta['output_key']
        output_type = model_meta['output_type']

        try:
            # Load model
            model_path = models_dir / model_name

            if output_type == '1D':
                model = ScalarNNModel.load(model_path)
            elif output_type == '2D':
                model = FieldNNModel.load(model_path)
            else:  # 3D
                model = VolumeNNModel.load(model_path)

            # Predict
            Y_pred = model.predict(X_custom)[0]

            model_results[output_key] = {
                'prediction': Y_pred,
                'model_name': model_name,
                'output_type': output_type,
                'model_meta': model_meta
            }

            print(f"\n  {model_name} ({output_type}): ✓")

        except Exception as e:
            print(f"\n  {model_name}: [X] Error: {e}")

    # Compare results
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")

    # Separate 1D scalar results from plottable results
    scalar_results = {}
    plottable_results = {}

    for output_key, model_data in model_results.items():
        npz_key = model_data['model_meta']['npz_key'] if 'npz_key' in model_data['model_meta'] else None

        # Find matching Fluent result
        fluent_data = None
        if npz_key and npz_key in fluent_results:
            fluent_data = fluent_results[npz_key]
        else:
            # Try to find by matching location and field
            location = model_data['model_meta']['location']
            field = model_data['model_meta']['field_name']
            search_key = f"{location}|{field}"
            if search_key in fluent_results:
                fluent_data = fluent_results[search_key]

        if fluent_data is not None:
            Y_true = fluent_data
            Y_pred = model_data['prediction']

            # Check dimension mismatch
            if len(Y_true) != len(Y_pred):
                print(f"\n  {model_data['model_name']}:")
                print(f"    [X] Dimension mismatch!")
                print(f"        Fluent extracted: {len(Y_true)} points")
                print(f"        Model expects: {len(Y_pred)} points")
                print(f"    ⚠ CAUSE: The case file used for training is different from the one being used now.")
                print(f"       The mesh or surface/zone definition has changed.")
                print(f"    ")
                print(f"    SOLUTIONS:")
                print(f"       1. Use the SAME case file that was used during training")
                print(f"          (Check model_setup.json for the training case file path)")
                print(f"       2. OR re-run ALL simulations and retrain models with the current case file")
                print(f"          (Go to: I/O Setup -> Run Simulations, then Train Models)")
                continue

            # Calculate metrics
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            r2 = r2_score(Y_true, Y_pred)
            mae = mean_absolute_error(Y_true, Y_pred)
            rmse = np.sqrt(mean_squared_error(Y_true, Y_pred))

            model_data['fluent'] = Y_true
            model_data['metrics'] = {'r2': r2, 'mae': mae, 'rmse': rmse}

            # Store coordinates from Fluent results if available
            location = model_data['model_meta']['location']
            coord_key = f"{location}|coordinates"
            if coord_key in fluent_results:
                model_data['fluent_coordinates'] = fluent_results[coord_key]

            # Separate 1D scalars from plottable results
            output_type = model_data['output_type']
            if output_type == '1D':
                scalar_results[output_key] = model_data
            else:
                plottable_results[output_key] = model_data

    # Save diagnostic report
    _save_diagnostic_report(dataset_dir, X_custom, scalar_results, plottable_results, setup_data)

    # Display 1D scalar results as text
    if scalar_results:
        print(f"\n{'='*70}")
        print("1D SCALAR RESULTS (No plot needed)")
        print(f"{'='*70}")
        for output_key, model_data in scalar_results.items():
            Y_true = model_data['fluent']
            Y_pred = model_data['prediction']
            metrics = model_data['metrics']

            print(f"\n  {model_data['model_name']}:")
            print(f"    Fluent Value:     {Y_true[0]:.6e}")
            print(f"    Predicted Value:  {Y_pred[0]:.6e}")
            print(f"    Absolute Error:   {abs(Y_pred[0] - Y_true[0]):.6e}")
            print(f"    Relative Error:   {abs(Y_pred[0] - Y_true[0]) / abs(Y_true[0]) * 100:.2f}%")
            print(f"    R² Score:         {metrics['r2']:.6f}")

    # Interactive plot generation menu (loop until user exits)
    if plottable_results:
        while True:
            print(f"\n{'='*70}")
            print("GENERATE COMPARISON PLOTS")
            print(f"{'='*70}")

            # Build list of plottable models
            plot_list = list(plottable_results.items())

            for i, (output_key, model_data) in enumerate(plot_list, 1):
                print(f"  [{i}] {model_data['model_name']} ({model_data['output_type']})")

            print(f"  [A] All")
            print(f"  [0] Done (Exit)")

            choice = input("\nSelect plot(s) to generate: ").strip().upper()

            if choice == '0':
                break
            elif choice == 'A':
                # Plot all
                for output_key, model_data in plottable_results.items():
                    _plot_fluent_vs_model(model_data, dataset_dir)
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(plot_list):
                    output_key, model_data = plot_list[idx]
                    _plot_fluent_vs_model(model_data, dataset_dir)
                else:
                    print("\n  [!] Invalid selection")
            else:
                print("\n  [!] Invalid choice")

    ui_helpers.pause()


def _plot_fluent_vs_model(model_data, dataset_dir):
    """Helper function to plot Fluent vs Model comparison."""
    Y_true = model_data['fluent']
    Y_pred = model_data['prediction']
    model_name = model_data['model_name']
    output_type = model_data['output_type']
    metrics = model_data['metrics']

    # Load coordinates - prioritize Fluent comparison results, fallback to training data
    coordinates = None
    if output_type in ['2D', '3D']:
        # First, try to use coordinates from the Fluent comparison simulation
        if 'fluent_coordinates' in model_data:
            coordinates = model_data['fluent_coordinates']
            print(f"  Using coordinates from Fluent comparison simulation ({len(coordinates)} points)")
        else:
            # Fallback: load from training dataset (this is less accurate but better than nothing)
            location = model_data['model_meta']['location']
            coord_key = f"{location}|coordinates"

            dataset_output_dir = dataset_dir / "dataset"
            output_files = sorted(dataset_output_dir.glob("sim_*.npz"))
            if output_files:
                try:
                    sample_file = np.load(output_files[0], allow_pickle=True)
                    if coord_key in sample_file.files:
                        coordinates = sample_file[coord_key]
                        print(f"  Warning: Using coordinates from training dataset (sim_0001.npz)")
                        print(f"           This may not match current simulation conditions")
                except:
                    pass

    # Use existing plot_comparison function
    plot_comparison(
        Y_true, Y_pred, output_type, model_name,
        metrics['r2'], metrics['mae'], metrics['rmse'],
        coordinates
    )


def _save_diagnostic_report(dataset_dir, X_custom, scalar_results, plottable_results, setup_data):
    """Save detailed diagnostic report for troubleshooting."""
    from datetime import datetime

    report_path = dataset_dir / "models" / "comparison_diagnostic.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SURROGATE MODEL COMPARISON - DIAGNOSTIC REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Case: {dataset_dir.name}\n\n")

        # Input parameters
        f.write("-"*80 + "\n")
        f.write("CUSTOM INPUT PARAMETERS\n")
        f.write("-"*80 + "\n")
        param_names = []
        doe_config = setup_data.get('doe_configuration', {})
        for bc_name, params in doe_config.items():
            for param_name in params.keys():
                param_names.append(f"{bc_name}.{param_name}")

        f.write(f"Number of parameters: {len(X_custom[0])}\n")
        for i, (name, value) in enumerate(zip(param_names, X_custom[0])):
            f.write(f"  {name}: {value:.6f}\n")
            # Show training range
            # Split on first dot to get bc_name, rest is param path
            parts = name.split('.', 1)
            bc_name = parts[0]
            param_path = parts[1] if len(parts) > 1 else name

            # Find the training values for this parameter
            training_values = None
            if bc_name in doe_config:
                for param_key, values in doe_config[bc_name].items():
                    if param_key == param_path or param_key in param_path or param_path in param_key:
                        training_values = values
                        break

            if training_values:
                f.write(f"    Training range: [{min(training_values):.6f}, {max(training_values):.6f}]\n")
                if value < min(training_values):
                    f.write(f"    ⚠ WARNING: Below training range (extrapolation)\n")
                elif value > max(training_values):
                    f.write(f"    ⚠ WARNING: Above training range (extrapolation)\n")
                else:
                    f.write(f"    ✓ Within training range (interpolation)\n")
            else:
                f.write(f"    Training range: Not found\n")

        f.write("\n")

        # 1D Scalar results
        if scalar_results:
            f.write("-"*80 + "\n")
            f.write("1D SCALAR MODEL RESULTS\n")
            f.write("-"*80 + "\n\n")

            for output_key, model_data in scalar_results.items():
                Y_true = model_data['fluent']
                Y_pred = model_data['prediction']
                metrics = model_data['metrics']

                f.write(f"Model: {model_data['model_name']}\n")
                f.write(f"  Output key: {output_key}\n")
                f.write(f"  Fluent value:     {Y_true[0]:.8e}\n")
                f.write(f"  Predicted value:  {Y_pred[0]:.8e}\n")
                f.write(f"  Absolute error:   {abs(Y_pred[0] - Y_true[0]):.8e}\n")
                f.write(f"  Relative error:   {abs(Y_pred[0] - Y_true[0]) / abs(Y_true[0]) * 100:.4f}%\n")
                f.write(f"  R² Score:         {metrics['r2']:.8f}\n")
                f.write("\n")

        # 2D/3D Field results
        if plottable_results:
            f.write("-"*80 + "\n")
            f.write("2D/3D FIELD MODEL RESULTS\n")
            f.write("-"*80 + "\n\n")

            for output_key, model_data in plottable_results.items():
                Y_true = model_data['fluent']
                Y_pred = model_data['prediction']
                metrics = model_data['metrics']
                meta = model_data['model_meta']

                f.write(f"Model: {model_data['model_name']}\n")
                f.write(f"  Output key: {output_key}\n")
                f.write(f"  Output type: {model_data['output_type']}\n")
                f.write(f"  Location: {meta['location']}\n")
                f.write(f"  Field: {meta['field_name']}\n")
                f.write(f"  Number of points: {meta['n_points']}\n")

                if 'n_modes' in meta:
                    f.write(f"  POD modes: {meta['n_modes']}\n")
                if 'variance_explained' in meta:
                    f.write(f"  Variance explained: {meta['variance_explained']*100:.6f}%\n")

                f.write(f"\n  Metrics:\n")
                f.write(f"    R² Score: {metrics['r2']:.8f}\n")
                f.write(f"    MAE:      {metrics['mae']:.8e}\n")
                f.write(f"    RMSE:     {metrics['rmse']:.8e}\n")

                # Statistics on the data
                f.write(f"\n  Fluent Data Statistics:\n")
                f.write(f"    Min:    {Y_true.min():.8e}\n")
                f.write(f"    Max:    {Y_true.max():.8e}\n")
                f.write(f"    Mean:   {Y_true.mean():.8e}\n")
                f.write(f"    Std:    {Y_true.std():.8e}\n")

                f.write(f"\n  Predicted Data Statistics:\n")
                f.write(f"    Min:    {Y_pred.min():.8e}\n")
                f.write(f"    Max:    {Y_pred.max():.8e}\n")
                f.write(f"    Mean:   {Y_pred.mean():.8e}\n")
                f.write(f"    Std:    {Y_pred.std():.8e}\n")

                # Error analysis
                error = Y_pred - Y_true
                f.write(f"\n  Error Statistics:\n")
                f.write(f"    Min error:      {error.min():.8e}\n")
                f.write(f"    Max error:      {error.max():.8e}\n")
                f.write(f"    Mean error:     {error.mean():.8e}\n")
                f.write(f"    Std error:      {error.std():.8e}\n")
                f.write(f"    Max abs error:  {np.abs(error).max():.8e}\n")

                # Check if predictions are reasonable
                fluent_range = Y_true.max() - Y_true.min()
                pred_range = Y_pred.max() - Y_pred.min()
                f.write(f"\n  Range Comparison:\n")
                f.write(f"    Fluent range: {fluent_range:.8e}\n")
                f.write(f"    Predicted range: {pred_range:.8e}\n")
                f.write(f"    Range ratio: {pred_range / fluent_range:.4f}\n")

                if abs(pred_range / fluent_range - 1.0) > 0.5:
                    f.write(f"    ⚠ WARNING: Predicted range differs significantly from Fluent!\n")

                # Sample some point values
                f.write(f"\n  Sample Point Comparison (first 10 points):\n")
                f.write(f"    {'Index':<8} {'Fluent':<15} {'Predicted':<15} {'Error':<15}\n")
                for i in range(min(10, len(Y_true))):
                    f.write(f"    {i:<8} {Y_true[i]:<15.6e} {Y_pred[i]:<15.6e} {error[i]:<15.6e}\n")

                f.write("\n")

        # Training data info
        f.write("-"*80 + "\n")
        f.write("TRAINING DATASET INFORMATION\n")
        f.write("-"*80 + "\n\n")

        # Count simulation files
        dataset_output_dir = dataset_dir / "dataset"
        output_files = list(dataset_output_dir.glob("sim_*.npz"))
        f.write(f"Total simulation files: {len(output_files)}\n")

        # Calculate DOE grid size properly
        grid_sizes = []
        for bc_name, params in doe_config.items():
            for param_name, values in params.items():
                grid_sizes.append(len(values))
        f.write(f"DOE grid size: {' × '.join([str(s) for s in grid_sizes])}\n")

        # Load one sim file to check structure
        if output_files:
            sample_file = np.load(output_files[0], allow_pickle=True)
            f.write(f"\nSample simulation file keys:\n")
            for key in sorted(sample_file.files):
                data_shape = sample_file[key].shape
                f.write(f"  {key}: {data_shape}\n")

        f.write("\n")
        f.write("="*80 + "\n")
        f.write("END OF DIAGNOSTIC REPORT\n")
        f.write("="*80 + "\n")

    print(f"\n  ✓ Diagnostic report saved: {report_path}")
    print(f"    Please review this file to understand the prediction issues")
