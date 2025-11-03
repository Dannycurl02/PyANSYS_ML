#!/usr/bin/env python
"""
Surrogate Model Inference
=========================
Use trained surrogate models to make fast predictions for new parameter combinations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_surrogate(model_dir, field_name):
    """
    Load a trained surrogate model.

    Parameters
    ----------
    model_dir : Path
        Directory containing saved models
    field_name : str
        Field variable name

    Returns
    -------
    surrogate_dict : dict
        Dictionary with model components
    """
    # Load POD and scaler components
    npz_path = model_dir / f"surrogate_{field_name}.npz"
    data = np.load(npz_path)

    # Load Keras model (compile=False to avoid metric deserialization issues)
    model_path = model_dir / f"surrogate_{field_name}_model.h5"
    nn_model = keras.models.load_model(model_path, compile=False)

    # Reconstruct PCA
    pca = PCA(n_components=len(data['pca_variance']))
    pca.components_ = data['pca_components']
    pca.mean_ = data['pca_mean']
    pca.explained_variance_ratio_ = data['pca_variance']

    # Reconstruct scalers
    param_scaler = StandardScaler()
    param_scaler.mean_ = data['param_scaler_mean']
    param_scaler.scale_ = data['param_scaler_scale']

    mode_scaler = StandardScaler()
    mode_scaler.mean_ = data['mode_scaler_mean']
    mode_scaler.scale_ = data['mode_scaler_scale']

    return {
        'model': nn_model,
        'pca': pca,
        'param_scaler': param_scaler,
        'mode_scaler': mode_scaler,
        'field_name': field_name
    }


def predict_field(surrogate_dict, cold_vel, hot_vel):
    """
    Predict field for given parameters.

    Parameters
    ----------
    surrogate_dict : dict
        Loaded surrogate model
    cold_vel : float
        Cold inlet velocity (m/s)
    hot_vel : float
        Hot inlet velocity (m/s)

    Returns
    -------
    field : np.ndarray
        Predicted field values
    """
    # Prepare parameters
    params = np.array([[cold_vel, hot_vel]])

    # Scale parameters
    params_scaled = surrogate_dict['param_scaler'].transform(params)

    # Predict modes
    modes_scaled = surrogate_dict['model'].predict(params_scaled, verbose=0)
    modes = surrogate_dict['mode_scaler'].inverse_transform(modes_scaled)

    # Reconstruct field
    field = surrogate_dict['pca'].inverse_transform(modes)[0]

    return field


def predict_all_fields(model_dir, cold_vel, hot_vel, target_coordinates):
    """
    Predict all fields for given parameters and interpolate to target coordinates.

    Parameters
    ----------
    model_dir : Path
        Directory with saved models
    cold_vel : float
        Cold inlet velocity
    hot_vel : float
        Hot inlet velocity
    target_coordinates : np.ndarray
        Target node coordinates (n_points, 3) for interpolation

    Returns
    -------
    predictions : dict
        Dictionary of predicted fields interpolated to target coordinates
    """
    from scipy.interpolate import LinearNDInterpolator

    # Load the dataset to get training coordinates
    dataset_files = list(model_dir.glob("*.npz"))
    if dataset_files:
        # Find the main dataset file (not surrogate_*.npz)
        main_dataset = [f for f in dataset_files if not f.name.startswith('surrogate_')]
        if main_dataset:
            data = np.load(main_dataset[0], allow_pickle=True)
            training_coords = data['coordinates']
        else:
            raise ValueError("No main dataset file found in model directory")
    else:
        raise ValueError("No dataset files found in model directory")

    fields = ['temperature', 'pressure', 'velocity_x', 'velocity_y', 'velocity_z']

    predictions = {}
    for field_name in fields:
        surrogate = load_surrogate(model_dir, field_name)
        # Get predictions on training coordinates
        field_values = predict_field(surrogate, cold_vel, hot_vel)

        # Interpolate to target coordinates
        interpolator = LinearNDInterpolator(training_coords, field_values)
        predictions[field_name] = interpolator(target_coordinates)

    # Calculate velocity magnitude
    vx = predictions['velocity_x']
    vy = predictions['velocity_y']
    vz = predictions['velocity_z']
    predictions['velocity_magnitude'] = np.sqrt(vx**2 + vy**2 + vz**2)

    return predictions


def visualize_prediction(predictions, coordinates, cold_vel, hot_vel, save_path=None):
    """
    Visualize predicted fields.

    Parameters
    ----------
    predictions : dict
        Predicted fields
    coordinates : np.ndarray
        Node coordinates
    cold_vel : float
        Cold inlet velocity
    hot_vel : float
        Hot inlet velocity
    save_path : Path, optional
        Path to save figure
    """
    x = coordinates[:, 0]
    y = coordinates[:, 1]

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # Temperature
    scatter1 = axes[0].scatter(x, y, c=predictions['temperature'],
                               cmap='hot', s=15, alpha=0.8, edgecolors='none')
    axes[0].set_xlabel('X (m)', fontsize=12)
    axes[0].set_ylabel('Y (m)', fontsize=12)
    axes[0].set_title(f'Predicted Temperature (Cold={cold_vel:.2f} m/s, Hot={hot_vel:.2f} m/s)',
                      fontsize=14, fontweight='bold')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Temperature (K)', fontsize=10)

    # Pressure
    scatter2 = axes[1].scatter(x, y, c=predictions['pressure'],
                               cmap='viridis', s=15, alpha=0.8, edgecolors='none')
    axes[1].set_xlabel('X (m)', fontsize=12)
    axes[1].set_ylabel('Y (m)', fontsize=12)
    axes[1].set_title(f'Predicted Pressure (Cold={cold_vel:.2f} m/s, Hot={hot_vel:.2f} m/s)',
                      fontsize=14, fontweight='bold')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Pressure (Pa)', fontsize=10)

    # Velocity magnitude
    scatter3 = axes[2].scatter(x, y, c=predictions['velocity_magnitude'],
                               cmap='plasma', s=15, alpha=0.8, edgecolors='none')
    axes[2].set_xlabel('X (m)', fontsize=12)
    axes[2].set_ylabel('Y (m)', fontsize=12)
    axes[2].set_title(f'Predicted Velocity Magnitude (Cold={cold_vel:.2f} m/s, Hot={hot_vel:.2f} m/s)',
                      fontsize=14, fontweight='bold')
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)
    cbar3 = plt.colorbar(scatter3, ax=axes[2])
    cbar3.set_label('Velocity Magnitude (m/s)', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.show()


def run_fluent_simulation(cold_vel, hot_vel, case_file, plane_name, iterations=200):
    """
    Run a single Fluent simulation for comparison.

    Parameters
    ----------
    cold_vel : float
        Cold inlet velocity
    hot_vel : float
        Hot inlet velocity
    case_file : Path
        Fluent case file
    plane_name : str
        Surface name
    iterations : int
        Number of iterations

    Returns
    -------
    fluent_results : dict
        Dictionary with field data from Fluent
    """
    import ansys.fluent.core as pyfluent
    import tempfile
    import subprocess
    import sys

    print(f"\n  Launching Fluent for ground truth simulation...")

    # Create temp log file for Fluent output
    fluent_log_file = tempfile.NamedTemporaryFile(
        mode='w+', suffix='.log', prefix='fluent_validation_', delete=False, buffering=1
    )
    fluent_log_path = fluent_log_file.name

    # Write header to Fluent log
    from datetime import datetime
    fluent_log_file.write("="*70 + "\n")
    fluent_log_file.write("FLUENT VALIDATION SIMULATION OUTPUT\n")
    fluent_log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    fluent_log_file.write("="*70 + "\n")
    fluent_log_file.write(f"Validation Case:\n")
    fluent_log_file.write(f"  Cold inlet velocity: {cold_vel:.3f} m/s\n")
    fluent_log_file.write(f"  Hot inlet velocity: {hot_vel:.3f} m/s\n")
    fluent_log_file.write(f"  Iterations: {iterations}\n")
    fluent_log_file.write("="*70 + "\n\n")
    fluent_log_file.flush()

    # Launch PowerShell window to tail the log
    ps_cmd = f'Get-Content -Path "{fluent_log_path}" -Wait'
    subprocess.Popen(['powershell', '-NoExit', '-Command', ps_cmd],
                     creationflags=subprocess.CREATE_NEW_CONSOLE)

    print(f"  Fluent output will appear in separate window...")

    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Launch Fluent
    solver = pyfluent.launch_fluent(
        precision="single",
        processor_count=6,
        dimension=pyfluent.Dimension.THREE,
        ui_mode=pyfluent.UIMode.HIDDEN_GUI
    )

    # Load case
    print(f"  Reading case file...")
    sys.stdout = fluent_log_file
    sys.stderr = fluent_log_file
    solver.settings.file.read_case(file_name=str(case_file))
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    # Set boundary conditions (using newer syntax)
    print(f"  Setting BCs: Cold={cold_vel:.3f} m/s, Hot={hot_vel:.3f} m/s")
    sys.stdout = fluent_log_file
    sys.stderr = fluent_log_file
    solver.settings.setup.boundary_conditions.velocity_inlet["cold-inlet"].momentum.velocity_magnitude.value = float(cold_vel)
    solver.settings.setup.boundary_conditions.velocity_inlet["hot-inlet"].momentum.velocity_magnitude.value = float(hot_vel)
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    # Solve
    print(f"  Solving ({iterations} iterations)...")
    print(f"  (Detailed iteration info in separate window)")
    sys.stdout = fluent_log_file
    sys.stderr = fluent_log_file
    solver.settings.solution.initialization.initialization_type = "standard"
    solver.settings.solution.initialization.standard_initialize()
    solver.settings.solution.run_calculation.iterate(iter_count=iterations)
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    # Extract field data
    print(f"  Extracting field data from '{plane_name}'...")
    fd = solver.fields.field_data

    # Using get_scalar_field_data (deprecated but functional)
    temp_dict = fd.get_scalar_field_data(field_name='temperature', surfaces=[plane_name], node_value=True)
    press_dict = fd.get_scalar_field_data(field_name='absolute-pressure', surfaces=[plane_name], node_value=True)
    vx_dict = fd.get_scalar_field_data(field_name='x-velocity', surfaces=[plane_name], node_value=True)
    vy_dict = fd.get_scalar_field_data(field_name='y-velocity', surfaces=[plane_name], node_value=True)
    vz_dict = fd.get_scalar_field_data(field_name='z-velocity', surfaces=[plane_name], node_value=True)

    # Get coordinate data
    x_dict = fd.get_scalar_field_data(field_name='x-coordinate', surfaces=[plane_name], node_value=True)
    y_dict = fd.get_scalar_field_data(field_name='y-coordinate', surfaces=[plane_name], node_value=True)
    z_dict = fd.get_scalar_field_data(field_name='z-coordinate', surfaces=[plane_name], node_value=True)

    # Extract data - check if it's directly an array or has 'scalar-field' key
    def extract_field_data(field_dict, surface_name):
        """Extract field data from the returned dictionary."""
        surface_data = field_dict[surface_name]
        if isinstance(surface_data, dict) and 'scalar-field' in surface_data:
            return np.array(surface_data['scalar-field'])
        else:
            # Data is already an array
            return np.array(surface_data)

    results = {
        'temperature': extract_field_data(temp_dict, plane_name),
        'pressure': extract_field_data(press_dict, plane_name),
        'velocity_x': extract_field_data(vx_dict, plane_name),
        'velocity_y': extract_field_data(vy_dict, plane_name),
        'velocity_z': extract_field_data(vz_dict, plane_name),
        'coordinates': np.column_stack([
            extract_field_data(x_dict, plane_name),
            extract_field_data(y_dict, plane_name),
            extract_field_data(z_dict, plane_name)
        ])
    }

    # Calculate velocity magnitude
    results['velocity_magnitude'] = np.sqrt(
        results['velocity_x']**2 + results['velocity_y']**2 + results['velocity_z']**2
    )

    # Close Fluent
    print(f"  Closing Fluent...")
    sys.stdout = fluent_log_file
    sys.stderr = fluent_log_file
    solver.exit()
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    # Write completion message to log
    fluent_log_file.write("\n" + "="*70 + "\n")
    fluent_log_file.write("VALIDATION SIMULATION COMPLETE\n")
    fluent_log_file.write("="*70 + "\n")
    fluent_log_file.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    fluent_log_file.write("\nFluent session closed\n")
    fluent_log_file.flush()

    # Close log file
    fluent_log_file.close()

    return results


def create_comparison_plot(fluent_results, predictions, coordinates, cold_vel, hot_vel, save_path=None):
    """
    Create 3x3 comparison plot: Fluent | Prediction | Error.

    Parameters
    ----------
    fluent_results : dict
        Ground truth from Fluent
    predictions : dict
        Surrogate predictions
    coordinates : np.ndarray
        Node coordinates
    cold_vel : float
        Cold inlet velocity
    hot_vel : float
        Hot inlet velocity
    save_path : Path, optional
        Path to save figure
    """
    from sklearn.metrics import r2_score, mean_absolute_error
    from matplotlib.colors import TwoSlopeNorm

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    # Fields to plot
    fields = ['temperature', 'pressure', 'velocity_magnitude']
    labels = ['Temperature (K)', 'Pressure (Pa)', 'Velocity Magnitude (m/s)']
    cmaps = ['hot', 'viridis', 'plasma']

    # Create 3x3 subplot
    fig, axes = plt.subplots(3, 3, figsize=(18, 21))

    for row, (field, label, cmap) in enumerate(zip(fields, labels, cmaps)):
        # Get data
        fluent_data = fluent_results[field]
        pred_data = predictions[field]
        error = pred_data - fluent_data
        error_percent = 100 * np.abs(error) / (fluent_data + 1e-10)

        # Calculate metrics
        r2 = r2_score(fluent_data, pred_data)
        mae = mean_absolute_error(fluent_data, pred_data)
        rmse = np.sqrt(np.mean(error**2))
        max_error = np.abs(error).max()

        # Common color scale for Fluent and Prediction
        vmin = min(fluent_data.min(), pred_data.min())
        vmax = max(fluent_data.max(), pred_data.max())

        # Column 1: Fluent (Ground Truth)
        ax1 = axes[row, 0]
        scatter1 = ax1.scatter(x, y, c=fluent_data, cmap=cmap, s=12, alpha=0.8,
                               edgecolors='none', vmin=vmin, vmax=vmax)
        ax1.set_xlabel('X (m)', fontsize=10)
        ax1.set_ylabel('Y (m)', fontsize=10)
        ax1.set_title(f'Fluent - {field.replace("_", " ").title()}', fontsize=11, fontweight='bold')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label(label, fontsize=9)

        # Column 2: Prediction
        ax2 = axes[row, 1]
        scatter2 = ax2.scatter(x, y, c=pred_data, cmap=cmap, s=12, alpha=0.8,
                               edgecolors='none', vmin=vmin, vmax=vmax)
        ax2.set_xlabel('X (m)', fontsize=10)
        ax2.set_ylabel('Y (m)', fontsize=10)
        ax2.set_title(f'Surrogate - {field.replace("_", " ").title()}\nR²={r2:.4f}',
                      fontsize=11, fontweight='bold')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label(label, fontsize=9)

        # Column 3: Error (centered at zero: white=0, blue=positive, red=negative)
        ax3 = axes[row, 2]

        # Create diverging colormap centered at zero
        error_max = max(abs(error.min()), abs(error.max()))
        norm = TwoSlopeNorm(vmin=-error_max, vcenter=0, vmax=error_max)

        scatter3 = ax3.scatter(x, y, c=error, cmap='RdBu_r', s=12, alpha=0.8,
                               edgecolors='none', norm=norm)
        ax3.set_xlabel('X (m)', fontsize=10)
        ax3.set_ylabel('Y (m)', fontsize=10)
        ax3.set_title(f'Error (Pred - Fluent)\nMAE={mae:.4f}, Max={max_error:.4f}',
                      fontsize=11, fontweight='bold')
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        cbar3 = plt.colorbar(scatter3, ax=ax3)
        cbar3.set_label(f'Error ({label.split("(")[1].split(")")[0]})', fontsize=9)

    plt.suptitle(f'Fluent vs Surrogate Comparison (Cold={cold_vel:.3f} m/s, Hot={hot_vel:.3f} m/s)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  Comparison plot saved: {save_path}")

    plt.show()


if __name__ == "__main__":
    print("="*70)
    print("SURROGATE MODEL VALIDATION")
    print("="*70)

    # Configuration
    project_dir = Path(__file__).parent.parent  # Go up from modules/ to project root
    CASE_FILE = project_dir / "elbow.cas.h5"
    PLANE_NAME = "mid-plane"
    ITERATIONS = 200

    # Find dataset folders containing trained models
    # Look for folders that have both dataset .npz and trained model files
    model_dirs = []
    for item in project_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name != 'modules':
            # Check if it has model files
            if list(item.glob("surrogate_*.npz")) and list(item.glob("surrogate_*.h5")):
                model_dirs.append(item)

    if not model_dirs:
        print("\n✗ No trained models found!")
        print(f"  Directory: {project_dir}")
        print("\nPlease run train_surrogate.py first to train a model.")
        exit(1)

    # Display available models
    print("\nAvailable trained models:")
    for i, model_dir in enumerate(model_dirs, 1):
        # Get dataset file in this model folder
        dataset_files = list(model_dir.glob("*.npz"))
        dataset_info = dataset_files[0].name if dataset_files else "No dataset"
        model_files = list(model_dir.glob("surrogate_*.h5"))
        print(f"  [{i}] {model_dir.name} ({len(model_files)} fields, {dataset_info})")

    # Get user selection
    try:
        selection = int(input(f"\nSelect model [1-{len(model_dirs)}]: ").strip())
        if selection < 1 or selection > len(model_dirs):
            print("Invalid selection!")
            exit(1)
        MODEL_DIR = model_dirs[selection - 1]
    except (ValueError, KeyboardInterrupt):
        print("\nCancelled by user.")
        exit(1)

    # Find dataset in model directory
    dataset_files = list(MODEL_DIR.glob("*.npz"))
    if not dataset_files:
        print(f"\n✗ No dataset found in model directory: {MODEL_DIR}")
        exit(1)
    DATASET_FILE = dataset_files[0]

    # Create validation output directory
    VALIDATION_DIR = MODEL_DIR / "validation"
    VALIDATION_DIR.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Validation Configuration")
    print(f"{'='*70}")
    print(f"  Model: {MODEL_DIR.name}")
    print(f"  Dataset: {DATASET_FILE.name}")
    print(f"  Output: {VALIDATION_DIR}")

    # Load coordinates from dataset
    print(f"\n{'='*70}")
    print(f"Loading dataset...")
    print(f"{'='*70}")
    data = np.load(DATASET_FILE)
    coordinates = data['coordinates']
    print(f"  Coordinates: {coordinates.shape}")

    # Get user input
    print(f"\n{'='*70}")
    print("Enter Parameters for Validation")
    print(f"{'='*70}")

    try:
        cold_vel = float(input("\nEnter cold inlet velocity (m/s) [0.1-0.7]: ").strip())
        hot_vel = float(input("Enter hot inlet velocity (m/s) [0.8-2.0]: ").strip())

        print(f"\n{'='*70}")
        print(f"Validation Case: Cold={cold_vel:.3f} m/s, Hot={hot_vel:.3f} m/s")
        print(f"{'='*70}")

        # Step 1: Run Fluent simulation to get ground truth coordinates
        print(f"\n[1/2] Running Fluent Simulation for Ground Truth...")
        fluent_results = run_fluent_simulation(
            cold_vel=cold_vel,
            hot_vel=hot_vel,
            case_file=CASE_FILE,
            plane_name=PLANE_NAME,
            iterations=ITERATIONS
        )
        print(f"  ✓ Fluent simulation complete")
        print(f"    Temperature: {fluent_results['temperature'].min():.2f} - {fluent_results['temperature'].max():.2f} K")
        print(f"    Pressure: {fluent_results['pressure'].min():.2f} - {fluent_results['pressure'].max():.2f} Pa")
        print(f"    Velocity: {fluent_results['velocity_magnitude'].min():.4f} - {fluent_results['velocity_magnitude'].max():.4f} m/s")
        print(f"    Coordinates: {fluent_results['coordinates'].shape}")

        # Step 2: Get surrogate predictions using Fluent's coordinates
        print(f"\n[2/2] Running Surrogate Prediction...")
        fluent_coords = fluent_results['coordinates']
        predictions = predict_all_fields(MODEL_DIR, cold_vel, hot_vel, fluent_coords)
        print(f"  ✓ Surrogate prediction complete")
        print(f"    Temperature: {predictions['temperature'].min():.2f} - {predictions['temperature'].max():.2f} K")
        print(f"    Pressure: {predictions['pressure'].min():.2f} - {predictions['pressure'].max():.2f} Pa")
        print(f"    Velocity: {predictions['velocity_magnitude'].min():.4f} - {predictions['velocity_magnitude'].max():.4f} m/s")

        # Step 3: Create comparison plot
        print(f"\n{'='*70}")
        print("Creating Comparison Plots...")
        print(f"{'='*70}")

        save_path = VALIDATION_DIR / f"validation_cold{cold_vel:.2f}_hot{hot_vel:.2f}.png"
        create_comparison_plot(
            fluent_results=fluent_results,
            predictions=predictions,
            coordinates=fluent_coords,
            cold_vel=cold_vel,
            hot_vel=hot_vel,
            save_path=save_path
        )

        print(f"\n{'='*70}")
        print("VALIDATION COMPLETED")
        print(f"{'='*70}")

    except (ValueError, KeyboardInterrupt) as e:
        print(f"\n✗ Error or cancelled: {e}")
        print("Exiting...")
