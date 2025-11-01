#!/usr/bin/env python
"""
Field Surrogate Training - POD + Neural Network
================================================
Trains a surrogate model using Proper Orthogonal Decomposition (POD)
and Neural Networks to predict field distributions.

Approach:
1. Load NPZ dataset
2. Apply POD to reduce dimensionality (4235 → ~10 modes per field)
3. Train neural network to map parameters → modes
4. Evaluate and visualize predictions vs ground truth
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras


class FieldSurrogate:
    """POD-based field surrogate model."""

    def __init__(self, n_modes=10, field_name='temperature'):
        """
        Initialize surrogate model.

        Parameters
        ----------
        n_modes : int
            Number of POD modes to retain
        field_name : str
            Field variable name (temperature, pressure, velocity_x, etc.)
        """
        self.n_modes = n_modes
        self.field_name = field_name

        # Components
        self.pca = PCA(n_components=n_modes)
        self.param_scaler = StandardScaler()
        self.mode_scaler = StandardScaler()
        self.model = None

        # Metrics storage
        self.train_metrics = {}
        self.test_metrics = {}
        self.variance_explained = None

    def build_model(self, input_dim):
        """Build neural network architecture."""
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dense(self.n_modes)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def fit(self, parameters, fields, validation_split=0.2, epochs=500, verbose=1):
        """
        Train the surrogate model.

        Parameters
        ----------
        parameters : np.ndarray
            Input parameters, shape (n_samples, n_params)
        fields : np.ndarray
            Field data, shape (n_samples, n_points)
        validation_split : float
            Fraction of data for validation
        epochs : int
            Number of training epochs
        verbose : int
            Verbosity level
        """
        print(f"\n{'='*70}")
        print(f"Training {self.field_name} surrogate")
        print(f"{'='*70}")

        # Step 1: Apply POD
        print(f"\n[1/4] Applying POD (reducing {fields.shape[1]} → {self.n_modes} modes)...")
        modes = self.pca.fit_transform(fields)
        self.variance_explained = self.pca.explained_variance_ratio_

        print(f"  Variance explained by {self.n_modes} modes: {self.variance_explained.sum()*100:.2f}%")
        print(f"  Per mode: {(self.variance_explained*100).tolist()}")

        # Step 2: Scale data
        print(f"\n[2/4] Scaling parameters and modes...")
        params_scaled = self.param_scaler.fit_transform(parameters)
        modes_scaled = self.mode_scaler.fit_transform(modes)

        # Step 3: Build and train neural network
        print(f"\n[3/4] Building neural network...")
        self.model = self.build_model(input_dim=parameters.shape[1])
        print(self.model.summary())

        print(f"\n[4/4] Training (epochs={epochs})...")

        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=20,
            min_lr=1e-6
        )

        # Train
        history = self.model.fit(
            params_scaled, modes_scaled,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=8,
            callbacks=[early_stop, reduce_lr],
            verbose=verbose
        )

        print(f"\n✓ Training complete!")
        print(f"  Best val_loss: {min(history.history['val_loss']):.6f}")
        print(f"  Stopped at epoch: {len(history.history['loss'])}")

        return history

    def predict(self, parameters):
        """
        Predict field for given parameters.

        Parameters
        ----------
        parameters : np.ndarray
            Input parameters, shape (n_samples, n_params)

        Returns
        -------
        fields : np.ndarray
            Predicted fields, shape (n_samples, n_points)
        """
        # Scale parameters
        params_scaled = self.param_scaler.transform(parameters)

        # Predict modes
        modes_scaled = self.model.predict(params_scaled, verbose=0)
        modes = self.mode_scaler.inverse_transform(modes_scaled)

        # Reconstruct field from modes
        fields = self.pca.inverse_transform(modes)

        return fields

    def evaluate(self, parameters, true_fields, dataset_name='test'):
        """
        Evaluate model performance.

        Parameters
        ----------
        parameters : np.ndarray
            Input parameters
        true_fields : np.ndarray
            Ground truth fields
        dataset_name : str
            Name for reporting (train/test)

        Returns
        -------
        metrics : dict
            Performance metrics
        """
        pred_fields = self.predict(parameters)

        # Overall metrics
        r2 = r2_score(true_fields.flatten(), pred_fields.flatten())
        rmse = np.sqrt(mean_squared_error(true_fields.flatten(), pred_fields.flatten()))
        mae = mean_absolute_error(true_fields.flatten(), pred_fields.flatten())

        # Per-sample metrics
        r2_per_sample = [
            r2_score(true_fields[i], pred_fields[i])
            for i in range(len(true_fields))
        ]

        metrics = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'r2_per_sample': r2_per_sample,
            'r2_mean': np.mean(r2_per_sample),
            'r2_std': np.std(r2_per_sample),
            'r2_min': np.min(r2_per_sample),
            'r2_max': np.max(r2_per_sample)
        }

        # Store metrics
        if dataset_name == 'train':
            self.train_metrics = metrics
        elif dataset_name == 'test':
            self.test_metrics = metrics

        return metrics

    def print_metrics(self):
        """Print formatted metrics table."""
        print(f"\n{'='*70}")
        print(f"Performance Metrics - {self.field_name}")
        print(f"{'='*70}")

        print(f"\n{'Metric':<20} {'Train':<15} {'Test':<15}")
        print(f"{'-'*50}")

        if self.train_metrics and self.test_metrics:
            print(f"{'R² (overall)':<20} {self.train_metrics['r2']:>14.4f} {self.test_metrics['r2']:>14.4f}")
            print(f"{'R² (mean)':<20} {self.train_metrics['r2_mean']:>14.4f} {self.test_metrics['r2_mean']:>14.4f}")
            print(f"{'R² (std)':<20} {self.train_metrics['r2_std']:>14.4f} {self.test_metrics['r2_std']:>14.4f}")
            print(f"{'R² (min)':<20} {self.train_metrics['r2_min']:>14.4f} {self.test_metrics['r2_min']:>14.4f}")
            print(f"{'R² (max)':<20} {self.train_metrics['r2_max']:>14.4f} {self.test_metrics['r2_max']:>14.4f}")
            print(f"{'RMSE':<20} {self.train_metrics['rmse']:>14.4f} {self.test_metrics['rmse']:>14.4f}")
            print(f"{'MAE':<20} {self.train_metrics['mae']:>14.4f} {self.test_metrics['mae']:>14.4f}")

    def save(self, filepath):
        """Save model to file."""
        save_dict = {
            'n_modes': self.n_modes,
            'field_name': self.field_name,
            'pca_components': self.pca.components_,
            'pca_mean': self.pca.mean_,
            'pca_variance': self.variance_explained,
            'param_scaler_mean': self.param_scaler.mean_,
            'param_scaler_scale': self.param_scaler.scale_,
            'mode_scaler_mean': self.mode_scaler.mean_,
            'mode_scaler_scale': self.mode_scaler.scale_,
        }
        np.savez_compressed(filepath, **save_dict)

        # Save Keras model separately
        model_path = str(filepath).replace('.npz', '_model.h5')
        self.model.save(model_path)

        print(f"Model saved: {filepath}")


def load_dataset(npz_file):
    """Load NPZ dataset."""
    print(f"\nLoading dataset: {npz_file}")
    data = np.load(npz_file, allow_pickle=True)

    print(f"\nDataset contents:")
    for key in data.keys():
        if key != 'metadata':
            print(f"  {key}: {data[key].shape}")

    return data


def train_all_surrogates(dataset_file, n_modes=10, test_size=0.2, epochs=500):
    """
    Train surrogates for all field variables.

    Parameters
    ----------
    dataset_file : str or Path
        Path to NPZ dataset
    n_modes : int
        Number of POD modes
    test_size : float
        Fraction of data for testing
    epochs : int
        Training epochs

    Returns
    -------
    surrogates : dict
        Dictionary of trained surrogates
    """
    # Load data
    data = load_dataset(dataset_file)

    parameters = data['parameters']
    fields = {
        'temperature': data['temperature'],
        'pressure': data['pressure'],
        'velocity_x': data['velocity_x'],
        'velocity_y': data['velocity_y'],
        'velocity_z': data['velocity_z']
    }

    # Split data
    print(f"\n{'='*70}")
    print(f"Splitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
    print(f"{'='*70}")

    indices = np.arange(len(parameters))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=42
    )

    print(f"  Train samples: {len(train_idx)}")
    print(f"  Test samples: {len(test_idx)}")

    # Train surrogates
    surrogates = {}

    for field_name, field_data in fields.items():
        print(f"\n{'='*70}")
        print(f"Training surrogate for: {field_name.upper()}")
        print(f"{'='*70}")

        surrogate = FieldSurrogate(n_modes=n_modes, field_name=field_name)

        # Train
        surrogate.fit(
            parameters[train_idx],
            field_data[train_idx],
            validation_split=0.2,
            epochs=epochs,
            verbose=0
        )

        # Evaluate on train set
        print(f"\nEvaluating on training set...")
        surrogate.evaluate(parameters[train_idx], field_data[train_idx], 'train')

        # Evaluate on test set
        print(f"Evaluating on test set...")
        surrogate.evaluate(parameters[test_idx], field_data[test_idx], 'test')

        # Print metrics
        surrogate.print_metrics()

        surrogates[field_name] = surrogate

    # Save training/test indices
    surrogates['_train_idx'] = train_idx
    surrogates['_test_idx'] = test_idx
    surrogates['_data'] = data

    return surrogates


def visualize_predictions(surrogates, sim_index=None, save_dir=None):
    """
    Create side-by-side comparison plots of ground truth vs predictions.

    Parameters
    ----------
    surrogates : dict
        Dictionary of trained surrogates
    sim_index : int, optional
        Simulation index to visualize. If None, uses random test sample.
    save_dir : Path, optional
        Directory to save plots
    """
    data = surrogates['_data']
    test_idx = surrogates['_test_idx']

    # Select simulation
    if sim_index is None:
        sim_index = np.random.choice(test_idx)

    print(f"\nVisualizing simulation {sim_index}")

    # Get parameters
    params = data['parameters'][sim_index:sim_index+1]
    cold_vel, hot_vel = params[0]
    print(f"  Parameters: Cold={cold_vel:.2f} m/s, Hot={hot_vel:.2f} m/s")

    # Get coordinates
    coords = data['coordinates']
    x = coords[:, 0]
    y = coords[:, 1]

    # Fields to plot
    fields_to_plot = ['temperature', 'pressure', 'velocity_magnitude']

    # Create figure
    fig, axes = plt.subplots(len(fields_to_plot), 2, figsize=(14, 5*len(fields_to_plot)))

    for i, field_name in enumerate(fields_to_plot):
        # Get ground truth
        if field_name == 'velocity_magnitude':
            vx_true = data['velocity_x'][sim_index]
            vy_true = data['velocity_y'][sim_index]
            vz_true = data['velocity_z'][sim_index]
            true_field = np.sqrt(vx_true**2 + vy_true**2 + vz_true**2)

            vx_pred = surrogates['velocity_x'].predict(params)[0]
            vy_pred = surrogates['velocity_y'].predict(params)[0]
            vz_pred = surrogates['velocity_z'].predict(params)[0]
            pred_field = np.sqrt(vx_pred**2 + vy_pred**2 + vz_pred**2)

            label = 'Velocity Magnitude (m/s)'
            cmap = 'plasma'
        else:
            true_field = data[field_name][sim_index]
            pred_field = surrogates[field_name].predict(params)[0]

            if field_name == 'temperature':
                label = 'Temperature (K)'
                cmap = 'hot'
            else:
                label = 'Pressure (Pa)'
                cmap = 'viridis'

        # Calculate error
        r2 = r2_score(true_field, pred_field)
        mae = mean_absolute_error(true_field, pred_field)

        # Ground truth plot
        ax_true = axes[i, 0] if len(fields_to_plot) > 1 else axes[0]
        scatter_true = ax_true.scatter(x, y, c=true_field, cmap=cmap, s=15, alpha=0.8, edgecolors='none')
        ax_true.set_xlabel('X (m)', fontsize=11)
        ax_true.set_ylabel('Y (m)', fontsize=11)
        ax_true.set_title(f'{field_name.replace("_", " ").title()} - Ground Truth', fontsize=12, fontweight='bold')
        ax_true.set_aspect('equal')
        ax_true.grid(True, alpha=0.3)
        cbar_true = plt.colorbar(scatter_true, ax=ax_true)
        cbar_true.set_label(label, fontsize=10)

        # Prediction plot
        ax_pred = axes[i, 1] if len(fields_to_plot) > 1 else axes[1]
        scatter_pred = ax_pred.scatter(x, y, c=pred_field, cmap=cmap, s=15, alpha=0.8, edgecolors='none')
        ax_pred.set_xlabel('X (m)', fontsize=11)
        ax_pred.set_ylabel('Y (m)', fontsize=11)
        ax_pred.set_title(
            f'{field_name.replace("_", " ").title()} - Prediction\n(R²={r2:.4f}, MAE={mae:.4f})',
            fontsize=12, fontweight='bold'
        )
        ax_pred.set_aspect('equal')
        ax_pred.grid(True, alpha=0.3)
        cbar_pred = plt.colorbar(scatter_pred, ax=ax_pred)
        cbar_pred.set_label(label, fontsize=10)

        # Match color scales
        vmin = min(true_field.min(), pred_field.min())
        vmax = max(true_field.max(), pred_field.max())
        scatter_true.set_clim(vmin, vmax)
        scatter_pred.set_clim(vmin, vmax)

    plt.suptitle(f'Ground Truth vs Prediction (Cold={cold_vel:.2f} m/s, Hot={hot_vel:.2f} m/s)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_dir:
        save_path = save_dir / f"comparison_sim{sim_index:03d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.show()


if __name__ == "__main__":
    print("="*70)
    print("FIELD SURROGATE MODEL TRAINING")
    print("="*70)

    # Find available datasets
    project_dir = Path(__file__).parent
    datasets = list(project_dir.glob("*.npz"))

    if not datasets:
        print("\n✗ No dataset files (.npz) found in the project directory!")
        print(f"  Directory: {project_dir}")
        print("\nPlease run runner.py first to generate a dataset.")
        exit(1)

    # Display available datasets
    print("\nAvailable datasets:")
    for i, dataset in enumerate(datasets, 1):
        size_mb = dataset.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {dataset.name} ({size_mb:.2f} MB)")

    # Get user selection
    try:
        selection = int(input(f"\nSelect dataset [1-{len(datasets)}]: ").strip())
        if selection < 1 or selection > len(datasets):
            print("Invalid selection!")
            exit(1)
        DATASET_FILE = datasets[selection - 1]
    except (ValueError, KeyboardInterrupt):
        print("\nCancelled by user.")
        exit(1)

    # Create model name from dataset name
    dataset_name = DATASET_FILE.stem  # e.g., "field_surrogate_dataset"
    model_name = dataset_name.replace("_dataset", "").replace("field_surrogate", "model")
    if model_name == "":
        model_name = "model"

    # Ask for custom model name
    print(f"\nDefault model name: {model_name}")
    custom_name = input("Enter custom model name (or press Enter to use default): ").strip()
    if custom_name:
        model_name = custom_name

    # Create model-specific directory
    OUTPUT_DIR = project_dir / "surrogate_models" / model_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy dataset to model folder
    import shutil
    dataset_copy = OUTPUT_DIR / DATASET_FILE.name
    if not dataset_copy.exists():
        print(f"\nCopying dataset to model folder...")
        shutil.copy2(DATASET_FILE, dataset_copy)
        print(f"  Copied: {dataset_copy}")

    print(f"\n{'='*70}")
    print(f"Training Configuration")
    print(f"{'='*70}")
    print(f"  Dataset: {DATASET_FILE.name}")
    print(f"  Model name: {model_name}")
    print(f"  Output directory: {OUTPUT_DIR}")

    N_MODES = 10
    TEST_SIZE = 0.2
    EPOCHS = 500

    print("="*70)
    print("FIELD SURROGATE TRAINING - POD + NEURAL NETWORK")
    print("="*70)

    # Train surrogates
    surrogates = train_all_surrogates(
        dataset_file=DATASET_FILE,
        n_modes=N_MODES,
        test_size=TEST_SIZE,
        epochs=EPOCHS
    )

    # Save models
    print(f"\n{'='*70}")
    print("Saving models...")
    print(f"{'='*70}")

    for field_name, surrogate in surrogates.items():
        if not field_name.startswith('_'):
            save_path = OUTPUT_DIR / f"surrogate_{field_name}.npz"
            surrogate.save(save_path)

    # Visualize predictions
    print(f"\n{'='*70}")
    print("Generating visualizations...")
    print(f"{'='*70}")

    # Random test sample
    visualize_predictions(surrogates, save_dir=OUTPUT_DIR)

    # Best and worst predictions
    test_idx = surrogates['_test_idx']
    temp_surrogate = surrogates['temperature']

    best_idx = test_idx[np.argmax(temp_surrogate.test_metrics['r2_per_sample'])]
    worst_idx = test_idx[np.argmin(temp_surrogate.test_metrics['r2_per_sample'])]

    print(f"\nBest prediction (highest R²):")
    visualize_predictions(surrogates, sim_index=best_idx, save_dir=OUTPUT_DIR)

    print(f"\nWorst prediction (lowest R²):")
    visualize_predictions(surrogates, sim_index=worst_idx, save_dir=OUTPUT_DIR)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"\nModels saved to: {OUTPUT_DIR}")
    
