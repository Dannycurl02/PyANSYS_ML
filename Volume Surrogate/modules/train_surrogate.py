#!/usr/bin/env python
"""
Volume Surrogate Training - POD + Neural Network
=================================================
Trains a surrogate model using Proper Orthogonal Decomposition (POD)
and Neural Networks to predict 3D volume field distributions.

Approach:
1. Load NPZ dataset (3D volume with ~17,865 cells)
2. Apply POD to reduce dimensionality (17865 → ~20 modes per field)
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
    """POD-based 3D volume surrogate model."""

    def __init__(self, n_modes=20, field_name='temperature'):
        """
        Initialize surrogate model for 3D volume data.

        Parameters
        ----------
        n_modes : int
            Number of POD modes to retain (default 20 for 3D volume)
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
        """Build neural network architecture optimized for 3D volume data."""
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.15),
            keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.15),
            keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
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

        # Store history for later visualization
        self.history = history

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

    # Adjust n_modes if dataset is too small
    n_train = len(train_idx)
    n_features = fields['temperature'].shape[1]
    max_modes = min(n_train, n_features)

    if n_modes > max_modes:
        print(f"\n⚠️  WARNING: Requested {n_modes} POD modes but only {n_train} training samples available.")
        print(f"   Automatically reducing to {max_modes} modes (max possible with this dataset).")
        n_modes = max_modes

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


def plot_training_curves(surrogates, save_dir=None):
    """
    Plot training curves (loss and MAE) for all fields.
    Creates separate plots for temperature, pressure, and velocity (combined).

    Parameters
    ----------
    surrogates : dict
        Dictionary of trained surrogates
    save_dir : Path, optional
        Directory to save plots
    """
    fields = [k for k in surrogates.keys() if not k.startswith('_')]

    # Group fields: temperature, pressure, velocity (x, y, z combined)
    field_groups = {
        'temperature': ['temperature'],
        'pressure': ['pressure'],
        'velocity': ['velocity_x', 'velocity_y', 'velocity_z']
    }

    colors = {'temperature': 'firebrick', 'pressure': 'steelblue',
              'velocity_x': 'forestgreen', 'velocity_y': 'darkorange', 'velocity_z': 'purple'}

    # Create separate plots for each group
    for group_name, field_list in field_groups.items():
        # Check if any field in this group exists
        if not any(f in fields for f in field_list):
            continue

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot loss curves
        ax_loss = axes[0]
        for field_name in field_list:
            if field_name not in fields:
                continue
            surrogate = surrogates[field_name]
            if hasattr(surrogate, 'history'):
                history = surrogate.history.history
                epochs = range(1, len(history['loss']) + 1)

                color = colors.get(field_name, 'black')
                label = field_name.replace('_', ' ').title()

                ax_loss.plot(epochs, history['loss'],
                            color=color, linestyle='-', linewidth=2.5,
                            label=f'{label} (train)', alpha=0.7)
                ax_loss.plot(epochs, history['val_loss'],
                            color=color, linestyle='--', linewidth=2.5,
                            label=f'{label} (val)', alpha=0.9)

        ax_loss.set_xlabel('Epoch', fontsize=13)
        ax_loss.set_ylabel('Loss (MSE)', fontsize=13)
        title = group_name.replace('_', ' ').title()
        ax_loss.set_title(f'{title} - Training and Validation Loss', fontsize=15, fontweight='bold')
        ax_loss.legend(loc='upper right', fontsize=10)
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_yscale('log')

        # Plot MAE curves
        ax_mae = axes[1]
        for field_name in field_list:
            if field_name not in fields:
                continue
            surrogate = surrogates[field_name]
            if hasattr(surrogate, 'history'):
                history = surrogate.history.history
                epochs = range(1, len(history['mae']) + 1)

                color = colors.get(field_name, 'black')
                label = field_name.replace('_', ' ').title()

                ax_mae.plot(epochs, history['mae'],
                           color=color, linestyle='-', linewidth=2.5,
                           label=f'{label} (train)', alpha=0.7)
                ax_mae.plot(epochs, history['val_mae'],
                           color=color, linestyle='--', linewidth=2.5,
                           label=f'{label} (val)', alpha=0.9)

        ax_mae.set_xlabel('Epoch', fontsize=13)
        ax_mae.set_ylabel('MAE', fontsize=13)
        ax_mae.set_title(f'{title} - Training and Validation MAE', fontsize=15, fontweight='bold')
        ax_mae.legend(loc='upper right', fontsize=10)
        ax_mae.grid(True, alpha=0.3)
        ax_mae.set_yscale('log')

        plt.tight_layout()

        if save_dir:
            save_path = save_dir / f"training_curves_{group_name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ {title} training curves saved: {save_path}")

        plt.show()


def save_performance_report(surrogates, save_dir=None, dataset_file=None):
    """
    Save comprehensive performance report to text file.

    Parameters
    ----------
    surrogates : dict
        Dictionary of trained surrogates
    save_dir : Path, optional
        Directory to save report
    dataset_file : Path, optional
        Path to dataset file
    """
    if save_dir is None:
        return

    report_path = save_dir / "performance_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("="*80 + "\n")
        f.write("FIELD SURROGATE MODEL - PERFORMANCE REPORT\n")
        f.write("="*80 + "\n\n")

        # Dataset info
        from datetime import datetime
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if dataset_file:
            f.write(f"Dataset: {dataset_file.name}\n")
            data = surrogates['_data']
            n_sims = data['parameters'].shape[0]
            n_nodes = data['coordinates'].shape[0]
            f.write(f"Total simulations: {n_sims}\n")
            f.write(f"Mesh nodes: {n_nodes}\n")
        f.write("\n")

        # Training configuration
        f.write("-"*80 + "\n")
        f.write("TRAINING CONFIGURATION\n")
        f.write("-"*80 + "\n")
        train_idx = surrogates['_train_idx']
        test_idx = surrogates['_test_idx']
        f.write(f"Train samples: {len(train_idx)}\n")
        f.write(f"Test samples: {len(test_idx)}\n")
        f.write(f"Train/Test split: {len(train_idx)/(len(train_idx)+len(test_idx))*100:.1f}% / {len(test_idx)/(len(train_idx)+len(test_idx))*100:.1f}%\n")

        # Get POD info from first surrogate
        fields = [k for k in surrogates.keys() if not k.startswith('_')]
        first_surrogate = surrogates[fields[0]]
        f.write(f"POD modes: {first_surrogate.n_modes}\n")
        if hasattr(first_surrogate, 'history'):
            f.write(f"Training epochs: {len(first_surrogate.history.history['loss'])}\n")
        f.write("\n")

        # POD variance explained
        f.write("-"*80 + "\n")
        f.write("POD VARIANCE EXPLAINED\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Field':<20} {'Variance Explained':<20} {'Cumulative':<20}\n")
        f.write("-"*80 + "\n")
        for field_name in fields:
            surrogate = surrogates[field_name]
            if surrogate.variance_explained is not None:
                var_explained = surrogate.variance_explained
                cumsum = var_explained.sum()
                f.write(f"{field_name:<20} {var_explained[-1]*100:>18.4f}% {cumsum*100:>18.4f}%\n")
        f.write("\n")

        # Overall performance metrics
        f.write("-"*80 + "\n")
        f.write("OVERALL PERFORMANCE METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Field':<20} {'Dataset':<10} {'R²':<12} {'RMSE':<15} {'MAE':<15}\n")
        f.write("-"*80 + "\n")
        for field_name in fields:
            surrogate = surrogates[field_name]
            # Train metrics
            f.write(f"{field_name:<20} {'Train':<10} "
                   f"{surrogate.train_metrics['r2']:>11.6f} "
                   f"{surrogate.train_metrics['rmse']:>14.6f} "
                   f"{surrogate.train_metrics['mae']:>14.6f}\n")
            # Test metrics
            f.write(f"{'':<20} {'Test':<10} "
                   f"{surrogate.test_metrics['r2']:>11.6f} "
                   f"{surrogate.test_metrics['rmse']:>14.6f} "
                   f"{surrogate.test_metrics['mae']:>14.6f}\n")
            f.write("\n")

        # Per-sample R² statistics
        f.write("-"*80 + "\n")
        f.write("PER-SAMPLE R² STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Field':<20} {'Dataset':<10} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}\n")
        f.write("-"*80 + "\n")
        for field_name in fields:
            surrogate = surrogates[field_name]
            # Train stats
            f.write(f"{field_name:<20} {'Train':<10} "
                   f"{surrogate.train_metrics['r2_mean']:>11.6f} "
                   f"{surrogate.train_metrics['r2_std']:>11.6f} "
                   f"{surrogate.train_metrics['r2_min']:>11.6f} "
                   f"{surrogate.train_metrics['r2_max']:>11.6f}\n")
            # Test stats
            f.write(f"{'':<20} {'Test':<10} "
                   f"{surrogate.test_metrics['r2_mean']:>11.6f} "
                   f"{surrogate.test_metrics['r2_std']:>11.6f} "
                   f"{surrogate.test_metrics['r2_min']:>11.6f} "
                   f"{surrogate.test_metrics['r2_max']:>11.6f}\n")
            f.write("\n")

        # Training history summary
        f.write("-"*80 + "\n")
        f.write("TRAINING HISTORY SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Field':<20} {'Final Loss':<15} {'Final Val Loss':<15} {'Best Val Loss':<15}\n")
        f.write("-"*80 + "\n")
        for field_name in fields:
            surrogate = surrogates[field_name]
            if hasattr(surrogate, 'history'):
                history = surrogate.history.history
                final_loss = history['loss'][-1]
                final_val_loss = history['val_loss'][-1]
                best_val_loss = min(history['val_loss'])
                f.write(f"{field_name:<20} {final_loss:>14.6e} {final_val_loss:>14.6e} {best_val_loss:>14.6e}\n")
        f.write("\n")

        # Best and worst predictions
        f.write("-"*80 + "\n")
        f.write("BEST AND WORST PREDICTIONS (based on temperature field)\n")
        f.write("-"*80 + "\n")
        temp_surrogate = surrogates['temperature']
        r2_per_sample = temp_surrogate.test_metrics['r2_per_sample']
        best_idx_local = np.argmax(r2_per_sample)
        worst_idx_local = np.argmin(r2_per_sample)
        best_global_idx = test_idx[best_idx_local]
        worst_global_idx = test_idx[worst_idx_local]

        data = surrogates['_data']
        best_params = data['parameters'][best_global_idx]
        worst_params = data['parameters'][worst_global_idx]

        f.write(f"Best prediction:\n")
        f.write(f"  Simulation index: {best_global_idx}\n")
        f.write(f"  Parameters: Cold={best_params[0]:.3f} m/s, Hot={best_params[1]:.3f} m/s\n")
        f.write(f"  Temperature R²: {r2_per_sample[best_idx_local]:.6f}\n\n")

        f.write(f"Worst prediction:\n")
        f.write(f"  Simulation index: {worst_global_idx}\n")
        f.write(f"  Parameters: Cold={worst_params[0]:.3f} m/s, Hot={worst_params[1]:.3f} m/s\n")
        f.write(f"  Temperature R²: {r2_per_sample[worst_idx_local]:.6f}\n\n")

        # Model architecture
        f.write("-"*80 + "\n")
        f.write("MODEL ARCHITECTURE\n")
        f.write("-"*80 + "\n")
        if first_surrogate.model:
            f.write("Neural Network Summary:\n")
            # Capture model summary
            import io
            stream = io.StringIO()
            first_surrogate.model.summary(print_fn=lambda x: stream.write(x + '\n'))
            f.write(stream.getvalue())
        f.write("\n")

        # Footer
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"  ✓ Performance report saved: {report_path}")


def plot_performance_metrics(surrogates, save_dir=None):
    """
    Create bar charts showing performance metrics for all fields.

    Parameters
    ----------
    surrogates : dict
        Dictionary of trained surrogates
    save_dir : Path, optional
        Directory to save plots
    """
    fields = [k for k in surrogates.keys() if not k.startswith('_')]

    # Collect metrics
    train_r2 = []
    test_r2 = []
    train_rmse = []
    test_rmse = []
    train_mae = []
    test_mae = []

    for field_name in fields:
        surrogate = surrogates[field_name]
        train_r2.append(surrogate.train_metrics['r2'])
        test_r2.append(surrogate.test_metrics['r2'])
        train_rmse.append(surrogate.train_metrics['rmse'])
        test_rmse.append(surrogate.test_metrics['rmse'])
        train_mae.append(surrogate.train_metrics['mae'])
        test_mae.append(surrogate.test_metrics['mae'])

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x = np.arange(len(fields))
    width = 0.35

    # R² scores
    ax_r2 = axes[0]
    bars1 = ax_r2.bar(x - width/2, train_r2, width, label='Train', color='steelblue', alpha=0.8)
    bars2 = ax_r2.bar(x + width/2, test_r2, width, label='Test', color='coral', alpha=0.8)
    ax_r2.set_ylabel('R² Score', fontsize=12)
    ax_r2.set_title('R² Score by Field', fontsize=14, fontweight='bold')
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels([f.replace('_', '\n') for f in fields], fontsize=10)
    ax_r2.legend()
    ax_r2.grid(True, alpha=0.3, axis='y')
    ax_r2.set_ylim([0.9, 1.0])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_r2.text(bar.get_x() + bar.get_width()/2., height,
                      f'{height:.4f}', ha='center', va='bottom', fontsize=8)

    # RMSE
    ax_rmse = axes[1]
    bars3 = ax_rmse.bar(x - width/2, train_rmse, width, label='Train', color='steelblue', alpha=0.8)
    bars4 = ax_rmse.bar(x + width/2, test_rmse, width, label='Test', color='coral', alpha=0.8)
    ax_rmse.set_ylabel('RMSE', fontsize=12)
    ax_rmse.set_title('RMSE by Field', fontsize=14, fontweight='bold')
    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels([f.replace('_', '\n') for f in fields], fontsize=10)
    ax_rmse.legend()
    ax_rmse.grid(True, alpha=0.3, axis='y')

    # MAE
    ax_mae = axes[2]
    bars5 = ax_mae.bar(x - width/2, train_mae, width, label='Train', color='steelblue', alpha=0.8)
    bars6 = ax_mae.bar(x + width/2, test_mae, width, label='Test', color='coral', alpha=0.8)
    ax_mae.set_ylabel('MAE', fontsize=12)
    ax_mae.set_title('MAE by Field', fontsize=14, fontweight='bold')
    ax_mae.set_xticks(x)
    ax_mae.set_xticklabels([f.replace('_', '\n') for f in fields], fontsize=10)
    ax_mae.legend()
    ax_mae.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_dir:
        save_path = save_dir / "performance_metrics.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Performance metrics saved: {save_path}")

    plt.show()


def visualize_predictions(surrogates, sim_index=None, save_dir=None):
    """
    Create side-by-side comparison plots of ground truth vs predictions (3D volume).

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

    # Get 3D coordinates (subsample for visualization performance)
    coords = data['coordinates']
    n_cells = coords.shape[0]
    step = max(1, n_cells // 3000)  # Show ~3000 points

    x = coords[::step, 0]
    y = coords[::step, 1]
    z = coords[::step, 2]

    print(f"  Plotting {len(x)} of {n_cells} cells")

    # Fields to plot
    fields_to_plot = ['temperature', 'pressure', 'velocity_magnitude']

    # Create 3D figure with 3 columns: Ground Truth, Prediction, Error
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(20, 7*len(fields_to_plot)))
    axes = []
    for i in range(len(fields_to_plot)):
        axes.append([])
        for j in range(3):
            ax = fig.add_subplot(len(fields_to_plot), 3, i*3 + j + 1, projection='3d')
            axes[i].append(ax)

    for i, field_name in enumerate(fields_to_plot):
        # Get ground truth
        if field_name == 'velocity_magnitude':
            vx_true = data['velocity_x'][sim_index][::step]
            vy_true = data['velocity_y'][sim_index][::step]
            vz_true = data['velocity_z'][sim_index][::step]
            true_field = np.sqrt(vx_true**2 + vy_true**2 + vz_true**2)

            vx_pred = surrogates['velocity_x'].predict(params)[0][::step]
            vy_pred = surrogates['velocity_y'].predict(params)[0][::step]
            vz_pred = surrogates['velocity_z'].predict(params)[0][::step]
            pred_field = np.sqrt(vx_pred**2 + vy_pred**2 + vz_pred**2)

            label = 'Velocity Magnitude (m/s)'
            cmap = 'plasma'
        else:
            true_field = data[field_name][sim_index][::step]
            pred_field = surrogates[field_name].predict(params)[0][::step]

            if field_name == 'temperature':
                label = 'Temperature (K)'
                cmap = 'hot'
            else:
                label = 'Pressure (Pa)'
                cmap = 'viridis'

        # Calculate error metrics (on full data, not subsampled)
        if field_name == 'velocity_magnitude':
            vx_true_full = data['velocity_x'][sim_index]
            vy_true_full = data['velocity_y'][sim_index]
            vz_true_full = data['velocity_z'][sim_index]
            true_field_full = np.sqrt(vx_true_full**2 + vy_true_full**2 + vz_true_full**2)

            vx_pred_full = surrogates['velocity_x'].predict(params)[0]
            vy_pred_full = surrogates['velocity_y'].predict(params)[0]
            vz_pred_full = surrogates['velocity_z'].predict(params)[0]
            pred_field_full = np.sqrt(vx_pred_full**2 + vy_pred_full**2 + vz_pred_full**2)
        else:
            true_field_full = data[field_name][sim_index]
            pred_field_full = surrogates[field_name].predict(params)[0]

        error = pred_field - true_field
        r2 = r2_score(true_field_full, pred_field_full)
        mae = mean_absolute_error(true_field_full, pred_field_full)
        rmse = np.sqrt(mean_squared_error(true_field_full, pred_field_full))

        # Column 1: Ground truth (3D)
        ax_true = axes[i][0]
        scatter_true = ax_true.scatter(x, y, z, c=true_field, cmap=cmap, s=8, alpha=0.6, edgecolors='none')
        ax_true.set_xlabel('X (m)', fontsize=9)
        ax_true.set_ylabel('Y (m)', fontsize=9)
        ax_true.set_zlabel('Z (m)', fontsize=9)
        ax_true.set_title(f'{field_name.replace("_", " ").title()} - Ground Truth', fontsize=11, fontweight='bold')
        cbar_true = plt.colorbar(scatter_true, ax=ax_true, pad=0.1, shrink=0.8)
        cbar_true.set_label(label, fontsize=9)

        # Column 2: Prediction (3D)
        ax_pred = axes[i][1]
        scatter_pred = ax_pred.scatter(x, y, z, c=pred_field, cmap=cmap, s=8, alpha=0.6, edgecolors='none')
        ax_pred.set_xlabel('X (m)', fontsize=9)
        ax_pred.set_ylabel('Y (m)', fontsize=9)
        ax_pred.set_zlabel('Z (m)', fontsize=9)
        ax_pred.set_title(
            f'{field_name.replace("_", " ").title()} - Prediction\n(R²={r2:.4f}, MAE={mae:.4f})',
            fontsize=11, fontweight='bold'
        )
        cbar_pred = plt.colorbar(scatter_pred, ax=ax_pred, pad=0.1, shrink=0.8)
        cbar_pred.set_label(label, fontsize=9)

        # Match color scales for ground truth and prediction
        vmin = min(true_field.min(), pred_field.min())
        vmax = max(true_field.max(), pred_field.max())
        scatter_true.set_clim(vmin, vmax)
        scatter_pred.set_clim(vmin, vmax)

        # Column 3: Error (3D with centered diverging colormap)
        from matplotlib.colors import TwoSlopeNorm
        ax_error = axes[i][2]
        error_max = max(abs(error.min()), abs(error.max()))
        norm = TwoSlopeNorm(vmin=-error_max, vcenter=0, vmax=error_max)
        scatter_error = ax_error.scatter(x, y, z, c=error, cmap='RdBu_r', s=8, alpha=0.6,
                                        edgecolors='none', norm=norm)
        ax_error.set_xlabel('X (m)', fontsize=9)
        ax_error.set_ylabel('Y (m)', fontsize=9)
        ax_error.set_zlabel('Z (m)', fontsize=9)
        ax_error.set_title(
            f'{field_name.replace("_", " ").title()} - Error\n(RMSE={rmse:.4f})',
            fontsize=11, fontweight='bold'
        )
        cbar_error = plt.colorbar(scatter_error, ax=ax_error, pad=0.1, shrink=0.8)
        cbar_error.set_label(f'Error ({label.split("(")[1].split(")")[0]})', fontsize=9)

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

    # Find available datasets (look inside dataset folders)
    project_dir = Path(__file__).parent.parent  # Go up from modules/ to project root
    datasets = []
    for item in project_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name != 'modules':
            # Look for NPZ files inside this folder
            npz_files = list(item.glob("*.npz"))
            if npz_files:
                datasets.append(npz_files[0])

    if not datasets:
        print("\n✗ No dataset files (.npz) found in the project directory!")
        print(f"  Directory: {project_dir}")
        print("\nPlease run runner.py first to generate a dataset.")
        exit(1)

    # Display available datasets
    print("\nAvailable datasets:")
    for i, dataset in enumerate(datasets, 1):
        size_mb = dataset.stat().st_size / (1024 * 1024)
        rel_path = dataset.relative_to(project_dir)
        print(f"  [{i}] {rel_path} ({size_mb:.2f} MB)")

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

    # Output directory is the parent folder of the NPZ file
    OUTPUT_DIR = DATASET_FILE.parent

    # Create subfolder for training outputs
    TRAINING_OUTPUT_DIR = OUTPUT_DIR / "training_results"
    TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {OUTPUT_DIR.relative_to(project_dir)}/")
    print(f"Training results subfolder: {TRAINING_OUTPUT_DIR.relative_to(project_dir)}/")

    print(f"\n{'='*70}")
    print(f"Training Configuration")
    print(f"{'='*70}")
    print(f"  Dataset: {DATASET_FILE.relative_to(project_dir)}")
    print(f"  Output directory: {OUTPUT_DIR.relative_to(project_dir)}")
    print(f"  Training results: {TRAINING_OUTPUT_DIR.relative_to(project_dir)}")

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

    # Plot training curves
    print(f"\n[1/4] Training curves...")
    plot_training_curves(surrogates, save_dir=TRAINING_OUTPUT_DIR)

    # Plot performance metrics
    print(f"\n[2/4] Performance metrics...")
    plot_performance_metrics(surrogates, save_dir=TRAINING_OUTPUT_DIR)

    # Random test sample
    print(f"\n[3/4] Random test sample...")
    visualize_predictions(surrogates, save_dir=TRAINING_OUTPUT_DIR)

    # Best and worst predictions
    test_idx = surrogates['_test_idx']
    temp_surrogate = surrogates['temperature']

    best_idx = test_idx[np.argmax(temp_surrogate.test_metrics['r2_per_sample'])]
    worst_idx = test_idx[np.argmin(temp_surrogate.test_metrics['r2_per_sample'])]

    print(f"\n[4/4] Best and worst predictions...")
    print(f"\n  Best prediction (highest R²):")
    visualize_predictions(surrogates, sim_index=best_idx, save_dir=TRAINING_OUTPUT_DIR)

    print(f"\n  Worst prediction (lowest R²):")
    visualize_predictions(surrogates, sim_index=worst_idx, save_dir=TRAINING_OUTPUT_DIR)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"\nModels saved to: {OUTPUT_DIR}")
    print(f"Visualizations saved to: {TRAINING_OUTPUT_DIR}")
    
