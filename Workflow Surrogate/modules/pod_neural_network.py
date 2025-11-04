"""
POD Neural Network Module
==========================
Implements POD-based neural network surrogate modeling with real-time training visualization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime


class PODCompressor:
    """
    Performs Proper Orthogonal Decomposition on field data.
    """

    def __init__(self, n_modes=50, energy_threshold=0.99):
        """
        Parameters
        ----------
        n_modes : int
            Number of POD modes to retain
        energy_threshold : float
            Energy capture threshold (0-1)
        """
        self.n_modes = n_modes
        self.energy_threshold = energy_threshold
        self.pod_basis = None
        self.mean_field = None
        self.singular_values = None
        self.actual_modes = 0
        self.energy_captured = 0.0

    def fit(self, field_data):
        """
        Compute POD basis from field data.

        Parameters
        ----------
        field_data : ndarray
            Field data with shape (n_samples, n_field_points)

        Returns
        -------
        self
        """
        print(f"\nComputing POD decomposition...")
        print(f"  Input shape: {field_data.shape}")
        print(f"  Requested modes: {self.n_modes}")

        # Compute mean field and center data
        self.mean_field = np.mean(field_data, axis=0)
        centered_data = field_data - self.mean_field

        # Perform SVD
        # For efficiency, use reduced SVD: U, S, Vt = svd(data, full_matrices=False)
        # field_data = U @ diag(S) @ Vt
        # POD basis is V (Vt transposed)
        print("  Performing SVD...")
        U, S, Vt = np.linalg.svd(centered_data, full_matrices=False)

        self.singular_values = S

        # Determine number of modes based on energy threshold
        total_energy = np.sum(S**2)
        cumulative_energy = np.cumsum(S**2) / total_energy

        # Use minimum of: requested modes, modes for energy threshold, available modes
        modes_for_energy = np.searchsorted(cumulative_energy, self.energy_threshold) + 1
        self.actual_modes = min(self.n_modes, modes_for_energy, len(S))
        self.energy_captured = cumulative_energy[self.actual_modes - 1]

        # POD basis: transpose of V (first actual_modes rows of Vt)
        self.pod_basis = Vt[:self.actual_modes, :].T

        print(f"  ✓ POD complete")
        print(f"    Modes retained: {self.actual_modes}")
        print(f"    Energy captured: {self.energy_captured*100:.2f}%")

        return self

    def transform(self, field_data):
        """
        Project field data onto POD basis to get coefficients.

        Parameters
        ----------
        field_data : ndarray
            Field data with shape (n_samples, n_field_points)

        Returns
        -------
        ndarray
            POD coefficients with shape (n_samples, n_modes)
        """
        centered_data = field_data - self.mean_field
        coefficients = centered_data @ self.pod_basis
        return coefficients

    def inverse_transform(self, coefficients):
        """
        Reconstruct field data from POD coefficients.

        Parameters
        ----------
        coefficients : ndarray
            POD coefficients with shape (n_samples, n_modes)

        Returns
        -------
        ndarray
            Reconstructed field data with shape (n_samples, n_field_points)
        """
        reconstructed = coefficients @ self.pod_basis.T + self.mean_field
        return reconstructed

    def save(self, filepath):
        """Save POD compressor to file."""
        save_dict = {
            'n_modes': self.n_modes,
            'actual_modes': self.actual_modes,
            'energy_threshold': self.energy_threshold,
            'energy_captured': self.energy_captured,
            'pod_basis': self.pod_basis,
            'mean_field': self.mean_field,
            'singular_values': self.singular_values
        }
        np.savez(filepath, **save_dict)

    @classmethod
    def load(cls, filepath):
        """Load POD compressor from file."""
        data = np.load(filepath, allow_pickle=True)
        compressor = cls(
            n_modes=int(data['n_modes']),
            energy_threshold=float(data['energy_threshold'])
        )
        compressor.actual_modes = int(data['actual_modes'])
        compressor.energy_captured = float(data['energy_captured'])
        compressor.pod_basis = data['pod_basis']
        compressor.mean_field = data['mean_field']
        compressor.singular_values = data['singular_values']
        return compressor


class EncoderNetwork(nn.Module):
    """
    Encoder neural network: Input parameters -> POD coefficients
    """

    def __init__(self, input_dim, hidden_layers, output_dim, activation='relu'):
        """
        Parameters
        ----------
        input_dim : int
            Number of input parameters
        hidden_layers : list
            List of hidden layer sizes
        output_dim : int
            Number of POD coefficients (modes)
        activation : str
            Activation function ('relu', 'tanh', 'elu')
        """
        super().__init__()

        self.activation = self._get_activation(activation)

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim

        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, activation):
        """Get activation function."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'elu':
            return nn.ELU()
        else:
            return nn.ReLU()

    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class DecoderNetwork(nn.Module):
    """
    Decoder neural network: POD coefficients -> Reconstructed field
    """

    def __init__(self, input_dim, hidden_layers, output_dim, activation='relu'):
        """
        Parameters
        ----------
        input_dim : int
            Number of POD coefficients (modes)
        hidden_layers : list
            List of hidden layer sizes
        output_dim : int
            Size of reconstructed field
        activation : str
            Activation function ('relu', 'tanh', 'elu')
        """
        super().__init__()

        self.activation = self._get_activation(activation)

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim

        # Output layer (no activation for field reconstruction)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, activation):
        """Get activation function."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'elu':
            return nn.ELU()
        else:
            return nn.ReLU()

    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class SimulationDataset(Dataset):
    """PyTorch dataset for simulation data."""

    def __init__(self, inputs, outputs):
        """
        Parameters
        ----------
        inputs : ndarray
            Input parameters (n_samples, n_inputs)
        outputs : ndarray
            Output fields or POD coefficients (n_samples, n_outputs)
        """
        self.inputs = torch.FloatTensor(inputs)
        self.outputs = torch.FloatTensor(outputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


class PODNNModel:
    """
    Complete POD-NN surrogate model with training and inference.
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Model configuration including:
            - input_dim
            - encoder_hidden_layers
            - pod_modes
            - decoder_hidden_layers
            - output_dim
            - activation
            - learning_rate
            - batch_size
            - epochs
            - validation_split
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create networks
        self.encoder = EncoderNetwork(
            input_dim=config['input_dim'],
            hidden_layers=config['encoder_hidden_layers'],
            output_dim=config['pod_modes'],
            activation=config.get('activation', 'relu')
        ).to(self.device)

        self.decoder = DecoderNetwork(
            input_dim=config['pod_modes'],
            hidden_layers=config['decoder_hidden_layers'],
            output_dim=config['output_dim'],
            activation=config.get('activation', 'relu')
        ).to(self.device)

        # Optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(params, lr=config['learning_rate'])

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=50
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }

        # POD compressor
        self.pod_compressor = None

        print(f"\n{'='*70}")
        print(f"POD-NN Model Created")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Encoder: {config['input_dim']} -> {config['encoder_hidden_layers']} -> {config['pod_modes']}")
        print(f"Decoder: {config['pod_modes']} -> {config['decoder_hidden_layers']} -> {config['output_dim']}")
        print(f"Total parameters: {self._count_parameters():,}")
        print(f"{'='*70}")

    def _count_parameters(self):
        """Count total trainable parameters."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        return encoder_params + decoder_params

    def train(self, X_train, Y_train, use_pod=True, pod_modes=None,
              validation_split=0.2, early_stopping_patience=100):
        """
        Train the POD-NN model with real-time visualization.

        Parameters
        ----------
        X_train : ndarray
            Training inputs (n_samples, input_dim)
        Y_train : ndarray
            Training outputs (n_samples, output_dim)
        use_pod : bool
            Whether to use POD compression on outputs
        pod_modes : int
            Number of POD modes (if None, uses config)
        validation_split : float
            Fraction of data for validation
        early_stopping_patience : int
            Number of epochs without improvement before stopping
        """
        print(f"\n{'='*70}")
        print("STARTING TRAINING")
        print(f"{'='*70}")

        # Apply POD to analyze output structure (but don't transform training data!)
        if use_pod:
            if pod_modes is None:
                pod_modes = self.config['pod_modes']

            self.pod_compressor = PODCompressor(n_modes=pod_modes)
            self.pod_compressor.fit(Y_train)

            print(f"\nPOD analysis of outputs:")
            print(f"  Original dimension: {Y_train.shape[1]}")
            print(f"  POD modes (latent space): {self.pod_compressor.actual_modes}")
            print(f"  Energy captured: {self.pod_compressor.energy_captured*100:.2f}%")
            print(f"  Compression ratio: {Y_train.shape[1] / self.pod_compressor.actual_modes:.1f}x")

            # Update config to match actual POD modes
            self.config['pod_modes'] = self.pod_compressor.actual_modes

            print(f"\nTraining mode: End-to-end autoencoder")
            print(f"  Encoder will learn to compress to {self.pod_compressor.actual_modes} latent features")
            print(f"  Decoder will learn to reconstruct full {Y_train.shape[1]}-dimensional output")

        # Create dataset with ORIGINAL outputs (not POD-transformed!)
        # The encoder-decoder will learn the compression/reconstruction
        full_dataset = SimulationDataset(X_train, Y_train)

        # Split into train/validation
        n_val = int(len(full_dataset) * validation_split)
        n_train = len(full_dataset) - n_val
        train_dataset, val_dataset = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )

        print(f"\nDataset split:")
        print(f"  Training samples: {n_train}")
        print(f"  Validation samples: {n_val}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Batches per epoch: {len(train_loader)}")

        # Setup real-time plotting
        plotter = RealTimePlotter(self.config['epochs'])

        # Training loop
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        print(f"\n{'='*70}")
        print("Training Progress:")
        print(f"{'='*70}\n")

        try:
            for epoch in range(self.config['epochs']):
                epoch_start = time.time()

                # Training phase
                self.encoder.train()
                self.decoder.train()
                train_loss = 0.0

                for batch_inputs, batch_outputs in train_loader:
                    batch_inputs = batch_inputs.to(self.device)
                    batch_outputs = batch_outputs.to(self.device)

                    # Forward pass - full encoder-decoder pipeline
                    self.optimizer.zero_grad()
                    latent = self.encoder(batch_inputs)  # Compress to latent space
                    reconstructed = self.decoder(latent)  # Reconstruct full field

                    # Loss in full field space (autoencoder reconstruction loss)
                    loss = self.criterion(reconstructed, batch_outputs)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # Validation phase
                self.encoder.eval()
                self.decoder.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_inputs, batch_outputs in val_loader:
                        batch_inputs = batch_inputs.to(self.device)
                        batch_outputs = batch_outputs.to(self.device)

                        latent = self.encoder(batch_inputs)
                        reconstructed = self.decoder(latent)

                        # Loss in full field space
                        loss = self.criterion(reconstructed, batch_outputs)

                        val_loss += loss.item()

                val_loss /= len(val_loader)

                # Update learning rate
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']

                # Record history
                epoch_time = time.time() - epoch_start
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['learning_rate'].append(current_lr)
                self.history['epoch_time'].append(epoch_time)

                # Update plot
                plotter.update(epoch, train_loss, val_loss, current_lr)

                # Print progress
                if epoch % 10 == 0 or epoch == self.config['epochs'] - 1:
                    print(f"Epoch [{epoch+1:4d}/{self.config['epochs']}] "
                          f"Train Loss: {train_loss:.6f} | "
                          f"Val Loss: {val_loss:.6f} | "
                          f"LR: {current_lr:.6f} | "
                          f"Time: {epoch_time:.2f}s")

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping_patience:
                    print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                    print(f"  Best validation loss: {best_val_loss:.6f}")
                    break

        except KeyboardInterrupt:
            print("\n\n⚠ Training interrupted by user")

        finally:
            plotter.finalize()

        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Final training loss: {train_loss:.6f}")
        print(f"Total training time: {sum(self.history['epoch_time']):.1f}s")
        print(f"{'='*70}\n")

    def predict(self, X):
        """
        Make predictions on new input data.

        Parameters
        ----------
        X : ndarray
            Input parameters (n_samples, input_dim)

        Returns
        -------
        ndarray
            Predicted outputs (n_samples, output_dim)
        """
        self.encoder.eval()
        self.decoder.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            latent = self.encoder(X_tensor)  # Compress to latent space
            Y_pred = self.decoder(latent)    # Decode to full field

        # Decoder already outputs full field - no POD inverse needed
        return Y_pred.cpu().numpy()

    def save(self, save_dir):
        """
        Save complete model to directory.

        Parameters
        ----------
        save_dir : Path
            Directory to save model files
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save PyTorch models
        torch.save(self.encoder.state_dict(), save_dir / 'encoder.pth')
        torch.save(self.decoder.state_dict(), save_dir / 'decoder.pth')

        # Save POD compressor if exists
        if self.pod_compressor is not None:
            self.pod_compressor.save(save_dir / 'pod_compressor.npz')

        # Save configuration and history
        save_dict = {
            'config': self.config,
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }

        with open(save_dir / 'model_info.json', 'w') as f:
            json.dump(save_dict, f, indent=2)

        print(f"\n✓ Model saved to: {save_dir}")

    @classmethod
    def load(cls, load_dir):
        """
        Load complete model from directory.

        Parameters
        ----------
        load_dir : Path
            Directory containing model files

        Returns
        -------
        PODNNModel
            Loaded model
        """
        load_dir = Path(load_dir)

        # Load configuration
        with open(load_dir / 'model_info.json', 'r') as f:
            save_dict = json.load(f)

        # Create model
        model = cls(save_dict['config'])
        model.history = save_dict['history']

        # Load PyTorch models
        model.encoder.load_state_dict(torch.load(load_dir / 'encoder.pth'))
        model.decoder.load_state_dict(torch.load(load_dir / 'decoder.pth'))

        # Load POD compressor if exists
        pod_file = load_dir / 'pod_compressor.npz'
        if pod_file.exists():
            model.pod_compressor = PODCompressor.load(pod_file)

        print(f"\n✓ Model loaded from: {load_dir}")

        return model


class RealTimePlotter:
    """
    Real-time training visualization with matplotlib.
    """

    def __init__(self, total_epochs):
        """
        Parameters
        ----------
        total_epochs : int
            Total number of training epochs
        """
        self.total_epochs = total_epochs

        # Create figure with subplots
        plt.ion()  # Interactive mode
        self.fig, self.axes = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('POD-NN Training Progress', fontsize=14, fontweight='bold')

        # Loss plot
        self.ax_loss = self.axes[0]
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss (MSE)')
        self.ax_loss.set_title('Training and Validation Loss')
        self.ax_loss.set_yscale('log')
        self.ax_loss.grid(True, alpha=0.3)

        self.line_train, = self.ax_loss.plot([], [], 'b-', label='Training', linewidth=2)
        self.line_val, = self.ax_loss.plot([], [], 'r-', label='Validation', linewidth=2)
        self.ax_loss.legend()

        # Learning rate plot
        self.ax_lr = self.axes[1]
        self.ax_lr.set_xlabel('Epoch')
        self.ax_lr.set_ylabel('Learning Rate')
        self.ax_lr.set_title('Learning Rate Schedule')
        self.ax_lr.set_yscale('log')
        self.ax_lr.grid(True, alpha=0.3)

        self.line_lr, = self.ax_lr.plot([], [], 'g-', linewidth=2)

        # Data storage
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        plt.tight_layout()
        plt.show(block=False)

    def update(self, epoch, train_loss, val_loss, learning_rate):
        """Update plots with new data."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(learning_rate)

        # Update loss plot
        self.line_train.set_data(self.epochs, self.train_losses)
        self.line_val.set_data(self.epochs, self.val_losses)

        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

        # Update learning rate plot
        self.line_lr.set_data(self.epochs, self.learning_rates)

        self.ax_lr.relim()
        self.ax_lr.autoscale_view()

        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def finalize(self):
        """Finalize plot and keep window open."""
        plt.ioff()
        print("\n✓ Training plot window will remain open. Close it to continue.")


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate trained model on test data.

    Parameters
    ----------
    model : PODNNModel
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


def plot_predictions_vs_actual(Y_test, Y_pred, output_indices=None, max_plots=4):
    """
    Create scatter plots comparing predictions to actual values.
    For field data (many outputs), creates overall and sample-wise comparisons.

    Parameters
    ----------
    Y_test : ndarray
        Actual test outputs (n_samples, n_outputs)
    Y_pred : ndarray
        Predicted outputs (n_samples, n_outputs)
    output_indices : list, optional
        Specific sample indices to plot. If None, selects samples automatically.
    max_plots : int
        Maximum number of sample plots to show
    """
    n_samples, n_outputs = Y_test.shape

    # If we have many outputs (field data), create different visualizations
    if n_outputs > 100:
        # Field data visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Field Data Predictions vs. Actual', fontsize=14, fontweight='bold')

        # Plot 1: Overall scatter of all field values
        ax1 = axes[0]
        sample_size = min(10000, Y_test.size)  # Limit to 10k points for performance
        indices = np.random.choice(Y_test.size, sample_size, replace=False)
        y_true_flat = Y_test.flatten()[indices]
        y_pred_flat = Y_pred.flatten()[indices]

        ax1.scatter(y_true_flat, y_pred_flat, alpha=0.3, s=10)

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

        # Plot 2: Per-sample errors
        ax2 = axes[1]
        sample_errors = np.sqrt(np.mean((Y_test - Y_pred) ** 2, axis=1))  # RMSE per sample

        ax2.bar(range(n_samples), sample_errors, color='steelblue', alpha=0.7)
        ax2.axhline(np.mean(sample_errors), color='r', linestyle='--',
                    linewidth=2, label=f'Mean RMSE={np.mean(sample_errors):.4e}')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Prediction Error per Sample')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

    else:
        # Original visualization for small number of outputs
        if output_indices is None:
            # Plot first few outputs
            output_indices = list(range(min(max_plots, n_outputs)))

        n_plots = len(output_indices)

        # Create figure
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        fig.suptitle('Predictions vs. Actual Values', fontsize=14, fontweight='bold')

        for idx, output_idx in enumerate(output_indices):
            ax = axes[idx]

            y_true = Y_test[:, output_idx]
            y_pred = Y_pred[:, output_idx]

            # Scatter plot
            ax.scatter(y_true, y_pred, alpha=0.5, s=20)

            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')

            # Calculate R² for this output
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'Output {output_idx+1} (R²={r2:.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)

    print("\n✓ Prediction plots displayed. Close window to continue.")
