"""
Autoencoder Neural Network Module
===================================
Implements autoencoder-based neural network surrogate modeling with real-time training visualization.
Fully adaptive to any input/output size.
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


class EncoderNetwork(nn.Module):
    """
    Encoder neural network: Input parameters -> Latent representation
    """

    def __init__(self, input_dim, hidden_layers, latent_dim, activation='relu'):
        """
        Parameters
        ----------
        input_dim : int
            Number of input parameters
        hidden_layers : list
            List of hidden layer sizes
        latent_dim : int
            Size of latent space (compressed representation)
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
        layers.append(nn.Linear(prev_dim, latent_dim))

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
    Decoder neural network: Latent representation -> Reconstructed field
    """

    def __init__(self, latent_dim, hidden_layers, output_dim, activation='relu'):
        """
        Parameters
        ----------
        latent_dim : int
            Size of latent space (compressed representation)
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
        prev_dim = latent_dim

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
            Output fields (n_samples, n_outputs)
        """
        self.inputs = torch.FloatTensor(inputs)
        self.outputs = torch.FloatTensor(outputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


class AutoencoderModel:
    """
    Complete autoencoder surrogate model with training and inference.
    Fully adaptive to any input/output size.
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Model configuration including:
            - input_dim: Number of input parameters
            - encoder_hidden_layers: List of encoder hidden layer sizes
            - latent_dim: Size of latent space
            - decoder_hidden_layers: List of decoder hidden layer sizes
            - output_dim: Size of output field
            - activation: Activation function
            - learning_rate: Learning rate for optimizer
            - batch_size: Batch size for training
            - epochs: Number of training epochs
            - validation_split: Fraction of data for validation
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create networks
        self.encoder = EncoderNetwork(
            input_dim=config['input_dim'],
            hidden_layers=config['encoder_hidden_layers'],
            latent_dim=config['latent_dim'],
            activation=config.get('activation', 'relu')
        ).to(self.device)

        self.decoder = DecoderNetwork(
            latent_dim=config['latent_dim'],
            hidden_layers=config['decoder_hidden_layers'],
            output_dim=config['output_dim'],
            activation=config.get('activation', 'relu')
        ).to(self.device)

        # Optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(params, lr=config['learning_rate'])

        # Learning rate scheduler - use dynamically calculated patience
        scheduler_patience = config.get('lr_patience', 10)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=scheduler_patience
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

        print(f"\n{'='*70}")
        print(f"Autoencoder Model Created")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Encoder: {config['input_dim']} -> {config['encoder_hidden_layers']} -> {config['latent_dim']}")
        print(f"Decoder: {config['latent_dim']} -> {config['decoder_hidden_layers']} -> {config['output_dim']}")
        print(f"Total parameters: {self._count_parameters():,}")
        print(f"Compression ratio: {config['output_dim'] / config['latent_dim']:.1f}x")
        print(f"{'='*70}")

    def _count_parameters(self):
        """Count total trainable parameters."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        return encoder_params + decoder_params

    def train(self, X_train, Y_train, validation_split=0.2, early_stopping_patience=100):
        """
        Train the autoencoder model with real-time visualization.

        Parameters
        ----------
        X_train : ndarray
            Training inputs (n_samples, input_dim)
        Y_train : ndarray
            Training outputs (n_samples, output_dim)
        validation_split : float
            Fraction of data for validation
        early_stopping_patience : int
            Number of epochs without improvement before stopping
        """
        print(f"\n{'='*70}")
        print("STARTING TRAINING")
        print(f"{'='*70}")
        print(f"\nTraining mode: End-to-end autoencoder")
        print(f"  Input dimension: {X_train.shape[1]}")
        print(f"  Output dimension: {Y_train.shape[1]}")
        print(f"  Latent dimension: {self.config['latent_dim']}")
        print(f"  Samples: {X_train.shape[0]}")

        # Create dataset
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
                prev_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']

                # Log LR changes
                if current_lr < prev_lr:
                    print(f"\n[LR] Learning rate reduced: {prev_lr:.6f} -> {current_lr:.6f}")

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
                    improvement = best_val_loss - val_loss
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    if epoch % 10 != 0 and epoch > 10:  # Don't clutter regular output
                        print(f"\n[OK] New best validation loss: {best_val_loss:.6f} (improved by {improvement:.6f})")
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping_patience:
                    print(f"\n\n{'='*70}")
                    print(f"[STOP] Early stopping triggered after {epoch+1} epochs")
                    print(f"  Best validation loss: {best_val_loss:.6f}")
                    print(f"  No improvement for {early_stopping_patience} epochs")
                    print(f"{'='*70}")
                    break

        except KeyboardInterrupt:
            print("\n\n[STOP] Training interrupted by user")

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

        # Save configuration and history
        save_dict = {
            'config': self.config,
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }

        with open(save_dir / 'model_info.json', 'w') as f:
            json.dump(save_dict, f, indent=2)

        print(f"\n[OK] Model saved to: {save_dir}")

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
        AutoencoderModel
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

        print(f"\n[OK] Model loaded from: {load_dir}")

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
        self.fig.suptitle('Autoencoder Training Progress', fontsize=14, fontweight='bold')

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
        print("\n[OK] Training plot window will remain open. Close it to continue.")


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate trained model on test data.

    Parameters
    ----------
    model : AutoencoderModel
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


# Legacy function for backward compatibility
def plot_predictions_vs_actual(Y_test, Y_pred, output_indices=None, max_plots=4):
    """
    Legacy function - redirects to adaptive visualization.

    Parameters
    ----------
    Y_test : ndarray
        Actual test outputs (n_samples, n_outputs)
    Y_pred : ndarray
        Predicted outputs (n_samples, n_outputs)
    output_indices : list, optional
        Ignored - kept for backward compatibility
    max_plots : int
        Ignored - kept for backward compatibility
    """
    create_adaptive_visualizations(Y_test, Y_pred)
