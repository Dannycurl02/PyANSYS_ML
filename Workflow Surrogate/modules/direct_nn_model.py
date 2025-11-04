"""
Direct Neural Network Model
============================
Feedforward neural network without bottleneck compression.
Maps directly from input parameters to output fields.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class DirectNN(nn.Module):
    """
    Direct feedforward neural network.

    Architecture: input -> hidden_layers -> output
    No bottleneck/compression.
    """

    def __init__(self, input_dim, output_dim, hidden_dims=None):
        """
        Initialize Direct NN.

        Parameters
        ----------
        input_dim : int
            Number of input parameters
        output_dim : int
            Number of output values
        hidden_dims : list, optional
            Hidden layer dimensions (auto-calculated if None)
        """
        super(DirectNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Auto-calculate hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = self._calculate_hidden_dims(input_dim, output_dim)

        self.hidden_dims = hidden_dims

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def _calculate_hidden_dims(self, input_dim, output_dim):
        """
        Calculate hidden layer dimensions based on problem size.

        Strategy: Pyramid structure expanding from input to middle,
        then contracting toward output.
        """
        # Calculate middle layer size based on output dimension
        if output_dim < 500:
            # Small problem: [64, 128, 64]
            return [64, 128, 64]
        elif output_dim < 2000:
            # Medium problem: [128, 256, 512, 256, 128]
            return [128, 256, 512, 256, 128]
        elif output_dim < 5000:
            # Large problem: [256, 512, 1024, 512, 256]
            return [256, 512, 1024, 512, 256]
        else:
            # Very large problem: [512, 1024, 2048, 1024, 512]
            return [512, 1024, 2048, 1024, 512]

    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class DirectNNModel:
    """
    Direct NN model wrapper for training and inference.
    """

    def __init__(self, config):
        """
        Initialize Direct NN model.

        Parameters
        ----------
        config : dict
            Configuration dictionary with keys:
            - input_dim: int
            - output_dim: int
            - hidden_dims: list (optional)
            - learning_rate: float
            - batch_size: int
            - epochs: int
            - validation_split: float (optional)
            - lr_patience: int (optional)
        """
        self.config = config
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.hidden_dims = config.get('hidden_dims', None)

        # Create model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DirectNN(
            self.input_dim,
            self.output_dim,
            self.hidden_dims
        ).to(self.device)

        # Get actual hidden dims used
        self.hidden_dims = self.model.hidden_dims

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate']
        )

        # Learning rate scheduler
        scheduler_patience = config.get('lr_patience', 10)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=scheduler_patience
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Normalization parameters
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }

    def fit_normalization(self, X_train, Y_train):
        """
        Fit normalization parameters.

        Parameters
        ----------
        X_train : np.ndarray
            Training inputs (n_samples, input_dim)
        Y_train : np.ndarray
            Training outputs (n_samples, output_dim)
        """
        self.input_mean = np.mean(X_train, axis=0, keepdims=True)
        self.input_std = np.std(X_train, axis=0, keepdims=True)
        self.input_std[self.input_std < 1e-8] = 1.0

        self.output_mean = np.mean(Y_train, axis=0, keepdims=True)
        self.output_std = np.std(Y_train, axis=0, keepdims=True)
        self.output_std[self.output_std < 1e-8] = 1.0

    def normalize_input(self, X):
        """Normalize input."""
        return (X - self.input_mean) / self.input_std

    def normalize_output(self, Y):
        """Normalize output."""
        return (Y - self.output_mean) / self.output_std

    def denormalize_output(self, Y_norm):
        """Denormalize output."""
        return Y_norm * self.output_std + self.output_mean

    def train(self, X_train, Y_train, validation_split=0.2, early_stopping_patience=50):
        """
        Train the Direct NN model.

        Parameters
        ----------
        X_train : np.ndarray
            Training inputs
        Y_train : np.ndarray
            Training outputs
        validation_split : float
            Fraction of data to use for validation
        early_stopping_patience : int
            Early stopping patience

        Returns
        -------
        dict
            Training history
        """
        # Fit normalization
        self.fit_normalization(X_train, Y_train)

        # Normalize data
        X_train_norm = self.normalize_input(X_train)
        Y_train_norm = self.normalize_output(Y_train)

        # Split validation
        n_samples = len(X_train)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        X_train_split = X_train_norm[train_indices]
        Y_train_split = Y_train_norm[train_indices]
        X_val = X_train_norm[val_indices]
        Y_val = Y_train_norm[val_indices]

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_split).to(self.device)
        Y_train_t = torch.FloatTensor(Y_train_split).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        Y_val_t = torch.FloatTensor(Y_val).to(self.device)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )

        # Training loop
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        print("\n" + "="*70)
        print("DIRECT NN MODEL")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Architecture: {self.input_dim} -> {' -> '.join(map(str, self.hidden_dims))} -> {self.output_dim}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*70)

        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        print(f"Training samples: {len(X_train_split)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Batches per epoch: {len(train_loader)}")
        print("="*70)

        # Initialize plotter
       
        plotter = RealTimePlotter(self.config['epochs'])

        try:
            for epoch in range(self.config['epochs']):
                epoch_start = time.time()

                # Training phase
                self.model.train()
                train_loss = 0.0

                for batch_X, batch_Y in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_Y)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # Validation phase
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_t)
                    val_loss = self.criterion(val_outputs, Y_val_t).item()

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
                if (epoch + 1) % 10 == 0 or epoch == 0:
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
                    if epoch % 10 != 0 and epoch > 10:
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
        Make predictions.

        Parameters
        ----------
        X : np.ndarray
            Input parameters

        Returns
        -------
        np.ndarray
            Predictions
        """
        self.model.eval()

        X_norm = self.normalize_input(X)
        X_t = torch.FloatTensor(X_norm).to(self.device)

        with torch.no_grad():
            Y_norm = self.model(X_t).cpu().numpy()

        return self.denormalize_output(Y_norm)

    def save(self, save_dir):
        """
        Save model to directory.

        Parameters
        ----------
        save_dir : Path
            Directory to save model
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.model.state_dict(), save_dir / "direct_nn.pth")

        # Prepare save dictionary
        save_dict = {
            'config': self.config,
            'hidden_dims': self.hidden_dims,
            'normalization': {
                'input_mean': self.input_mean.tolist(),
                'input_std': self.input_std.tolist(),
                'output_mean': self.output_mean.tolist(),
                'output_std': self.output_std.tolist()
            },
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }

        with open(save_dir / 'model_info.json', 'w') as f:
            json.dump(save_dict, f, indent=2)

        print(f"\n[OK] Model saved to: {save_dir}")

    @classmethod
    def load(cls, load_dir):
        """
        Load model from directory.

        Parameters
        ----------
        load_dir : Path
            Directory containing saved model

        Returns
        -------
        DirectNNModel
            Loaded model
        """
        load_dir = Path(load_dir)

        # Load configuration
        with open(load_dir / 'model_info.json', 'r') as f:
            save_dict = json.load(f)

        # Create model
        config = save_dict['config']
        config['hidden_dims'] = save_dict['hidden_dims']
        model = cls(config)
        model.history = save_dict['history']

        # Load PyTorch model
        model.model.load_state_dict(
            torch.load(load_dir / 'direct_nn.pth', map_location=model.device)
        )

        # Load normalization parameters
        norm = save_dict['normalization']
        model.input_mean = np.array(norm['input_mean'])
        model.input_std = np.array(norm['input_std'])
        model.output_mean = np.array(norm['output_mean'])
        model.output_std = np.array(norm['output_std'])

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