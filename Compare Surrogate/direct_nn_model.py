"""
Direct Neural Network Model
============================
Simple feedforward neural network without bottleneck compression.
Maps directly from input parameters to output fields.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from datetime import datetime


class DirectNN(nn.Module):
    """
    Direct feedforward neural network.

    Architecture: input -> hidden_layers -> output
    No bottleneck/compression.
    """

    def __init__(self, input_dim, output_dim, hidden_dims=[128, 256, 512, 256, 128]):
        """
        Initialize Direct NN.

        Parameters
        ----------
        input_dim : int
            Number of input parameters
        output_dim : int
            Number of output values
        hidden_dims : list
            Hidden layer dimensions
        """
        super(DirectNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
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

    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class DirectNNModel:
    """
    Direct NN model wrapper for training and inference.
    """

    def __init__(self, input_dim, output_dim, hidden_dims=None):
        """
        Initialize Direct NN model.

        Parameters
        ----------
        input_dim : int
            Input dimension
        output_dim : int
            Output dimension
        hidden_dims : list, optional
            Hidden layer dimensions (auto-calculated if None)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Auto-calculate hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = self._calculate_hidden_dims(input_dim, output_dim)

        self.hidden_dims = hidden_dims

        # Create model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DirectNN(input_dim, output_dim, hidden_dims).to(self.device)

        # Normalization parameters
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None

        # Training history
        self.train_losses = []
        self.val_losses = []

    def _calculate_hidden_dims(self, input_dim, output_dim):
        """
        Calculate hidden layer dimensions based on problem size.

        Strategy: Pyramid structure expanding from input to middle,
        then contracting toward output.
        """
        # Calculate middle layer size (wider than both input and output)
        middle_size = max(128, min(1024, output_dim // 2))

        # Small problem: fewer layers
        if output_dim < 500:
            return [64, 128, 64]

        # Medium problem: moderate depth
        elif output_dim < 2000:
            return [128, 256, 512, 256, 128]

        # Large problem: deeper network
        else:
            return [256, 512, 1024, 512, 256]

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

    def train(self, X_train, Y_train, X_val, Y_val,
              epochs=1000, batch_size=16, learning_rate=0.001,
              early_stop_patience=50, lr_patience=20):
        """
        Train the Direct NN model.

        Parameters
        ----------
        X_train : np.ndarray
            Training inputs
        Y_train : np.ndarray
            Training outputs
        X_val : np.ndarray
            Validation inputs
        Y_val : np.ndarray
            Validation outputs
        epochs : int
            Maximum epochs
        batch_size : int
            Batch size
        learning_rate : float
            Initial learning rate
        early_stop_patience : int
            Early stopping patience
        lr_patience : int
            LR scheduler patience

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
        X_val_norm = self.normalize_input(X_val)
        Y_val_norm = self.normalize_output(Y_val)

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_norm).to(self.device)
        Y_train_t = torch.FloatTensor(Y_train_norm).to(self.device)
        X_val_t = torch.FloatTensor(X_val_norm).to(self.device)
        Y_val_t = torch.FloatTensor(Y_val_norm).to(self.device)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Setup training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=lr_patience
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        self.train_losses = []
        self.val_losses = []

        print("\n" + "="*60)
        print("TRAINING DIRECT NN")
        print("="*60)
        print(f"Architecture: {self.input_dim} -> {' -> '.join(map(str, self.hidden_dims))} -> {self.output_dim}")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*60)

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_Y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, Y_val_t).item()
                self.val_losses.append(val_loss)

            # LR scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            if (epoch + 1) % 50 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:4d}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"LR: {current_lr:.2e}")

            # Early stopping check
            if patience_counter >= early_stop_patience:
                print(f"\n[OK] Early stopping triggered at epoch {epoch+1}")
                break

        print("="*60)
        print("TRAINING COMPLETE")
        print("="*60)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'final_epoch': epoch + 1
        }

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
        torch.save(self.model.state_dict(), save_dir / "model.pth")

        # Save configuration
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'input_mean': self.input_mean.tolist(),
            'input_std': self.input_std.tolist(),
            'output_mean': self.output_mean.tolist(),
            'output_std': self.output_std.tolist()
        }

        with open(save_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(save_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)

    @classmethod
    def load(cls, save_dir):
        """
        Load model from directory.

        Parameters
        ----------
        save_dir : Path
            Directory containing saved model

        Returns
        -------
        DirectNNModel
            Loaded model
        """
        save_dir = Path(save_dir)

        # Load configuration
        with open(save_dir / "config.json", 'r') as f:
            config = json.load(f)

        # Create model
        model = cls(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            hidden_dims=config['hidden_dims']
        )

        # Load weights
        model.model.load_state_dict(
            torch.load(save_dir / "model.pth", map_location=model.device)
        )

        # Load normalization parameters
        model.input_mean = np.array(config['input_mean'])
        model.input_std = np.array(config['input_std'])
        model.output_mean = np.array(config['output_mean'])
        model.output_std = np.array(config['output_std'])

        # Load training history if available
        history_file = save_dir / "training_history.json"
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
                model.train_losses = history.get('train_losses', [])
                model.val_losses = history.get('val_losses', [])

        return model
