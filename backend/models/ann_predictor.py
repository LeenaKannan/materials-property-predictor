"""Artificial Neural Network model for materials property prediction."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict
import os


class MaterialsANN(nn.Module):
    """Neural network for materials property prediction."""
    
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3
    ):
        """
        Initialize neural network.
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout probability
        """
        super(MaterialsANN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty using Monte Carlo dropout.
        
        Args:
            x: Input features
            n_samples: Number of MC samples
            
        Returns:
            Tuple of (mean predictions, std predictions)
        """
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        self.eval()  # Disable dropout
        
        return mean_pred, std_pred


class ANNPredictor:
    """Wrapper for training and using ANN models."""
    
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        device: str = None
    ):
        """
        Initialize predictor.
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout probability
            learning_rate: Learning rate for optimizer
            device: Device to use ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = MaterialsANN(input_size, hidden_layers, dropout_rate).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.training_history = {
            "train_loss": [],
            "val_loss": []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Tuple[float, Optional[float]]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            
        Returns:
            Tuple of (train_loss, val_loss)
        """
        self.model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        val_loss = None
        if val_loader is not None:
            val_loss = self.evaluate(val_loader)
        
        return train_loss, val_loss
    
    def evaluate(self, data_loader: DataLoader) -> float:
        """Evaluate model on data."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                total_loss += loss.item()
        
        return total_loss / len(data_loader)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10
    ):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
        """
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).reshape(-1, 1)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val).reshape(-1, 1)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, val_loss = self.train_epoch(train_loader, val_loader)
            
            self.training_history["train_loss"].append(train_loss)
            if val_loss is not None:
                self.training_history["val_loss"].append(val_loss)
            
            # Early stopping
            if val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}", end="")
                if val_loss is not None:
                    print(f", Val Loss: {val_loss:.4f}")
                else:
                    print()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy().flatten()
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_samples: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation.
        
        Args:
            X: Input features
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (mean predictions, std predictions)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        mean_pred, std_pred = self.model.predict_with_uncertainty(X_tensor, n_samples)
        return mean_pred.flatten(), std_pred.flatten()
    
    def save(self, filepath: str):
        """Save model to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, filepath)
    
    def load(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']