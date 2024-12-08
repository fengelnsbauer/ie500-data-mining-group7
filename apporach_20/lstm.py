# lstm.py

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from features import RaceFeatures
import logging

# Define the F1Dataset class
class F1Dataset(Dataset):
    def __init__(self, sequences: np.ndarray, static_features: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.static_features = torch.FloatTensor(static_features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'static': self.static_features[idx],
            'target': self.targets[idx]
        }

# Define the data preprocessor
class F1DataPreprocessor:
    def __init__(self):
        self.static_scaler = StandardScaler()
        self.dynamic_scaler = StandardScaler()
        self.lap_time_scaler = StandardScaler()
        self.race_features = RaceFeatures()
        self.static_feature_names = self.race_features.static_features  # Initialize with feature names from RaceFeatures
    
    def fit_scalers(self, sequences: np.ndarray, static_features: np.ndarray, targets: np.ndarray):
        """
        Fit scalers on training data.
        """
        # Fit lap_time_scaler on targets
        self.lap_time_scaler.fit(targets.reshape(-1, 1))
        
        # Fit static_scaler on static features
        self.static_scaler.fit(static_features)
    
        # Extract dynamic features (excluding the lap time column)
        dynamic_features_flat = sequences[:, :, 1:].reshape(-1, sequences.shape[2] - 1)
        self.dynamic_scaler.fit(dynamic_features_flat)
    
    def transform_static_features(self, static_features: np.ndarray) -> np.ndarray:
        """
        Transform static features using fitted scaler.
        
        Parameters:
        -----------
        static_features : numpy.ndarray
            Static features to transform
            
        Returns:
        --------
        numpy.ndarray
            Scaled static features
        """
        if static_features.ndim == 1:
            static_features = static_features.reshape(1, -1)
        return self.static_scaler.transform(static_features)
    
    def transform_data(self, sequences: np.ndarray, static_features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform all data using fitted scalers.
        """
        # Scale targets (lap times)
        targets_scaled = self.lap_time_scaler.transform(targets.reshape(-1, 1)).flatten()
    
        # Scale static features
        static_features_scaled = self.transform_static_features(static_features)
    
        # Scale dynamic features
        dynamic_features_flat = sequences[:, :, 1:].reshape(-1, sequences.shape[2] - 1)
        dynamic_features_scaled = self.dynamic_scaler.transform(dynamic_features_flat)
    
        # Reconstruct sequences with scaled dynamic features
        dynamic_features_scaled = dynamic_features_scaled.reshape(sequences.shape[0], sequences.shape[1], -1)
        sequences_scaled = np.concatenate((
            self.lap_time_scaler.transform(sequences[:, :, 0].reshape(-1, 1)).reshape(sequences.shape[0], sequences.shape[1], 1),
            dynamic_features_scaled
        ), axis=2)
    
        return sequences_scaled, static_features_scaled, targets_scaled
    
    def inverse_transform_lap_times(self, scaled_lap_times: np.ndarray) -> np.ndarray:
        """
        Convert scaled lap times back to original scale.
        """
        if scaled_lap_times.ndim == 1:
            scaled_lap_times = scaled_lap_times.reshape(-1, 1)
        return self.lap_time_scaler.inverse_transform(scaled_lap_times).flatten()

# Define the LSTM-based prediction model
class F1PredictionModel(nn.Module):
    def __init__(self, sequence_dim: int, static_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout_prob: float = 0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=sequence_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob
        )

        self.static_network = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        self.final_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            sequence (torch.Tensor): Sequence input of shape (batch_size, window_size, sequence_dim).
            static (torch.Tensor): Static features input of shape (batch_size, static_dim).

        Returns:
            torch.Tensor: Predicted lap time.
        """
        lstm_out, _ = self.lstm(sequence)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step
        static_out = self.static_network(static)
        combined = torch.cat([lstm_out, static_out], dim=1)
        prediction = self.final_network(combined)
        return prediction.squeeze()

# Define the training function
def train_model(model: nn.Module, 
                train_loader: DataLoader,
                val_loader: DataLoader,
                epochs: int = 50,
                learning_rate: float = 0.001,
                patience: int = 10,      # Added for early stopping
                device: Optional[str] = None) -> Dict[str, List[float]]:
    """
    Trains the LSTM model.

    Args:
        model (nn.Module): The LSTM model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        epochs (int, optional): Number of training epochs. Defaults to 50.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        patience (int, optional): Patience for early stopping. Defaults to 10.
        device (str, optional): Device to train on ('cuda' or 'cpu'). Defaults to None.

    Returns:
        Dict[str, List[float]]: Dictionary containing training and validation loss history.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history = {'train_loss': [], 'val_loss': []}
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for batch in train_loader:
            sequences = batch['sequence'].to(device)
            static = batch['static'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            predictions = model(sequences, static)
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(device)
                static = batch['static'].to(device)
                targets = batch['target'].to(device)
                
                predictions = model(sequences, static)
                loss = criterion(predictions, targets)
                val_losses.append(loss.item())
        
        # Record metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'models/best_lstm_model.pth')
            logging.info(f"Best model saved at epoch {epoch+1} with validation loss {val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

    return history

# Define utility functions to save and load the model with preprocessor
def save_model_with_preprocessor(model: nn.Module, preprocessor: F1DataPreprocessor, path: str):
    """
    Saves the model and preprocessor to a file.

    Args:
        model (nn.Module): The trained model.
        preprocessor (F1DataPreprocessor): The data preprocessor with fitted scalers.
        path (str): File path to save the model and preprocessor.
    """
    model.eval()
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'sequence_dim': model.lstm.input_size,
            'static_dim': model.static_network[0].in_features,
            'hidden_dim': model.lstm.hidden_size,
            'num_layers': model.lstm.num_layers,
            'dropout_prob': model.lstm.dropout
        },
        'preprocessor_state': {
            'lap_time_scaler': preprocessor.lap_time_scaler,
            'dynamic_scaler': preprocessor.dynamic_scaler, 
            'static_scaler': preprocessor.static_scaler,
            'static_feature_names': preprocessor.static_feature_names  # Save static feature names
        }
    }, path)
    logging.info(f"Model and preprocessor saved to {path}")

def load_model_with_preprocessor(path: str) -> Tuple[nn.Module, F1DataPreprocessor]:
    """
    Loads the model and preprocessor from a file.

    Args:
        path (str): File path from where to load the model and preprocessor.

    Returns:
        Tuple[nn.Module, F1DataPreprocessor]: The loaded model and preprocessor.
    """
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    
    model = F1PredictionModel(
        sequence_dim=checkpoint['model_config']['sequence_dim'],
        static_dim=checkpoint['model_config']['static_dim'],
        hidden_dim=checkpoint['model_config']['hidden_dim'],
        num_layers=checkpoint['model_config']['num_layers'],
        dropout_prob=checkpoint['model_config']['dropout_prob']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    preprocessor = F1DataPreprocessor()
    preprocessor.lap_time_scaler = checkpoint['preprocessor_state']['lap_time_scaler']
    preprocessor.dynamic_scaler = checkpoint['preprocessor_state']['dynamic_scaler']
    preprocessor.static_scaler = checkpoint['preprocessor_state']['static_scaler']
    preprocessor.static_feature_names = checkpoint['preprocessor_state']['static_feature_names']  # Load static feature names
    
    logging.info(f"Model and preprocessor loaded from {path}")
    return model, preprocessor
