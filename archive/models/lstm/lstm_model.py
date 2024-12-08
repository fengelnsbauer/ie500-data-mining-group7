import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Optional
import logging
from common.features import RaceFeatures

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

class F1DataPreprocessor:
    def __init__(self):
        self.static_scaler = StandardScaler()
        self.dynamic_scaler = StandardScaler()
        self.lap_time_scaler = StandardScaler()
        self.race_features = RaceFeatures()
        self.static_feature_names = self.race_features.static_features
        self.window_size = 3

    def fit_scalers(self, sequences: np.ndarray, static_features: np.ndarray, targets: np.ndarray):
        # Fit on training data
        self.lap_time_scaler.fit(targets.reshape(-1, 1))
        self.static_scaler.fit(static_features)
        # dynamic features start from index 1 (since 0 is lap_time)
        dynamic_features_flat = sequences[:, :, 1:].reshape(-1, sequences.shape[2]-1)
        self.dynamic_scaler.fit(dynamic_features_flat)

    def transform_static_features(self, static_features: np.ndarray) -> np.ndarray:
        if static_features.ndim == 1:
            static_features = static_features.reshape(1, -1)
        return self.static_scaler.transform(static_features)

    def transform_data(self, sequences: np.ndarray, static_features: np.ndarray, targets: np.ndarray):
        # Targets
        targets_scaled = self.lap_time_scaler.transform(targets.reshape(-1, 1)).flatten()
        # Static
        static_features_scaled = self.transform_static_features(static_features)

        # Dynamic
        # sequences: shape (N, window_size, feature_dim)
        # feature_dim = 1 (lap_time) + len(dynamic_features)
        # Index 0 of sequences is currently unknown. We must first define that we have sequences with actual lap times?
        # Assume sequences[:, :, 0] will be actual lap times. We'll transform them.
        # Actually, we do not have lap times in sequences yet. If we want to predict the last lap in the sequence:
        # The target is the last lap_time. The sequences should not contain the target lap_time to avoid leakage.
        # But here, we do have the actual lap time in the sequence as a dynamic feature. Let's assume sequences[:, :, 0] is lap_time from the data.
        
        lap_times = sequences[:, :, 0].reshape(-1, 1)
        lap_times_scaled = self.lap_time_scaler.transform(lap_times).reshape(sequences.shape[0], sequences.shape[1], 1)

        dynamic_raw = sequences[:, :, 1:].reshape(-1, sequences.shape[2] - 1)
        dynamic_scaled = self.dynamic_scaler.transform(dynamic_raw).reshape(sequences.shape[0], sequences.shape[1], -1)

        sequences_scaled = np.concatenate([lap_times_scaled, dynamic_scaled], axis=2)

        return sequences_scaled, static_features_scaled, targets_scaled

    def inverse_transform_lap_times(self, scaled_lap_times: np.ndarray) -> np.ndarray:
        logging.debug(f"Pre-inverse transform: {scaled_lap_times}")
        result = self.lap_time_scaler.inverse_transform(scaled_lap_times.reshape(-1, 1)).flatten()
        logging.debug(f"Post-inverse transform: {result}")
        return result

class F1PredictionModel(nn.Module):
    def __init__(self, sequence_dim: int, static_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout_prob: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=sequence_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        self.static_network = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, sequence: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        # sequence: (batch, window_size, sequence_dim)
        # static: (batch, static_dim)
        lstm_out, _ = self.lstm(sequence)
        lstm_features = lstm_out[:, -1, :]  # last timestep
        static_features = self.static_network(static)
        combined = torch.cat([lstm_features, static_features], dim=1)
        return self.output_network(combined)

def train_model(model: nn.Module, 
                train_loader,
                val_loader,
                epochs=50,
                learning_rate=0.001,
                patience=10,
                device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            sequences = batch['sequence'].to(device)
            static = batch['static'].to(device)
            targets = batch['target'].to(device)
            optimizer.zero_grad()
            predictions = model(sequences, static).squeeze(-1)
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(device)
                static = batch['static'].to(device)
                targets = batch['target'].to(device)
                predictions = model(sequences, static).squeeze(-1)
                loss = criterion(predictions, targets)
                val_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        logging.info(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()
            patience_counter = 0
            logging.info(f"New best model at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f'Early stopping at epoch {epoch+1}')
                break
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    return history

def save_model_with_preprocessor(model: nn.Module, preprocessor: F1DataPreprocessor, path: str):
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
        'preprocessor_state': preprocessor
    }, path)
    logging.info(f"Model and preprocessor saved to {path}")

def load_model_with_preprocessor(path: str):
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
    preprocessor = checkpoint['preprocessor_state']
    logging.info(f"Model and preprocessor loaded from {path}")
    return model, preprocessor
