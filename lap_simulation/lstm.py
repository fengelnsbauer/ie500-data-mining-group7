import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from features import RaceFeatures

# Define the F1Dataset class
class F1Dataset(Dataset):
    def __init__(self, sequences, static_features, targets):
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

# Define the F1DataPreprocessor class
from sklearn.preprocessing import StandardScaler
import numpy as np

class F1DataPreprocessor:
    def __init__(self):
        self.static_scaler = StandardScaler()
        self.dynamic_scaler = StandardScaler()
        self.lap_time_scaler = StandardScaler()
        self.race_features = RaceFeatures()

    def fit_scalers(self, sequences, static_features, targets):
        """
        Fit scalers on training data.
        - Lap times scaler is fit on targets.
        - Static features scaler is fit on static_features.
        - Dynamic features scaler is fit on dynamic features (excluding 'tire_compound').
        """
        # Fit lap_time_scaler on targets
        self.lap_time_scaler.fit(targets.reshape(-1, 1))
        
        # Fit static_scaler on static features
        self.static_scaler.fit(static_features)

        # Extract dynamic features (excluding the lap time column)
        dynamic_features_flat = sequences[:, :, 1:].reshape(-1, sequences.shape[2] - 1)
        self.dynamic_scaler.fit(dynamic_features_flat)

    def transform_data(self, sequences, static_features, targets):
        """
        Transform data using fitted scalers.
        - Scales lap times, static features, and dynamic features.
        """
        # Scale targets (lap times)
        targets_scaled = self.lap_time_scaler.transform(targets.reshape(-1, 1)).flatten()

        # Scale static features
        static_features_scaled = self.static_scaler.transform(static_features)

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



class F1PredictionModel(nn.Module):
    def __init__(self, sequence_dim, static_dim, hidden_dim=128, num_layers=2, dropout_prob=0.2):
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

    def forward(self, sequence, static):
        # Process sequence through LSTM
        lstm_out, _ = self.lstm(sequence)
        lstm_out = lstm_out[:, -1, :]  # Output of the last time step

        # Process static features directly
        static_out = self.static_network(static)

        # Combine LSTM output and static features
        combined = torch.cat([lstm_out, static_out], dim=1)

        # Final prediction
        prediction = self.final_network(combined)

        return prediction.squeeze()


# Define the training function
def train_model(model: nn.Module, 
                train_loader: DataLoader,
                val_loader: DataLoader,
                epochs: int = 10,
                learning_rate: float = 0.001,
                device: Optional[str] = None) -> Dict[str, List[float]]:
    """
    Train the model and return training history
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history = {'train_loss': [], 'val_loss': []}
    
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
    
    return history

# Define a function to save the model
def save_model_with_preprocessor(model, preprocessor, sequence_dim, static_dim, path: str):
    torch.save({
        'model_state_dict': model.state_dict(),
        'lap_time_scaler': preprocessor.lap_time_scaler,
        'dynamic_scaler': preprocessor.dynamic_scaler, 
        'static_scaler': preprocessor.static_scaler,
        'sequence_dim': sequence_dim,
        'static_dim': static_dim
    }, path)
    print(f"Model and preprocessor saved to {path}")

def load_model_with_preprocessor(path: str) -> Tuple[F1PredictionModel, F1DataPreprocessor]:
    """Load saved model and preprocessor"""
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    
    model = F1PredictionModel(
        sequence_dim=checkpoint['sequence_dim'], 
        static_dim=checkpoint['static_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    preprocessor = F1DataPreprocessor()
    preprocessor.lap_time_scaler = checkpoint['lap_time_scaler']
    preprocessor.dynamic_scaler = checkpoint['dynamic_scaler']
    preprocessor.static_scaler = checkpoint['static_scaler']
    
    return model, preprocessor