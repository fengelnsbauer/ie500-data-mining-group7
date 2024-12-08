import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import optuna
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple
import logging

class F1LapTimePredictor:
    """Main class containing both the LSTM model and training functionality."""
    
    class LSTMModel(nn.Module):
        """LSTM model for lap time prediction."""
        def __init__(
            self,
            static_feature_size: int,
            dynamic_feature_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.2
        ):
            super().__init__()
            
            self.lstm = nn.LSTM(
                input_size=dynamic_feature_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            
            self.static_network = nn.Sequential(
                nn.Linear(static_feature_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )
            
            self.output_layer = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1)
            )
        
        def forward(self, dynamic_sequence: torch.Tensor, static_features: torch.Tensor) -> torch.Tensor:
            lstm_out, _ = self.lstm(dynamic_sequence)
            last_dynamic = lstm_out[:, -1, :]
            static_out = self.static_network(static_features)
            combined = torch.cat([last_dynamic, static_out], dim=1)
            return self.output_layer(combined)
    
    class SequenceDataset(Dataset):
        """Dataset class for handling F1 race sequences."""
        def __init__(
            self,
            features: np.ndarray,
            targets: np.ndarray,
            metadata: pd.DataFrame,
            static_feature_indices: List[int],
            sequence_length: int = 5
        ):
            self.features = features
            self.targets = targets
            self.static_feature_indices = static_feature_indices
            self.sequence_length = sequence_length
            
            # Sort metadata by raceId, driverId, and lap
            self.metadata = metadata.sort_values(['raceId', 'driverId', 'lap'])
            
            # Create valid sequences
            self.sequence_mapping = []
            
            # Process each race and driver group
            for (race_id, driver_id), group in self.metadata.groupby(['raceId', 'driverId']):
                # Get sorted indices
                group_indices = group.index.values
                
                # Only include groups with enough laps for a complete sequence
                if len(group_indices) > sequence_length:
                    for start_pos in range(len(group_indices) - sequence_length):
                        self.sequence_mapping.append({
                            'input_indices': group_indices[start_pos:start_pos + sequence_length],
                            'target_idx': group_indices[start_pos + sequence_length]
                        })
        
        def __len__(self):
            return len(self.sequence_mapping)
        
        def __getitem__(self, idx):
            sequence = self.sequence_mapping[idx]
            input_indices = sequence['input_indices']
            target_idx = sequence['target_idx']
            
            static_features = self.features[input_indices[0], self.static_feature_indices]
            dynamic_features = np.delete(self.features[input_indices], self.static_feature_indices, axis=1)
            
            return {
                'dynamic_sequence': torch.FloatTensor(dynamic_features),
                'static_features': torch.FloatTensor(static_features),
                'target': torch.FloatTensor([self.targets[target_idx]])
            }
    
    def __init__(self, processed_data: Dict, device: str = None):
        self.processed_data = processed_data
        self.static_features = processed_data['feature_info']['static_features']
        self.dynamic_features = processed_data['feature_info']['dynamic_features']
        
        # Automatically detect the device (MPS if available, otherwise CUDA or CPU)
        if device is None:
            self.device = (
                torch.device("mps") if torch.backends.mps.is_available()
                else torch.device("cuda") if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = torch.device(device)
        
        # Calculate static feature indices
        all_features = self.static_features + self.dynamic_features
        self.static_indices = [all_features.index(feat) for feat in self.static_features]

    
    def create_data_loaders(
        self,
        batch_size: int,
        sequence_length: int
    ) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders."""
        train_dataset = self.SequenceDataset(
            self.processed_data['train']['features'],
            self.processed_data['train']['targets'],
            self.processed_data['train']['metadata'],
            self.static_indices,
            sequence_length
        )
        
        val_dataset = self.SequenceDataset(
            self.processed_data['test']['features'],
            self.processed_data['test']['targets'],
            self.processed_data['test']['metadata'],
            self.static_indices,
            sequence_length
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=8  # default is 2, try increasing
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader

    def optimize(self, n_trials: int = 100) -> optuna.Study:
        """Optimize model hyperparameters using Optuna."""
        def objective(trial: optuna.Trial) -> float:
            # Define hyperparameters to optimize
            params = {
                'hidden_size': trial.suggest_int('hidden_size', 32, 256),
                'num_layers': trial.suggest_int('num_layers', 1, 4),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'sequence_length': trial.suggest_int('sequence_length', 3, 10),
                'batch_size': trial.suggest_int('batch_size', 16, 128),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
            }
            
            # Create model and data loaders
            model = self.LSTMModel(
                static_feature_size=len(self.static_features),
                dynamic_feature_size=len(self.dynamic_features),
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            ).to(self.device)

            print(f"Using device: {self.device}")
            print(f"Model is on correct device: {next(model.parameters()).device}")
            
            train_loader, val_loader = self.create_data_loaders(
                batch_size=params['batch_size'],
                sequence_length=params['sequence_length']
            )
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3
            )
            
            best_val_loss = float('inf')
            patience = 5
            patience_counter = 0
            
            for epoch in range(50):
                # Training phase
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    predictions = model(
                        batch['dynamic_sequence'].to(self.device),
                        batch['static_features'].to(self.device)
                    )
                    loss = criterion(predictions, batch['target'].to(self.device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                # Validation phase
                model.eval()
                val_losses = []
                val_predictions = []
                val_targets = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        predictions = model(
                            batch['dynamic_sequence'].to(self.device),
                            batch['static_features'].to(self.device)
                        )
                        targets = batch['target'].to(self.device)
                        loss = criterion(predictions, targets)
                        
                        val_losses.append(loss.item())
                        val_predictions.extend(predictions.cpu().numpy())
                        val_targets.extend(targets.cpu().numpy())
                
                val_loss = np.mean(val_losses)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    break
                
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.sqrt(mean_squared_error(val_targets, val_predictions))
        
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=3)
        )
        study.optimize(objective, n_trials=n_trials)
        return study
    
    def create_best_model(self, study: optuna.Study) -> LSTMModel:
        """Create a model with the best hyperparameters."""
        return self.LSTMModel(
            static_feature_size=len(self.static_features),
            dynamic_feature_size=len(self.dynamic_features),
            hidden_size=study.best_params["hidden_size"],
            num_layers=study.best_params["num_layers"],
            dropout=study.best_params["dropout"]
        )