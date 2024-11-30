import optuna
from torch.utils.data import DataLoader
from lstm import F1PredictionModel, train_model

def optimize_hyperparameters(train_dataset, val_dataset, sequence_dim, static_dim, n_trials=50):
    def objective(trial):
        # Parameter definitions
        params = {
            'hidden_dim': trial.suggest_int('hidden_dim', 32, 256),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'dropout_prob': trial.suggest_float('dropout_prob', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        }
        
        # Create and train model
        model = F1PredictionModel(
            sequence_dim=sequence_dim,
            static_dim=static_dim,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            dropout_prob=params['dropout_prob']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            shuffle=False
        )
        
        history = train_model(
            model,
            train_loader,
            val_loader,
            epochs=10,
            learning_rate=params['learning_rate']
        )
        
        return min(history['val_loss'])  # Return best validation loss
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params