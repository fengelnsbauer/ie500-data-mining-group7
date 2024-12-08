# models/lstm/lstm_utils.py

import torch
import logging
from typing import Tuple
import pickle  # For saving preprocessor

def save_model_with_preprocessor(model: torch.nn.Module, preprocessor: 'F1DataPreprocessor', path: str):
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
        'preprocessor_state': preprocessor  # Assuming preprocessor is picklable
    }, path)
    logging.info(f"Model and preprocessor saved to {path}")

def load_model_with_preprocessor(path: str) -> Tuple[torch.nn.Module, 'F1DataPreprocessor']:
    """
    Loads the model and preprocessor from a file.

    Args:
        path (str): File path from where to load the model and preprocessor.

    Returns:
        Tuple[nn.Module, F1DataPreprocessor]: The loaded model and preprocessor.
    """
    checkpoint = torch.load(path, map_location=torch.device('cpu'))

    from lstm import F1PredictionModel  # Import here to avoid circular import

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
