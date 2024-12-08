# models/xgboost/xgboost_model.py

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from typing import Tuple, Dict, Optional
from common.features import RaceFeatures
import logging

class F1DataPreprocessor:
    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.race_features = RaceFeatures()
        self.window_size = 3

    def fit_scalers(self, features: np.ndarray, targets: np.ndarray):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(targets.reshape(-1, 1))

    def transform_data(self, features: np.ndarray, targets: Optional[np.ndarray] = None):
        features_scaled = self.feature_scaler.transform(features)
        if targets is not None:
            targets_scaled = self.target_scaler.transform(targets.reshape(-1, 1)).flatten()
            return features_scaled, targets_scaled
        return features_scaled

    def inverse_transform_lap_times(self, scaled_lap_times: np.ndarray) -> np.ndarray:
        return self.target_scaler.inverse_transform(scaled_lap_times.reshape(-1, 1)).flatten()

def save_model_with_preprocessor(model: xgb.Booster, preprocessor: F1DataPreprocessor, path: str):
    with open(path, 'wb') as f:
        pickle.dump({
            'model': model,
            'preprocessor': preprocessor
        }, f)
    logging.info(f"Model and preprocessor saved to {path}")

def load_model_with_preprocessor(path: str) -> Tuple[xgb.Booster, F1DataPreprocessor]:
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    logging.info(f"Model and preprocessor loaded from {path}")
    return checkpoint['model'], checkpoint['preprocessor']
