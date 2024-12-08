import torch
import numpy as np
from typing import Dict
from common.base_simulator import BaseRaceSimulator
from common.race import Race
from common.driver import Driver

class LSTMRaceSimulator(BaseRaceSimulator):
    def __init__(self, model_path: str, sequence_length: int = 5):
        super().__init__()
        self.model = torch.load(model_path)
        self.model.eval()
        self.sequence_length = sequence_length
        self.feature_buffer = {}  # Store sequences for each driver
        
    def initialize_feature_buffer(self, driver: Driver):
        """Initialize the feature buffer for a driver with their first state."""
        driver_id = driver.driver_id
        if driver_id not in self.feature_buffer:
            # Create initial sequence buffer
            self.feature_buffer[driver_id] = {
                'static_features': None,
                'dynamic_features': []
            }
            
            # Set static features
            all_features = self.prepare_features(driver)
            static_indices = [i for i, feat in enumerate(self.model.static_features) 
                            if feat in driver.static_features]
            self.feature_buffer[driver_id]['static_features'] = all_features[static_indices]
    
    def prepare_features(self, driver: Driver) -> np.ndarray:
        """Prepare features in the correct order for the model."""
        features = []
        all_features = self.model.static_features + self.model.dynamic_features
        
        for feature in all_features:
            if feature in driver.static_features:
                features.append(driver.static_features[feature])
            elif feature in driver.dynamic_features:
                features.append(driver.dynamic_features[feature])
            else:
                features.append(0.0)  # Default value for missing features
                
        return np.array(features)
    
    def update_feature_buffer(self, driver: Driver):
        """Update the feature buffer with the latest driver state."""
        driver_id = driver.driver_id
        all_features = self.prepare_features(driver)
        
        # Update dynamic features
        dynamic_indices = [i for i, feat in enumerate(self.model.dynamic_features)]
        dynamic_features = all_features[dynamic_indices]
        
        self.feature_buffer[driver_id]['dynamic_features'].append(dynamic_features)
        
        # Keep only the last sequence_length features
        if len(self.feature_buffer[driver_id]['dynamic_features']) > self.sequence_length:
            self.feature_buffer[driver_id]['dynamic_features'].pop(0)
    
    def get_model_input(self, driver: Driver) -> Dict[str, torch.Tensor]:
        """Prepare input tensors for the LSTM model."""
        driver_id = driver.driver_id
        buffer = self.feature_buffer[driver_id]
        
        # If we don't have enough history, duplicate the first state
        while len(buffer['dynamic_features']) < self.sequence_length:
            buffer['dynamic_features'].append(buffer['dynamic_features'][0])
        
        return {
            'dynamic_sequence': torch.FloatTensor(buffer['dynamic_features']).unsqueeze(0),
            'static_features': torch.FloatTensor(buffer['static_features']).unsqueeze(0)
        }
    
    def simulate_driver_lap(self, driver: Driver, lap: int, race: Race) -> float:
        """Simulate a single lap using the LSTM model."""
        # Initialize or update feature buffer
        if lap == 1:
            self.initialize_feature_buffer(driver)
        self.update_feature_buffer(driver)
        
        # Prepare model input
        model_input = self.get_model_input(driver)
        
        # Get prediction from model
        with torch.no_grad():
            prediction = self.model(
                model_input['dynamic_sequence'],
                model_input['static_features']
            )
            lap_time = prediction.item()
        
        # Adjust lap time for pit stops
        if driver.dynamic_features.get('is_pit_lap', 0):
            pit_stop_time = driver.dynamic_features.get('pitstop_milliseconds', 
                                                      self.pit_stop_duration)
            lap_time += pit_stop_time
        
        return lap_time