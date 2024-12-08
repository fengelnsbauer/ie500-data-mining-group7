# models/lstm/race_simulator_lstm.py

import pandas as pd
import numpy as np
import torch
import logging
from typing import Dict
from common.base_simulator import BaseRaceSimulator
from common.driver import Driver
from common.race import Race
from common.features import RaceFeatures
from models.lstm.lstm_model import F1PredictionModel, F1DataPreprocessor  # Ensure correct import paths
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def log_simulation_inputs(lap: int, driver_id: int, sequence_tensor: torch.Tensor, static_tensor: torch.Tensor):
    seq_np = sequence_tensor.cpu().numpy()
    static_np = static_tensor.cpu().numpy()
    logging.debug(f"Lap {lap}, Driver {driver_id} input sequence: {seq_np}")
    logging.debug(f"Lap {lap}, Driver {driver_id} static features: {static_np}")

def log_dynamic_features(lap: int, driver_id: int, dynamic_features: Dict[str, float]):
    logging.debug(f"Lap {lap}, Driver {driver_id} dynamic features after update: {dynamic_features}")

def get_weather_from_cumulative_time(race_id: int, cumulative_ms: float, weather_df):
    # Unchanged logic to fetch nearest weather
    race_weather = weather_df[weather_df['raceId'] == race_id].sort_values('cumulative_milliseconds')
    if race_weather.empty:
        return {'TrackTemp': 35.0, 'AirTemp': 25.0, 'Humidity': 50.0}
    times = race_weather['cumulative_milliseconds'].values
    idx = np.searchsorted(times, cumulative_ms)
    if idx == 0:
        row = race_weather.iloc[0]
    elif idx >= len(times):
        row = race_weather.iloc[-1]
    else:
        prev_diff = abs(cumulative_ms - times[idx - 1])
        next_diff = abs(times[idx] - cumulative_ms)
        row = race_weather.iloc[idx - 1] if prev_diff < next_diff else race_weather.iloc[idx]
    return {
        'TrackTemp': row['TrackTemp'],
        'AirTemp': row['AirTemp'],
        'Humidity': row['Humidity']
    }

class LSTMRaceSimulator(BaseRaceSimulator):
    def __init__(self, model: torch.nn.Module, preprocessor: 'F1DataPreprocessor', weather_df=None):
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor
        self.race_features = RaceFeatures()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.window_size = preprocessor.window_size
        self.sequence_dim = len(self.race_features.dynamic_features) + 1
        self.weather_df = weather_df
        # Add storage for inputs
        self.input_data = []

    def simulate_driver_lap(self, driver: Driver, lap: int, race: Race) -> float:
        # Prepare input
        sequence_tensor, static_tensor = self.prepare_tensors(driver)
        
        # Store input data
        sequence_np = sequence_tensor.cpu().numpy()[0]  # Remove batch dimension
        static_np = static_tensor.cpu().numpy()[0]
        
        # Store all input data
        self.input_data.append({
            'lap': lap,
            'driver_id': driver.driver_id,
            'position': driver.current_position,
            'cumulative_time': driver.cumulative_race_time,
            'is_pit_lap': driver.dynamic_features['is_pit_lap'],
            'tire_age': driver.dynamic_features['tire_age'],
            'tire_compound': driver.dynamic_features['tire_compound'],
            'fuel_load': driver.dynamic_features['fuel_load'],
            'track_temp': driver.dynamic_features['TrackTemp'],
            'air_temp': driver.dynamic_features['AirTemp'],
            'humidity': driver.dynamic_features['Humidity'],
            'track_status': driver.dynamic_features['TrackStatus'],
            'gap_to_leader': driver.dynamic_features.get('GapToLeader_ms', 0),
            'interval_ahead': driver.dynamic_features.get('IntervalToPositionAhead_ms', 0),
            # Add sequence data
            'sequence_lap_times': sequence_np[:, 0].tolist(),  # First column is lap times
            'sequence_features': sequence_np[:, 1:].tolist(),  # Rest are dynamic features
            # Add static features
            'static_features': static_np.tolist()
        })

        # Regular prediction logic
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(sequence_tensor, static_tensor)
            lap_time_normalized = prediction.item()

        lap_time = self.preprocessor.inverse_transform_lap_times(np.array([lap_time_normalized]))[0]

        if driver.dynamic_features['TrackStatus'] == 4:
            lap_time *= 1.1
        if driver.dynamic_features['is_pit_lap'] == 1:
            lap_time += self.pit_stop_duration

        driver.update_race_progress(lap_time, race.circuit_length)
        
        # Store the prediction
        self.input_data[-1]['predicted_lap_time'] = lap_time

        return lap_time

    def get_input_data_df(self) -> pd.DataFrame:
        """Convert stored input data to a DataFrame for analysis."""
        df = pd.DataFrame(self.input_data)
        
        # Add column names for sequence features
        n_sequence_steps = len(df['sequence_features'].iloc[0])
        n_features = len(df['sequence_features'].iloc[0][0])
        
        # Expand sequence features into separate columns
        for step in range(n_sequence_steps):
            for feat in range(n_features):
                col_name = f'seq_step{step+1}_feat{feat+1}'
                df[col_name] = df['sequence_features'].apply(lambda x: x[step][feat])
                
        # Expand static features into separate columns
        n_static = len(df['static_features'].iloc[0])
        for i in range(n_static):
            df[f'static_feat{i+1}'] = df['static_features'].apply(lambda x: x[i])
            
        # Expand sequence lap times into separate columns
        for step in range(len(df['sequence_lap_times'].iloc[0])):
            df[f'seq_laptime{step+1}'] = df['sequence_lap_times'].apply(lambda x: x[step])
            
        # Drop the list columns since we've expanded them
        df = df.drop(columns=['sequence_features', 'static_features', 'sequence_lap_times'])
        
        return df

    def clear_input_data(self):
        """Clear stored input data."""
        self.input_data = []

    def prepare_tensors(self, driver: Driver):
        logging.debug(f"Raw sequence before scaling:\n{driver.sequence}")
        sequences_scaled, static_features_scaled, _ = self.preprocessor.transform_data(
            sequences=np.array([driver.sequence]),
            static_features=driver.static_features.reshape(1, -1),
            targets=np.array([0])
        )
        logging.debug(f"Scaled sequence:\n{sequences_scaled}")
        sequence_tensor = torch.FloatTensor(sequences_scaled).to(self.device)
        static_tensor = torch.FloatTensor(static_features_scaled).to(self.device)
        return sequence_tensor, static_tensor

    def update_driver_sequence(self, driver: Driver, lap_time: float):
        # After positions & gaps are updated in simulate_race, we call this method
        old_sequence = driver.sequence.copy()
        # Roll sequence
        driver.sequence = np.roll(driver.sequence, -1, axis=0)
        logging.debug(f"Sequence update:\nOld:\n{old_sequence}\nNew:\n{driver.sequence}")
        lap_time_scaled = self.preprocessor.lap_time_scaler.transform([[lap_time]])[0][0]

        # Retrieve up-to-date dynamic features (including GapToLeader_ms, track_position)
        dynamic_values = [driver.dynamic_features[f] for f in self.race_features.dynamic_features]
        dynamic_features_scaled = self.preprocessor.dynamic_scaler.transform(
            np.array(dynamic_values).reshape(1, -1)
        ).flatten()

        new_entry = np.concatenate(([lap_time_scaled], dynamic_features_scaled))

        driver.sequence[-1] = new_entry
        logging.debug(f"Driver {driver.driver_id} sequence updated with lap_time={lap_time}: {new_entry}")

    def update_dynamic_features(self, driver: Driver, lap: int, race: Race):
        super().update_dynamic_features(driver, lap, race)
        # Incorporate weather conditions from cumulative race time
        cumulative_time = driver.dynamic_features.get('cumulative_race_time', 0.0)
        lap_weather = get_weather_from_cumulative_time(race.race_id, cumulative_time, self.weather_df)
        driver.dynamic_features['TrackTemp'] = lap_weather['TrackTemp']
        driver.dynamic_features['AirTemp'] = lap_weather['AirTemp']
        driver.dynamic_features['Humidity'] = lap_weather['Humidity']
        log_dynamic_features(lap, driver.driver_id, driver.dynamic_features)
