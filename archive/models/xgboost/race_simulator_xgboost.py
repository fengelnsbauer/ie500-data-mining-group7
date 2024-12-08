# models/xgboost/race_simulator_xgboost.py

from common.base_simulator import BaseRaceSimulator
from common.driver import Driver
from common.race import Race
from common.features import RaceFeatures
import numpy as np
import pandas as pd
import xgboost as xgb
import logging
from typing import Dict

def get_weather_from_cumulative_time(race_id: int, cumulative_ms: float, weather_df):
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

class XGBoostRaceSimulator(BaseRaceSimulator):
    def __init__(self, model: xgb.Booster, preprocessor: 'F1DataPreprocessor', weather_df=None):
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor
        self.race_features = RaceFeatures()
        self.weather_df = weather_df
        self.input_data = []

    def prepare_features(self, driver: Driver, lap: int) -> np.ndarray:
        # Combine static and dynamic features
        feature_dict = {}
        
        # Add static features
        for i, feature_name in enumerate(self.race_features.static_features):
            feature_dict[feature_name] = driver.static_features[i]
            
        # Add dynamic features
        for feature_name in self.race_features.dynamic_features:
            feature_dict[feature_name] = driver.dynamic_features[feature_name]
            
        # Convert to array in correct order
        features = np.array([[
            feature_dict[feature] 
            for feature in self.race_features.static_features + self.race_features.dynamic_features
        ]])
        
        return features

    def simulate_driver_lap(self, driver: Driver, lap: int, race: Race) -> float:
        # Prepare input features
        features = self.prepare_features(driver, lap)
        
        # Store input data
        input_data = {
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
            'features': features.tolist()[0]
        }
        self.input_data.append(input_data)

        # Scale features
        features_scaled = self.preprocessor.transform_data(features)
        
        # Create DMatrix for prediction
        dmatrix = xgb.DMatrix(features_scaled)
        
        # Make prediction
        lap_time_normalized = self.model.predict(dmatrix)[0]
        
        # Inverse transform prediction
        lap_time = self.preprocessor.inverse_transform_lap_times(np.array([lap_time_normalized]))[0]

        # Apply adjustments
        if driver.dynamic_features['TrackStatus'] == 4:  # Safety car
            lap_time *= 1.1
        if driver.dynamic_features['is_pit_lap'] == 1:
            lap_time += self.pit_stop_duration

        # Update driver progress
        driver.update_race_progress(lap_time, race.circuit_length)
        
        # Store the prediction
        self.input_data[-1]['predicted_lap_time'] = lap_time

        return lap_time

    def update_dynamic_features(self, driver: Driver, lap: int, race: Race):
        super().update_dynamic_features(driver, lap, race)
        # Update weather conditions
        cumulative_time = driver.dynamic_features.get('cumulative_race_time', 0.0)
        lap_weather = get_weather_from_cumulative_time(race.race_id, cumulative_time, self.weather_df)
        driver.dynamic_features['TrackTemp'] = lap_weather['TrackTemp']
        driver.dynamic_features['AirTemp'] = lap_weather['AirTemp']
        driver.dynamic_features['Humidity'] = lap_weather['Humidity']

    def get_input_data_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.input_data)
        
        # Expand features into separate columns
        n_features = len(df['features'].iloc[0])
        feature_names = (
            self.race_features.static_features + 
            self.race_features.dynamic_features
        )
        
        for i, name in enumerate(feature_names):
            df[f'feature_{name}'] = df['features'].apply(lambda x: x[i])
            
        # Drop the list column
        df = df.drop(columns=['features'])
        
        return df

    def clear_input_data(self):
        self.input_data = []
