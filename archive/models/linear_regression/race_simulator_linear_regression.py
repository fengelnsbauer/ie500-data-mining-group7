# models/linear_regression/race_simulator_linear_regression.py

import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from typing import Dict, List
from common.base_simulator import BaseRaceSimulator
from common.driver import Driver
from common.race import Race

class LinearRegressionRaceSimulator(BaseRaceSimulator):
    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self.pipeline = pipeline

    def simulate_driver_lap(self, driver: Driver, lap: int, race: Race) -> float:
        # Prepare features for prediction
        features = self.prepare_features(driver, lap, race)

        # Predict lap time using the pipeline
        lap_time = self.pipeline.predict(features)[0]

        # Apply adjustments for safety car and pit stops
        if driver.dynamic_features['TrackStatus'] == 4:
            lap_time *= 1.1  # Increase lap time by 10% under safety car
        if driver.dynamic_features['is_pit_lap'] == 1:
            lap_time += self.pit_stop_duration  # Add pit stop duration

        return lap_time

    def prepare_features(self, driver: Driver, lap: int, race: Race) -> pd.DataFrame:
        # Combine static and dynamic features
        features = {
            **driver.static_features,
            **driver.dynamic_features,
            'raceId': race.race_id,
            'driverId': driver.driver_id,
            'lap': lap
        }

        # Convert to DataFrame
        driver_df = pd.DataFrame([features])

        # Ensure all necessary columns are present
        preprocessor = self.pipeline.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()
        for col in feature_names:
            if col not in driver_df.columns:
                driver_df[col] = 0  # Fill missing features with zeros

        return driver_df
