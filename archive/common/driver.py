# common/driver.py

from typing import Dict, List, Tuple
import numpy as np
from common.features import RaceFeatures

class Driver:
    def __init__(self, driver_id: int,
                 name: str, static_features: np.ndarray,
                 initial_dynamic_features: dict,
                 start_position: int,
                 pit_strategy: list,
                 starting_compound: int,
                 constructor_id: int = -1):
        """
        Driver object holding race information.
        """
        self.driver_id = driver_id
        self.name = name
        self.static_features = static_features
        self.dynamic_features = initial_dynamic_features.copy()
        self.current_position = start_position
        self.start_position = start_position
        self.pit_strategy = pit_strategy
        self.starting_compound = starting_compound
        self.constructor_id = constructor_id

        self.cumulative_race_time = 0.0  # ms
        self.lap_times = []
        self.race_distance = 0.0

        # For now, just create a dummy sequence
        self.sequence = np.zeros((3, len(RaceFeatures().dynamic_features)+1))

    def update_race_progress(self, lap_time: float, circuit_length: float):
        self.lap_times.append(lap_time)
        self.cumulative_race_time += lap_time
        self.race_distance += circuit_length
        self.dynamic_features['cumulative_race_time'] = self.cumulative_race_time