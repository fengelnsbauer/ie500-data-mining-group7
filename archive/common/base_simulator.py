# common/base_simulator.py

from common.race import Race
from common.driver import Driver
import pandas as pd
import numpy as np

class BaseRaceSimulator:
    def __init__(self):
        self.pit_stop_duration = 20000.0  # 20 seconds in ms

    def simulate_race(self, race: Race):
        # Clear lap_data
        for driver in race.drivers:
            race.lap_data[driver.driver_id] = {
                'predicted_lap_times': [],
                'positions': [],
                'inputs': []
            }

        for lap in range(1, race.total_laps + 1):
            # Update dynamic features
            for driver in race.drivers:
                self.update_dynamic_features(driver, lap, race)

            # Predict lap times
            lap_times = {}
            for driver in race.drivers:
                lap_time = self.simulate_driver_lap(driver, lap, race)
                lap_times[driver.driver_id] = lap_time
                race.lap_data[driver.driver_id]['predicted_lap_times'].append(lap_time)
                race.lap_data[driver.driver_id]['inputs'].append({
                    'lap': lap,
                    'driver_id': driver.driver_id,
                    'driver_name': driver.name,
                    'static_features': driver.static_features.copy(),
                    'dynamic_features': driver.dynamic_features.copy(),
                })

            # Update positions
            self.update_positions(race)

            # Update sequences
            for driver in race.drivers:
                self.update_driver_sequence(driver, lap_times[driver.driver_id])

        return race.lap_data

    def update_dynamic_features(self, driver: Driver, lap: int, race: Race):
        # Update tire age
        driver.dynamic_features['tire_age'] += 1

        # Fuel decreases as laps progress (simple linear model)
        driver.dynamic_features['fuel_load'] = max(0, 100 - ((lap-1) / race.total_laps) * 100)

        # Pit stops if scheduled
        pitted = False
        for pit_lap, new_compound in driver.pit_strategy:
            if pit_lap == lap:
                driver.dynamic_features['is_pit_lap'] = 1
                driver.dynamic_features['tire_age'] = 0  # Reset tire age after pit
                driver.dynamic_features['tire_compound'] = new_compound
                pitted = True
                break
        if not pitted:
            driver.dynamic_features['is_pit_lap'] = 0

        driver.dynamic_features['track_position'] = driver.current_position

        # For simplicity, keep weather static here (LSTMRaceSimulator may override)
        driver.dynamic_features['TrackTemp'] = 35.0
        random_factor = np.random.normal(1, 0.01)  # Small random variation
        driver.dynamic_features['TrackTemp'] *= random_factor
        driver.dynamic_features['AirTemp'] = 25.0
        driver.dynamic_features['Humidity'] = 50.0

        # TrackStatus: Assume always green (1)
        driver.dynamic_features['TrackStatus'] = 1

    def simulate_driver_lap(self, driver: Driver, lap: int, race: Race) -> float:
        # Simple model: base lap time ~90s (90000ms) + variations
        base_lap_time = 90000.0
        # Adjust for tire age: +0.2% per lap
        tire_effect = 1 + driver.dynamic_features['tire_age'] * 0.002
        # Adjust for fuel: heavier at start means slower
        fuel_effect = 1 + (driver.dynamic_features['fuel_load'] * 0.0005)
        # random noise
        random_variation = np.random.normal(1, 0.005)

        lap_time = base_lap_time * tire_effect * fuel_effect * random_variation

        # Pit stop penalty
        if driver.dynamic_features['is_pit_lap'] == 1:
            lap_time += self.pit_stop_duration

        # Update driver progress
        driver.update_race_progress(lap_time, race.circuit_length)

        return lap_time

    def update_positions(self, race: Race):
        # Sort drivers by cumulative race time (lowest first)
        race.drivers.sort(key=lambda d: d.cumulative_race_time)
        for pos, d in enumerate(race.drivers, start=1):
            d.current_position = pos
            race.lap_data[d.driver_id]['positions'].append(pos)

        # Update gaps after assigning positions
        self.update_gaps(race.drivers)

    def update_gaps(self, drivers):
        if not drivers:
            return
        leader_time = drivers[0].cumulative_race_time
        for i, d in enumerate(drivers):
            d.dynamic_features['GapToLeader_ms'] = d.cumulative_race_time - leader_time
            if i == 0:
                d.dynamic_features['IntervalToPositionAhead_ms'] = 0.0
            else:
                d.dynamic_features['IntervalToPositionAhead_ms'] = (
                    d.cumulative_race_time - drivers[i-1].cumulative_race_time
                )

    def collect_lap_data(self, race: Race):
        # Build a dictionary or DataFrame with lap data
        data = []
        for d in race.drivers:
            for lap_idx, lap_time in enumerate(d.lap_times, start=1):
                data.append({
                    'raceId': race.race_id,
                    'driverId': d.driver_id,
                    'lap': lap_idx,
                    'lap_time': lap_time,
                    'position': d.current_position
                })
        return pd.DataFrame(data)

    def update_driver_sequence(self, driver: Driver, lap_time: float):
        # This is a placeholder. The actual LSTMRaceSimulator class overrides this method.
        pass
