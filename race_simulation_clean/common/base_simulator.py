# base_simulator.py
import datetime
from typing import Dict, List, Optional
from race import Race
from driver import Driver
from simulation_validator import SimulationValidator
from simulation_logger import SimulationLogger
from config import DEFAULT_WEATHER, PIT_STOP_DURATION

class BaseRaceSimulator:
    def __init__(self):
        self.pit_stop_duration = PIT_STOP_DURATION
        self.default_weather = DEFAULT_WEATHER
        self.logger = SimulationLogger()
        self.validator = SimulationValidator()

    def parse_static_features(self, race_data, driver_data, circuit_data, constructor_data):
        """Parses all static features needed for the simulation."""
        return {
            # Race Info
            'raceId': race_data['raceId'],
            'year': race_data['year'],
            'round': race_data['round'],
            'circuitId': race_data['circuitId'],
            'grid': driver_data['grid'],
            
            # Circuit Info
            'circuit_length': circuit_data['circuit_length'],
            'circuit_type_encoded': circuit_data['circuit_type_encoded'],
            'alt': circuit_data['alt'],
            'circuit_country': circuit_data['country'],
            'circuit_lat': circuit_data['lat'],
            'circuit_lng': circuit_data['lng'],
            
            # Driver Info
            'driver_overall_skill': driver_data['driver_overall_skill'],
            'driver_circuit_skill': driver_data['driver_circuit_skill'],
            'driver_consistency': driver_data['driver_consistency'],
            'driver_aggression': driver_data['driver_aggression'],
            'driver_reliability': driver_data['driver_reliability'],
            'driver_risk_taking': driver_data['driver_risk_taking'],
            'driver_adaptability': driver_data['driver_adaptability'],
            'driver_code': driver_data['code'],
            'driver_nationality': driver_data['nationality'],
            
            # Constructor Info
            'constructor_performance': constructor_data['constructor_performance'],
            'constructor_id': constructor_data['constructorId'],
            'constructor_nationality': constructor_data['constructor_nationality'],
            'constructor_position': constructor_data['constructor_position'],
            
            # Historical Performance
            'fp1_median_time': driver_data['fp1_median_time'],
            'fp2_median_time': driver_data['fp2_median_time'],
            'fp3_median_time': driver_data['fp3_median_time'],
            'quali_time': driver_data['quali_time']
        }


    def simulate_race(self, race: Race):
        if not self.validator.validate_race_initialization(race):
            raise ValueError("Race validation failed")
            
        self.initialize_race_data(race)
        
        for lap in range(1, race.total_laps + 1):
            lap_times = self.simulate_lap(race, lap)
            self.update_race_status(race, lap_times)
            
        self.logger.export_simulation_data(
            f'race_{race.race_id}_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        return race.lap_data

    def initialize_race_data(self, race: Race):
        """Initializes data storage for race simulation."""
        for driver in race.drivers:
            race.lap_data[driver.driver_id] = {
                'lap_times': [],
                'positions': []
            }

    def simulate_lap(self, race: Race, lap: int) -> Dict[int, float]:
        lap_times = {}
        for driver in race.drivers:
            if not self.validator.validate_driver_state(driver):
                raise ValueError(f"Driver {driver.driver_id} validation failed")
                
            self.update_dynamic_features(driver, lap, race)
            lap_time = self.simulate_driver_lap(driver, lap, race)
            
            if not self.validator.validate_lap_time(lap_time, driver.lap_times):
                raise ValueError(f"Invalid lap time for driver {driver.driver_id}")
                
            lap_times[driver.driver_id] = lap_time
            race.lap_data[driver.driver_id]['lap_times'].append(lap_time)
            driver.update_race_progress(lap_time, race.circuit_length)
            
            self.logger.log_lap_data(
                race.race_id,
                lap,
                driver.driver_id,
                driver.static_features,
                driver.dynamic_features,
                race.current_conditions,
                lap_time
            )
            
        return lap_times

    def update_race_status(self, race: Race, lap_times: Dict[int, float]):
        """Updates race positions and gaps after each lap."""
        self.update_positions(race)
        self.update_gaps(race.drivers)

    def simulate_driver_lap(self, driver: Driver, lap: int, race: Race) -> float:
        """To be implemented by specific model simulators."""
        raise NotImplementedError

    def update_dynamic_features(self, driver: Driver, lap: int, race: Race):
        """Updates all dynamic features for a driver during the race."""
        # Consolidated timing updates
        driver.dynamic_features.update({
            'lap': lap,
            'position': driver.current_position,
            'positionOrder': driver.current_position,
            'track_position': driver.current_position,
            'cumulative_milliseconds': driver.cumulative_race_time,
            'GapToLeader_ms': driver.gaptoLeader_ms,
            'IntervalToPositionAhead_ms': driver.intervaltopositionahead_ms
        })
        
        self.update_car_state(driver, lap, race)
        self.update_environmental(driver, lap, race)

    def update_car_state(self, driver: Driver, lap: int, race: Race):
        """Updates car-specific features."""
        # Tire Management
        driver.dynamic_features['tire_age'] += 1
        
        # Fuel Management
        driver.dynamic_features['fuel_load'] = max(0, 100 - ((lap-1) / race.total_laps) * 100)
        
        # Pit Stop Handling
        driver.dynamic_features['is_pit_lap'] = 0
        for pit_lap, new_compound in driver.pit_strategy:
            if pit_lap == lap:
                driver.dynamic_features.update({
                    'is_pit_lap': 1,
                    'tire_age': 0,
                    'tire_compound': new_compound
                })
                break

    def get_weather_from_cumulative_time(self, race_id: int, cumulative_time: float) -> Dict[str, float]:
        """
        Retrieves weather conditions for a specific point in the race.
        
        Args:
            race_id: Identifier for the race
            cumulative_time: Current race time in milliseconds
        
        Returns:
            Dictionary containing weather conditions
        """
        if not hasattr(self, 'weather_df') or self.weather_df is None:
            return self.default_weather
            
        race_weather = self.weather_df[
            self.weather_df['raceId'] == race_id
        ].sort_values('cumulative_milliseconds')
        
        if race_weather.empty:
            return self.default_weather
            
        # Find closest weather reading
        idx = race_weather['cumulative_milliseconds'].searchsorted(cumulative_time)
        if idx == 0:
            row = race_weather.iloc[0]
        elif idx >= len(race_weather):
            row = race_weather.iloc[-1]
        else:
            prev_diff = abs(cumulative_time - race_weather.iloc[idx-1]['cumulative_milliseconds'])
            next_diff = abs(race_weather.iloc[idx]['cumulative_milliseconds'] - cumulative_time)
            row = race_weather.iloc[idx-1] if prev_diff < next_diff else race_weather.iloc[idx]
        
        return {
            'TrackTemp': row['TrackTemp'],
            'AirTemp': row['AirTemp'],
            'Humidity': row['Humidity']
        }

    def update_environmental(self, driver: Driver, lap: int, race: Race):
        """Updates environmental conditions."""
        if hasattr(self, 'weather_df'):
            weather = self.get_weather_from_cumulative_time(
                race.race_id, 
                driver.cumulative_race_time
            )
        else:
            weather = self.default_weather

        if not self.validator.validate_weather_conditions(race):
            raise ValueError("Weather validation failed")

        driver.dynamic_features.update(weather)
        driver.dynamic_features['TrackStatus'] = self.get_track_status(lap, race)

    def get_track_status(self, lap: int, race: Race) -> int:
        """
        Determines the track status for the current lap.
        
        Args:
            lap: Current lap number
            race: Race object containing safety car periods
            
        Returns:
            Track status code (1 for normal racing, 4 for safety car)
        """
        return 4 if self.check_safety_car(lap, race) else 1

    def update_track_status(self, driver: Driver, lap: int, race: Race):
        """Updates track status based on race conditions."""
        is_safety_car = self.check_safety_car(lap, race)
        driver.dynamic_features['TrackStatus'] = 4 if is_safety_car else 1

    def update_positions(self, race: Race):
        """Updates driver positions based on cumulative race time."""
        race.drivers.sort(key=lambda d: d.cumulative_race_time)
        for pos, driver in enumerate(race.drivers, start=1):
            driver.current_position = pos
            race.lap_data[driver.driver_id]['positions'].append(pos)

    def update_gaps(self, drivers: List[Driver]):
        """Updates gaps between drivers."""
        if not drivers:
            return
            
        leader_time = drivers[0].cumulative_race_time
        for i, driver in enumerate(drivers):
            driver.dynamic_features['GapToLeader_ms'] = driver.cumulative_race_time - leader_time
            driver.dynamic_features['IntervalToPositionAhead_ms'] = (
                0.0 if i == 0 else driver.cumulative_race_time - drivers[i-1].cumulative_race_time
            )