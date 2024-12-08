# simulation_validator.py
from typing import Dict, Any, List
import logging

class SimulationValidator:
    @staticmethod
    def validate_race_initialization(race) -> bool:
        required_attributes = ['race_id', 'circuit_id', 'year', 'total_laps', 
                             'circuit_length', 'current_conditions']
        
        for attr in required_attributes:
            if not hasattr(race, attr) or getattr(race, attr) is None:
                logging.error(f"Race missing required attribute: {attr}")
                return False
                
        if not race.drivers:
            logging.error("Race has no drivers")
            return False
            
        return True
        
    @staticmethod
    def validate_driver_state(driver) -> bool:
        required_features = ['tire_compound', 'tire_age', 'fuel_load', 
                           'current_position', 'cumulative_race_time']
                           
        for feature in required_features:
            if feature not in driver.dynamic_features:
                logging.error(f"Driver {driver.driver_id} missing feature: {feature}")
                return False
                
        return True
        
    @staticmethod
    def validate_lap_time(lap_time: float, previous_times: List[float]) -> bool:
        if lap_time <= 0:
            logging.error(f"Invalid negative lap time: {lap_time}")
            return False
            
        if previous_times and lap_time > max(previous_times) * 1.5:
            logging.warning(f"Unusually slow lap time detected: {lap_time}")
            
        return True
    
    @staticmethod
    def validate_weather_conditions(conditions: Dict[str, float]) -> bool:
        valid_ranges = {
            'TrackTemp': (-50, 70),
            'AirTemp': (-50, 50),
            'Humidity': (0, 100)
        }
        
        for key, (min_val, max_val) in valid_ranges.items():
            if key in conditions:
                if not min_val <= conditions[key] <= max_val:
                    logging.error(f"Invalid {key}: {conditions[key]}")
                    return False
        return True