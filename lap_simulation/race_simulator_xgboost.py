# race_simulator_xgboost.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional
from xgboost_utils import load_pipeline  # Updated to load_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Race:
    def __init__(self, race_id: int, circuit_id: int, total_laps: int, 
                 weather_conditions: Dict[int, Dict[str, float]], 
                 safety_car_periods: Optional[List[Tuple[int, int]]] = None):
        """
        Initializes a Race instance.

        Args:
            race_id (int): Unique identifier for the race.
            circuit_id (int): Identifier for the circuit.
            total_laps (int): Total number of laps in the race.
            weather_conditions (dict): Mapping of lap number to weather data.
            safety_car_periods (list of tuples): List of (start_lap, end_lap) tuples indicating safety car periods.
        """
        self.race_id = race_id
        self.circuit_id = circuit_id
        self.total_laps = total_laps
        self.weather_conditions = weather_conditions  # Dict mapping lap to weather data
        self.safety_car_periods = safety_car_periods or []  # List of (start_lap, end_lap)
        self.drivers: List['Driver'] = []
        self.lap_data: Dict[int, Dict[str, List]] = {}  # driver_id -> lap data

class Driver:
    def __init__(self, driver_id: int, name: str, static_features: Dict, 
                 initial_dynamic_features: Dict, start_position: int, 
                 pit_strategy: List[Tuple[int, int]], starting_compound: int):
        """
        Initializes a Driver instance.

        Args:
            driver_id (int): Unique identifier for the driver.
            name (str): Driver's name.
            static_features (dict): Static features (categorical) of the driver.
            initial_dynamic_features (dict): Initial dynamic features of the driver.
            start_position (int): Starting grid position.
            pit_strategy (list of tuples): List of (lap_number, new_compound) indicating pit stop plans.
            starting_compound (int): Initial tire compound.
        """
        self.driver_id = driver_id
        self.name = name
        self.static_features = static_features
        self.dynamic_features = initial_dynamic_features.copy()
        self.current_position = start_position
        self.start_position = start_position
        self.pit_strategy = pit_strategy  # List of (lap_number, new_compound)
        self.starting_compound = starting_compound

class RaceSimulator:
    def __init__(self, pipeline, model_type='xgboost'):
        """
        Initializes the RaceSimulator.

        Args:
            pipeline (Pipeline): The trained scikit-learn pipeline.
            model_type (str): Type of model ('xgboost').
        """
        self.pipeline = pipeline
        self.model_type = model_type.lower()
        self.pit_stop_duration = 25000  # Pit stop penalty in milliseconds
        self.tire_compound_effects = {
            3: {'base_speed': 0.98, 'degradation_per_lap': 500},   # Soft
            2: {'base_speed': 0.99, 'degradation_per_lap': 300},   # Medium  
            1: {'base_speed': 1.0, 'degradation_per_lap': 200},    # Hard
            4: {'base_speed': 1.05, 'degradation_per_lap': 200},   # Intermediate
            5: {'base_speed': 1.1, 'degradation_per_lap': 200},    # Wet
        }
    
    def prepare_features(self, driver: 'Driver', lap: int, race: 'Race') -> pd.DataFrame:
        """
        Prepares the features for a driver at a given lap.

        Args:
            driver (Driver): The driver.
            lap (int): Current lap number.
            race (Race): The race instance.

        Returns:
            pd.DataFrame: Processed features ready for prediction.
        """
        # Combine static and dynamic features
        features = {
            **driver.static_features,
            **driver.dynamic_features,
            'raceId': race.race_id,
            'driverId': driver.driver_id,
            'lap': lap
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        return df
    
    def simulate_driver_lap(self, driver: 'Driver', lap: int, race: 'Race') -> float:
        """
        Simulates a lap for a driver.

        Args:
            driver (Driver): The driver.
            lap (int): Current lap number.
            race (Race): The race instance.

        Returns:
            float: Predicted lap time in milliseconds.
        """
        # Prepare features
        driver_df = self.prepare_features(driver, lap, race)
        
        # Predict lap time using the entire pipeline
        lap_time = self.pipeline.predict(driver_df)[0]
        
        # Apply manual adjustments
        if self.is_safety_car(lap, race):
            lap_time *= 1.1  # Increase lap time by 10% under safety car
        if self.is_pit_lap(driver, lap):
            lap_time += self.pit_stop_duration  # Add pit stop duration
        
        return lap_time
    
    def is_safety_car(self, lap: int, race: 'Race') -> bool:
        """
        Checks if the current lap is under a safety car period.

        Args:
            lap (int): Current lap number.
            race (Race): The race instance.

        Returns:
            bool: True if under safety car, else False.
        """
        return any(start <= lap <= end for start, end in race.safety_car_periods)
    
    def is_pit_lap(self, driver: 'Driver', lap: int) -> bool:
        """
        Checks if the driver is pitting on the current lap.

        Args:
            driver (Driver): The driver.
            lap (int): Current lap number.

        Returns:
            bool: True if pitting, else False.
        """
        return any(stop_lap == lap for stop_lap, _ in driver.pit_strategy)
    
    def update_dynamic_features(self, driver: 'Driver', lap: int, race: 'Race'):
        """
        Updates the dynamic features of a driver after a lap.

        Args:
            driver (Driver): The driver.
            lap (int): Current lap number.
            race (Race): The race instance.
        """
        # Update tire age and fuel load
        driver.dynamic_features['tire_age'] += 1
        driver.dynamic_features['fuel_load'] = max(0, driver.dynamic_features.get('fuel_load', 100.0) - 1.5)  # Example consumption
        
        # Handle pit stops
        pitted = False
        for pit_lap, new_compound in driver.pit_strategy:
            if pit_lap == lap:
                driver.dynamic_features['is_pit_lap'] = 1
                driver.dynamic_features['tire_age'] = 0
                driver.dynamic_features['tire_compound'] = new_compound
                pitted = True
                break
        if not pitted:
            driver.dynamic_features['is_pit_lap'] = 0
        
        # Update weather features
        weather = race.weather_conditions.get(lap, {})
        for key in ['TrackTemp', 'AirTemp', 'Humidity']:
            if key in weather:
                driver.dynamic_features[key] = weather[key]
        
        # Update TrackStatus
        driver.dynamic_features['TrackStatus'] = 4 if self.is_safety_car(lap, race) else 1
    
    def update_positions(self, race: 'Race'):
        """
        Updates the positions of drivers based on their cumulative lap times.

        Args:
            race (Race): The race instance.
        """
        cumulative_times = {}
        for driver in race.drivers:
            cumulative_time = sum(race.lap_data[driver.driver_id]['lap_times'])
            cumulative_times[driver.driver_id] = cumulative_time
        
        # Sort drivers by cumulative time (lower is better)
        sorted_drivers = sorted(cumulative_times.items(), key=lambda x: x[1])
        
        # Assign positions
        for position, (driver_id, _) in enumerate(sorted_drivers, start=1):
            driver = next(d for d in race.drivers if d.driver_id == driver_id)
            driver.current_position = position
            race.lap_data[driver.driver_id]['positions'].append(position)
    
    def simulate_race(self, race: 'Race') -> Dict[int, Dict[str, List]]:
        """
        Simulates the entire race.

        Args:
            race (Race): The race instance.

        Returns:
            Dict: Lap data for each driver.
        """
        if not race.drivers:
            raise ValueError("No drivers added to the race.")
        
        # Initialize lap data
        for driver in race.drivers:
            race.lap_data[driver.driver_id] = {
                'lap_times': [],
                'positions': [],
                'inputs': []
            }
        
        # Simulate each lap
        for lap in range(1, race.total_laps + 1):
            logging.info(f"Simulating lap {lap}/{race.total_laps}")
            lap_times = {}
            
            # Simulate each driver
            for driver in race.drivers:
                # Simulate lap time
                lap_time = self.simulate_driver_lap(driver, lap, race)
                lap_times[driver.driver_id] = lap_time
                race.lap_data[driver.driver_id]['lap_times'].append(lap_time)
                
                # Update dynamic features after lap
                self.update_dynamic_features(driver, lap, race)
                
                # Store input features (optional)
                race.lap_data[driver.driver_id]['inputs'].append({
                    'lap': lap,
                    'driver_id': driver.driver_id,
                    'driver_name': driver.name,
                    'static_features': driver.static_features.copy(),
                    'dynamic_features': driver.dynamic_features.copy(),
                })
            
            # Update positions based on lap times
            self.update_positions(race)
        
        return race.lap_data

# Visualization functions remain unchanged
def plot_race_positions(race, constructor_mapping, driver_code_mapping, TEAM_COLORS):
    """
    Plots the positions of drivers over the course of the race with constructor-specific colors
    and driver codes as labels.

    Args:
        race (Race): The race instance.
        constructor_mapping (dict): Mapping from constructorId to constructor name.
        driver_code_mapping (dict): Mapping from driverId to driver code.
        TEAM_COLORS (dict): Mapping from constructor name to color.
    """
    plt.figure(figsize=(12, 6))
    
    for driver in race.drivers:
        driver_id = driver.driver_id
        positions = race.lap_data[driver_id]['positions']
        constructor_id = driver.constructorId
        color = get_constructor_color(constructor_id)
        driver_code = driver_code_mapping.get(driver_id, 'UNK')  # 'UNK' for unknown
        
        plt.plot(range(1, len(positions) + 1), positions, label=driver_code, color=color)
    
    plt.gca().invert_yaxis()  # Position 1 at top
    plt.xlabel('Lap')
    plt.ylabel('Position')
    plt.title('Race Simulation: Driver Positions Over Laps')
    plt.legend(title='Driver Codes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Shade safety car periods
    for start, end in race.safety_car_periods:
        plt.axvspan(start, end, color='yellow', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_lap_times(race: Race):
    """
    Plots the lap times of drivers over the course of the race.

    Args:
        race (Race): The race instance.
    """
    plt.figure(figsize=(12, 6))
    
    for driver in race.drivers:
        lap_times = race.lap_data[driver.driver_id]['lap_times']
        plt.plot(range(1, len(lap_times) + 1), lap_times, label=driver.name)
    
    plt.xlabel('Lap')
    plt.ylabel('Lap Time (ms)')
    plt.title('Race Simulation: Driver Lap Times')
    plt.legend()
    plt.grid(True)
    
    # Shade safety car periods
    for start, end in race.safety_car_periods:
        plt.axvspan(start, end, color='yellow', alpha=0.3)
    
    plt.show()

def create_lap_times_with_inputs_dataframe(race) -> pd.DataFrame:
    records = []
    for driver in race.drivers:
        lap_times = race.lap_data[driver.driver_id]['lap_times']
        positions = race.lap_data[driver.driver_id]['positions']
        inputs_list = race.lap_data[driver.driver_id]['inputs']
        
        for lap_index, (lap_time, position, inputs) in enumerate(zip(lap_times, positions, inputs_list)):
            # Create base record with lap info
            record = {
                'Lap': lap_index + 1,
                'Driver': driver.name,
                'LapTime': lap_time,
                'Position': position,
            }
            
            # Add all static features
            for feature_name, value in inputs['static_features'].items():
                record[f'static_{feature_name}'] = value
                
            # Add all dynamic features
            for feature_name, value in inputs['dynamic_features'].items():
                record[f'dynamic_{feature_name}'] = value
                
            records.append(record)
            
    # Create DataFrame and sort by lap and position
    lap_times_df = pd.DataFrame(records)
    lap_times_df = lap_times_df.sort_values(['Lap', 'Position'])
    
    return lap_times_df

# Example usage:
def analyze_race_results(race):
    # Create detailed DataFrame
    df = create_lap_times_with_inputs_dataframe(race)
    
    # Print summary statistics
    print("\nRace Analysis:")
    print(f"Total Laps: {df['Lap'].max()}")
    print(f"Number of Drivers: {len(race.drivers)}")
    
    # Best lap times
    best_laps = df.groupby('Driver')['LapTime'].agg(['min', 'mean', 'max']).round(2)
    best_laps.columns = ['Best Lap', 'Average Lap', 'Worst Lap']
    print("\nLap Time Analysis:")
    print(best_laps)
    
    # Tire compound usage
    tire_usage = df.groupby(['Driver', 'dynamic_tire_compound']).size().unstack(fill_value=0)
    print("\nTire Compound Usage (laps per compound):")
    print(tire_usage)
    
    # Save detailed lap times to CSV
    output_path = f'results/race_{race.race_id}_detailed.csv'
    df.to_csv(output_path, index=False)
    print(f"\nDetailed lap times saved to {output_path}")
    
    return df
