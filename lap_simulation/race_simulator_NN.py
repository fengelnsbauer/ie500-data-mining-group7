# race_simulator_NN.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Optional
from features import RaceFeatures
import logging
import joblib
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the Race class
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

# Define the Driver class
class Driver:
    def __init__(self, driver_id: int, name: str, static_features: np.ndarray, 
                 initial_dynamic_features: Dict, start_position: int, 
                 pit_strategy: List[Tuple[int, int]], starting_compound: int):
        """
        Initializes a Driver instance.

        Args:
            driver_id (int): Unique identifier for the driver.
            name (str): Driver's name.
            static_features (np.ndarray): Scaled static features of the driver.
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
        self.sequence = self.initialize_sequence()  # Initialize sequence with zeros

    def initialize_sequence(self) -> np.ndarray:
        """
        Initializes the driver's sequence with zeros for the required window size.

        Returns:
            np.ndarray: Initialized sequence array.
        """
        window_size = 3  # Assuming a window size of 3
        sequence_dim = len(self.dynamic_features) + 1  # +1 for lap time
        return np.zeros((window_size, sequence_dim))

# Define the RaceSimulator class
class RaceSimulator:
    def __init__(self, model: torch.nn.Module, preprocessor: 'F1DataPreprocessor', model_type: str = 'lstm'):
        """
        Initializes the RaceSimulator.

        Args:
            model (torch.nn.Module): The trained LSTM model.
            preprocessor (F1DataPreprocessor): The data preprocessor with fitted scalers.
            model_type (str): Type of model ('lstm').
        """
        self.model = model
        self.preprocessor = preprocessor
        self.model_type = model_type.lower()
        self.race_features = RaceFeatures()
        self.pit_stop_duration = 25000  # Pit stop duration in milliseconds
        self.tire_compound_effects = {
            3: {'base_speed': 0.98, 'degradation_per_lap': 500},   # Soft
            2: {'base_speed': 0.99, 'degradation_per_lap': 300},   # Medium  
            1: {'base_speed': 1.0, 'degradation_per_lap': 200},    # Hard
            4: {'base_speed': 1.05, 'degradation_per_lap': 200},   # Intermediate
            5: {'base_speed': 1.1, 'degradation_per_lap': 200},    # Wet
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def prepare_features(self, driver: Driver, lap: int, race: Race) -> pd.DataFrame:
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
            **{f'static_{i}': val for i, val in enumerate(driver.static_features)},
            **driver.dynamic_features,
            'raceId': race.race_id,
            'driverId': driver.driver_id,
            'lap': lap
        }

        # Convert to DataFrame
        df = pd.DataFrame([features])
        return df

    def simulate_driver_lap(self, driver: Driver, lap: int, race: Race) -> float:
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

        # Transform data
        sequences_scaled, static_features_scaled, _ = self.preprocessor.transform_data(
            sequences=np.array([driver.sequence]),
            static_features=driver.static_features.reshape(1, -1),
            targets=np.array([0])  # Placeholder, not used
        )

        # Convert to tensors
        sequence_tensor = torch.FloatTensor(sequences_scaled).to(self.device)  # Shape: (1, window_size, sequence_dim)
        static_tensor = torch.FloatTensor(static_features_scaled).to(self.device)  # Shape: (1, static_dim)

        # Predict lap time
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(sequence_tensor, static_tensor)
            lap_time_normalized = prediction.item()

        # Inverse transform to get actual lap time
        lap_time = self.preprocessor.inverse_transform_lap_times(np.array([lap_time_normalized]))[0]

        # Check if lap is under safety car
        if self.is_safety_car(lap, race):
            lap_time *= 1.1  # Increase lap time by 10% under safety car

        # Check if driver is pitting
        if self.is_pit_lap(driver, lap):
            lap_time += self.pit_stop_duration  # Add pit stop duration

        return lap_time

    def is_safety_car(self, lap: int, race: Race) -> bool:
        """
        Checks if the current lap is under a safety car period.

        Args:
            lap (int): Current lap number.
            race (Race): The race instance.

        Returns:
            bool: True if under safety car, else False.
        """
        return any(start <= lap <= end for start, end in race.safety_car_periods)

    def is_pit_lap(self, driver: Driver, lap: int) -> bool:
        """
        Checks if the driver is pitting on the current lap.

        Args:
            driver (Driver): The driver.
            lap (int): Current lap number.

        Returns:
            bool: True if pitting, else False.
        """
        return any(stop_lap == lap for stop_lap, _ in driver.pit_strategy)

    def update_dynamic_features(self, driver: Driver, lap: int, race: Race):
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
        driver.dynamic_features['TrackStatus'] = 4 if self.is_safety_car(lap, race) else 1  # 4 for safety car, 1 for normal

    def update_driver_sequence(self, driver: Driver, lap_time: float):
        """
        Updates the driver's sequence with the new lap time and dynamic features.

        Args:
            driver (Driver): The driver.
            lap_time (float): The lap time in milliseconds.
        """
        # Roll the sequence to remove the oldest entry
        driver.sequence = np.roll(driver.sequence, -1, axis=0)

        # Normalize lap time
        lap_time_scaled = self.preprocessor.lap_time_scaler.transform([[lap_time]]).flatten()

        # Scale dynamic features (excluding tire_compound)
        dynamic_features_flat = np.array([driver.dynamic_features[feature] for feature in self.preprocessor.race_features.dynamic_features])
        dynamic_features_scaled = self.preprocessor.dynamic_scaler.transform(dynamic_features_flat.reshape(1, -1)).flatten()

        # Reconstruct the new sequence entry
        new_sequence_entry = np.concatenate([lap_time_scaled, dynamic_features_scaled])

        # Update the sequence
        driver.sequence[-1] = new_sequence_entry

    def update_positions(self, race: Race):
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

    def simulate_race(self, race: Race) -> Dict[int, Dict[str, List]]:
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
        for lap in tqdm(range(1, race.total_laps + 1), desc="Simulating Race"):
            lap_times = {}
            
            # Simulate each driver
            for driver in race.drivers:
                # Update dynamic features before lap
                self.update_dynamic_features(driver, lap, race)
                
                # Simulate lap time
                lap_time = self.simulate_driver_lap(driver, lap, race)
                lap_times[driver.driver_id] = lap_time
                
                # Update driver's sequence with the new lap time
                self.update_driver_sequence(driver, lap_time)
                
                # Store input features (optional, can be expanded as needed)
                race.lap_data[driver.driver_id]['inputs'].append({
                    'lap': lap,
                    'driver_id': driver.driver_id,
                    'driver_name': driver.name,
                    'static_features': driver.static_features.copy(),
                    'dynamic_features': driver.dynamic_features.copy(),
                })
            
            # Update positions based on lap times
            self.update_positions(race)

            # Record lap times and positions
            for driver in race.drivers:
                race.lap_data[driver.driver_id]['lap_times'].append(lap_times[driver.driver_id])
                race.lap_data[driver.driver_id]['positions'].append(driver.current_position)

        logging.info(f"Race {race.race_id} simulation completed.")
        return race.lap_data

# Define the plotting functions
def plot_race_positions(race: Race, constructor_mapping: Dict[int, str], driver_code_mapping: Dict[int, str], TEAM_COLORS: Dict[str, str], save_path: Optional[str] = None):
    """
    Plots the positions of drivers over the course of the race with constructor-specific colors
    and driver codes as labels.

    Args:
        race (Race): The race instance.
        constructor_mapping (dict): Mapping from constructorId to constructor name.
        driver_code_mapping (dict): Mapping from driverId to driver code.
        TEAM_COLORS (dict): Mapping from constructor name to color.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    plt.figure(figsize=(12, 6))

    # Create a color cycle for drivers without constructor info
    default_colors = plt.cm.rainbow(np.linspace(0, 1, len(race.drivers)))
    color_idx = 0

    # Extract constructors for drivers
    for driver in race.drivers:
        driver_id = driver.driver_id
        positions = race.lap_data[driver_id]['positions']
        
        # Get constructor ID from the first input's static features (assuming constructorId is stored)
        if 'static_constructorId' in race.lap_data[driver_id]['inputs'][0]['static_features']:
            constructor_id = race.lap_data[driver_id]['inputs'][0]['static_features']['static_constructorId']
            constructor_name = constructor_mapping.get(constructor_id, 'unknown').lower()
            color = TEAM_COLORS.get(constructor_name, default_colors[color_idx])
            if constructor_name not in TEAM_COLORS:
                color_idx = (color_idx + 1) % len(default_colors)
        else:
            color = default_colors[color_idx]
            color_idx = (color_idx + 1) % len(default_colors)
        
        driver_code = driver_code_mapping.get(driver_id, f'D{driver_id}')
        
        plt.plot(range(1, len(positions) + 1), positions, label=driver_code, color=color)
        
        # Annotate starting position
        plt.scatter(1, positions[0], color=color, marker='o', s=100, edgecolors='black')
        plt.text(1, positions[0], f"Start: {positions[0]}", fontsize=9, ha='right')

    plt.gca().invert_yaxis()  # Position 1 at top
    plt.xlabel('Lap')
    plt.ylabel('Position')
    plt.title(f'Race {race.race_id} Simulation: Driver Positions Over Laps')
    plt.legend(title='Driver Codes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # Shade safety car periods
    if race.safety_car_periods:
        for start, end in race.safety_car_periods:
            plt.axvspan(start, end, color='yellow', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Race positions plot saved to {save_path}")

    plt.show()


def plot_lap_times(race: Race, constructor_mapping: Dict[int, str], driver_code_mapping: Dict[int, str], TEAM_COLORS: Dict[str, str], save_path: Optional[str] = None):
    """
    Plots the lap times of drivers over the course of the race with constructor-specific colors
    and driver codes as labels.

    Args:
        race (Race): The race instance.
        constructor_mapping (dict): Mapping from constructorId to constructor name.
        driver_code_mapping (dict): Mapping from driverId to driver code.
        TEAM_COLORS (dict): Mapping from constructor name to color.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    plt.figure(figsize=(12, 6))

    # Create a color cycle for drivers without constructor info
    default_colors = plt.cm.rainbow(np.linspace(0, 1, len(race.drivers)))
    color_idx = 0

    # Extract constructors for drivers
    for driver in race.drivers:
        driver_id = driver.driver_id
        lap_times = race.lap_data[driver_id]['lap_times']
        
        # Get constructor ID from the first input's static features (assuming constructorId is stored)
        if 'static_constructorId' in race.lap_data[driver_id]['inputs'][0]['static_features']:
            constructor_id = race.lap_data[driver_id]['inputs'][0]['static_features']['static_constructorId']
            constructor_name = constructor_mapping.get(constructor_id, 'unknown').lower()
            color = TEAM_COLORS.get(constructor_name, default_colors[color_idx])
            if constructor_name not in TEAM_COLORS:
                color_idx = (color_idx + 1) % len(default_colors)
        else:
            color = default_colors[color_idx]
            color_idx = (color_idx + 1) % len(default_colors)
        
        driver_code = driver_code_mapping.get(driver_id, f'D{driver_id}')
        
        plt.plot(range(1, len(lap_times) + 1), lap_times, label=driver_code, color=color)

    plt.xlabel('Lap')
    plt.ylabel('Lap Time (ms)')
    plt.title(f'Race {race.race_id} Simulation: Driver Lap Times')
    plt.legend(title='Driver Codes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    # Shade safety car periods
    if race.safety_car_periods:
        for start, end in race.safety_car_periods:
            plt.axvspan(start, end, color='yellow', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Lap times plot saved to {save_path}")

    plt.show()

def create_lap_times_with_inputs_dataframe(race: Race, race_features: RaceFeatures) -> pd.DataFrame:
    """
    Creates a detailed DataFrame containing lap times, positions, and input features for each driver.

    Args:
        race (Race): The race instance.
        race_features (RaceFeatures): The RaceFeatures instance containing feature names.

    Returns:
        pd.DataFrame: The detailed lap times DataFrame.
    """
    records = []
    for driver in race.drivers:
        lap_times = race.lap_data[driver.driver_id]['lap_times']
        positions = race.lap_data[driver.driver_id]['positions']
        inputs_list = race.lap_data[driver.driver_id]['inputs']
        
        for lap_index, (lap_time, position, inputs) in enumerate(zip(lap_times, positions, inputs_list)):
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
