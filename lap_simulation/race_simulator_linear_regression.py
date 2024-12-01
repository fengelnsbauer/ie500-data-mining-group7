import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from features import RaceFeatures
import logging
import pickle  # Use pickle for Linear Regression
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the Race class
class Race:
    def __init__(self, race_id, circuit_id, total_laps, weather_conditions, safety_car_periods=None):
        self.race_id = race_id
        self.circuit_id = circuit_id
        self.total_laps = total_laps
        self.weather_conditions = weather_conditions  # Dictionary with lap-wise weather data
        self.safety_car_periods = safety_car_periods or []  # List of tuples [(start_lap, end_lap), ...]
        self.drivers = []
        self.lap_data = {}  # To store lap times and positions for each driver

# Define the Driver class
class Driver:
    def __init__(self, driver_id, name, static_features, initial_dynamic_features, start_position, pit_strategy, starting_compound):
        self.driver_id = driver_id
        self.name = name
        self.static_features = static_features
        self.dynamic_features = initial_dynamic_features
        self.sequence = None  # To store previous laps' data
        self.current_position = start_position  # Updated each lap
        self.start_position = start_position  # Initial grid position
        self.pit_strategy = pit_strategy  # List of tuples: [(lap_number, new_compound), ...]
        self.starting_compound = starting_compound  # Starting tire compound

# Update the RaceSimulator class
class RaceSimulator:
    def __init__(self, model_pipeline, model_type='linear_regression'):
        """
        Initializes the RaceSimulator.

        Parameters:
            model_pipeline: The trained pipeline (includes preprocessing and model).
            model_type (str): Type of model ('linear_regression').
        """
        self.model_pipeline = model_pipeline
        self.model_type = model_type.lower()
        self.pit_stop_duration = 25000  # Pit stop penalty in milliseconds
        self.tire_compound_effects = {
            3: {'base_speed': 0.98, 'degradation_per_lap': 500},   # Soft
            2: {'base_speed': 0.99, 'degradation_per_lap': 300},   # Medium  
            1: {'base_speed': 1.0, 'degradation_per_lap': 200},    # Hard
            4: {'base_speed': 1.05, 'degradation_per_lap': 200},   # Intermediate
            5: {'base_speed': 1.1, 'degradation_per_lap': 200},    # Wet
        }

    def initialize_sequence(self, driver):
        if self.model_type == 'lstm':
            # Initialize with zeros or previous laps (e.g., from practice sessions)
            window_size = 3  # Assuming a window size of 3
            sequence_dim = len(self.preprocessor.get("race_features", RaceFeatures()).dynamic_features) + 1  # Lap time + dynamic features

            # For simplicity, initialize with zeros
            initial_sequence = np.zeros((window_size, sequence_dim))
            return initial_sequence
        elif self.model_type == 'linear_regression':
            # For Linear Regression, sequences are not required
            return None

    def update_dynamic_features(self, driver, lap, race):
        # Update features like tire_age, fuel_load, etc.
        driver.dynamic_features['tire_age'] += 1
        driver.dynamic_features['fuel_load'] -= 1.5  # Example consumption per lap

        # Check for pit stop
        pit_stops = [stop for stop in driver.pit_strategy if stop[0] == lap]
        if pit_stops:
            driver.dynamic_features['is_pit_lap'] = 1
            driver.dynamic_features['tire_age'] = 0  # Reset tire age after pit stop

            # Change tire compound to the specified compound in the strategy
            new_compound = pit_stops[0][1]
            driver.dynamic_features['tire_compound'] = new_compound
        else:
            driver.dynamic_features['is_pit_lap'] = 0

        # Update weather conditions (if any changes)
        weather = race.weather_conditions.get(lap, {})
        driver.dynamic_features['TrackTemp'] = weather.get('TrackTemp', driver.dynamic_features['TrackTemp'])
        driver.dynamic_features['AirTemp'] = weather.get('AirTemp', driver.dynamic_features['AirTemp'])
        driver.dynamic_features['Humidity'] = weather.get('Humidity', driver.dynamic_features['Humidity'])

        # Handle TrackStatus for safety car periods
        is_safety_car = any(start <= lap <= end for start, end in race.safety_car_periods)
        driver.dynamic_features['TrackStatus'] = 4 if is_safety_car else 1  # Explicitly set safety car or default


    def simulate_driver_lap(self, driver, lap, race):
        """
        Simulate the lap time for a driver using the pipeline.
        """
        original_track_status = driver.dynamic_features['TrackStatus']
        original_is_pit_lap = driver.dynamic_features['is_pit_lap']

        # Prepare features for prediction
        # Combine driver features with dynamic features
        features = driver.dynamic_features.copy()
        features.update({'driverId': driver.driver_id, 'raceId': race.race_id})

        # Convert features to DataFrame
        driver_df = pd.DataFrame([features])

        # Get the feature names from the pipeline's preprocessor
        preprocessor = self.model_pipeline.named_steps['preprocessor']
        numerical_features = list(preprocessor.transformers_[0][2])
        categorical_features = list(preprocessor.transformers_[1][2])

        # Ensure all necessary columns are present
        for col in numerical_features + categorical_features:
            if col not in driver_df.columns:
                driver_df[col] = 0  # Add missing features with zeros

        # Enforce data types
        driver_df[numerical_features] = driver_df[numerical_features].astype(float)
        driver_df[categorical_features] = driver_df[categorical_features].astype(str)

        # Reorder columns to match the training data
        driver_df = driver_df[numerical_features + categorical_features]

        # Check for missing values
        if driver_df.isnull().any().any():
            driver_df.fillna(0, inplace=True)

        # Predict lap time using the pipeline
        lap_time = self.model_pipeline.predict(driver_df)[0]

        # Apply manual adjustments for safety car or pit stop
        if original_track_status == 4:
            lap_time *= 1.1  # Increase lap time by 10% for safety car
        if original_is_pit_lap == 1:
            lap_time += self.pit_stop_duration  # Add pit stop penalty (25,000 ms)

        return lap_time


    def update_driver_sequence(self, driver, lap_time):
        if self.model_type == 'lstm':
            # Update the sequence with the new lap data
            driver.sequence = np.roll(driver.sequence, -1, axis=0)

            # Create a copy of dynamic features excluding tire_compound initially
            dynamic_features_dict = driver.dynamic_features.copy()
            tire_compound = dynamic_features_dict.pop('tire_compound')  # Remove and store tire compound

            # Ensure the order matches the preprocessor's expected order
            dynamic_features = []
            for feature in self.preprocessor.get("race_features", RaceFeatures()).dynamic_features:
                if feature == 'tire_compound':
                    dynamic_features.append(tire_compound)
                else:
                    dynamic_features.append(dynamic_features_dict.get(feature, 0))  # Handle missing features

            dynamic_features = np.array(dynamic_features)

            # Scale all features except tire_compound
            scaled_features = self.preprocessor["dynamic_scaler"].transform(
                dynamic_features.reshape(1, -1)
            ).flatten()

            # Normalize lap time
            lap_time_normalized = self.preprocessor["lap_time_scaler"].transform([[lap_time]]).flatten()

            # Combine normalized lap time and scaled features
            new_sequence_entry = np.concatenate([lap_time_normalized, scaled_features])

            # Update the sequence
            driver.sequence[-1] = new_sequence_entry

        elif self.model_type == 'linear_regression':
            # For Linear Regression, sequences are not maintained
            pass  # No action needed

    def update_positions(self, race, lap_times):
        # Get current lap number
        current_lap = len(next(iter(race.lap_data.values()))['lap_times'])
        is_safety_car = any(start <= current_lap <= end for start, end in race.safety_car_periods)

        if is_safety_car:
            # Positions remain the same during safety car
            for driver in race.drivers:
                race.lap_data[driver.driver_id]['positions'].append(driver.current_position)
        else:
            # Update positions based on cumulative times
            cumulative_times = {}
            for driver in race.drivers:
                total_time = sum(race.lap_data[driver.driver_id]['lap_times'])
                cumulative_times[driver.driver_id] = total_time

            # Sort drivers based on cumulative times
            sorted_drivers = sorted(cumulative_times.items(), key=lambda x: x[1])

            # Update driver positions
            for position, (driver_id, _) in enumerate(sorted_drivers, start=1):
                driver = next(d for d in race.drivers if d.driver_id == driver_id)
                driver.current_position = position
                race.lap_data[driver.driver_id]['positions'].append(position)

    def simulate_race(self, race: Race):
        """
        Simulate the race lap by lap for all drivers, updating dynamic features and predicting lap times.
        """
        # Initialize lap data storage
        if not race.drivers:
            raise ValueError("No drivers added to the race.")

        for driver in race.drivers:
            race.lap_data[driver.driver_id] = {
                'lap_times': [],
                'positions': [],
                'inputs': []  # To store input features
            }
            # Initialize driver sequence if using LSTM (not needed for Linear Regression)
            driver.sequence = self.initialize_sequence(driver)

        # Simulate each lap
        for lap in range(1, race.total_laps + 1):
            lap_times = {}
            # Simulate lap for each driver
            for driver in race.drivers:
                # Update driver's dynamic features
                self.update_dynamic_features(driver, lap, race)

                # Prepare input features for storage
                input_features = {
                    'lap': lap,
                    'driver_id': driver.driver_id,
                    'driver_name': driver.name,
                    'static_features': driver.static_features.copy(),
                    'dynamic_features': driver.dynamic_features.copy(),
                }

                # Simulate lap time
                lap_time = self.simulate_driver_lap(driver, lap, race)
                lap_times[driver.driver_id] = lap_time

                # Update driver's sequence if using LSTM (not needed for Linear Regression)
                self.update_driver_sequence(driver, lap_time)

                # Store input features
                race.lap_data[driver.driver_id]['inputs'].append(input_features)

            # Update positions based on lap times
            self.update_positions(race, lap_times)

            # Record lap times and positions
            for driver in race.drivers:
                race.lap_data[driver.driver_id]['lap_times'].append(lap_times[driver.driver_id])
                race.lap_data[driver.driver_id]['positions'].append(driver.current_position)

        return race.lap_data


# Model Loading Functions
def load_linear_regression_model(path):
    """
    Loads the Linear Regression model and its preprocessor from a pickle file.

    Parameters:
        path (str): Path to the pickle file.

    Returns:
        model: Trained Linear Regression model.
        preprocessor (dict): Preprocessing components.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['preprocessor']

def load_xgboost_model(path):
    return joblib.load(path)

def load_random_forest_model(path):
    return joblib.load(path)

# Analysis and Plotting Functions
def plot_race_positions(race):
    plt.figure(figsize=(12, 6))
    
    for driver in race.drivers:
        positions = race.lap_data[driver.driver_id]['positions']
        total_laps = len(positions)  # Get actual number of laps
        plt.plot(range(1, total_laps + 1), positions, label=driver.name)
    
    plt.gca().invert_yaxis()  # Invert y-axis so that position 1 is at the top
    plt.xlabel('Lap')
    plt.ylabel('Position')
    plt.title('Race Simulation: Driver Positions Over Laps')
    plt.legend()
    plt.grid(True)

    # Shade safety car periods
    for start, end in race.safety_car_periods:
        plt.axvspan(start, end, color='yellow', alpha=0.3)

    plt.show()

def plot_lap_times(race):
    plt.figure(figsize=(12, 6))
    
    for driver in race.drivers:
        lap_times = race.lap_data[driver.driver_id]['lap_times']
        total_laps = len(lap_times)  # Get actual number of laps
        plt.plot(range(1, total_laps + 1), lap_times, label=driver.name)
    
    plt.xlabel('Lap')
    plt.ylabel('Lap Time (ms)')
    plt.title('Race Simulation: Driver Lap Times')
    plt.legend()
    plt.grid(True)

    # Shade safety car periods
    for start, end in race.safety_car_periods:
        plt.axvspan(start, end, color='yellow', alpha=0.3)

    plt.show()

def create_lap_times_dataframe(race) -> pd.DataFrame:
    data = {'Lap': []}
    total_laps = race.total_laps
    for driver in race.drivers:
        data[driver.name] = race.lap_data[driver.driver_id]['lap_times']
    data['Lap'] = list(range(1, total_laps + 1))
    lap_times_df = pd.DataFrame(data)
    return lap_times_df

def create_lap_times_with_inputs_dataframe(race, race_features) -> pd.DataFrame:
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
            # Flatten static and dynamic features
            for i, feature_name in enumerate(race_features.static_features):
                record[feature_name] = inputs['static_features'][i]
            for feature_name, value in inputs['dynamic_features'].items():
                record[feature_name] = value
            records.append(record)
    lap_times_df = pd.DataFrame(records)
    return lap_times_df
