import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from features import RaceFeatures
import logging
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


# %%
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
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.race_features = RaceFeatures()
        self.pit_stop_duration = 25_000  # Average pit stop duration in milliseconds
        self.tire_compound_effects = {
            3: {'base_speed': 0.98, 'degradation_per_lap': 500},   # Soft
            2: {'base_speed': 0.99, 'degradation_per_lap': 300},  # Medium  
            1: {'base_speed': 1.0, 'degradation_per_lap': 200},    # Hard
            4: {'base_speed': 1.05, 'degradation_per_lap': 200},    # Intermediate
            5: {'base_speed': 1.1, 'degradation_per_lap': 200},    # Wet
        }
        
    def initialize_sequence(self, driver):
        # Initialize with zeros or previous laps (e.g., from practice sessions)
        window_size = 3  # Assuming a window size of 3
        sequence_dim = len(self.preprocessor.race_features.dynamic_features) + 1  # Lap time + dynamic features
        
        # For simplicity, initialize with zeros
        initial_sequence = np.zeros((window_size, sequence_dim))
        return initial_sequence

    def update_dynamic_features(self, driver, lap, race):
        # Update features like tire_age, fuel_load, etc.
        driver.dynamic_features['tire_age'] += 1
        driver.dynamic_features['fuel_load'] -= 1.5  # Example consumption per lap

        # Check for pit stop
        pit_stops = [stop for stop in driver.pit_strategy if stop[0] == lap]
        if pit_stops:
            driver.dynamic_features['is_pit_lap'] = 1
            driver.dynamic_features['tire_age'] = 0  # Reset tire age after pit stop

            # Change tire compound to the specified compound
            new_compound = pit_stops[0][1]
            driver.dynamic_features['tire_compound'] = new_compound
        else:
            driver.dynamic_features['is_pit_lap'] = 0

        # Update weather conditions (if any changes)
        weather = race.weather_conditions.get(lap, {})
        driver.dynamic_features['track_temp'] = weather.get('track_temp', driver.dynamic_features['track_temp'])
        driver.dynamic_features['air_temp'] = weather.get('air_temp', driver.dynamic_features['air_temp'])
        driver.dynamic_features['humidity'] = weather.get('humidity', driver.dynamic_features['humidity'])

        # Handle TrackStatus for safety car periods
        is_safety_car = any(start <= lap <= end for start, end in race.safety_car_periods)
        driver.dynamic_features['TrackStatus'] = '4' if is_safety_car else '1'


    def simulate_driver_lap(self, driver):
        # Prepare input data
        sequence = driver.sequence
        static = driver.static_features

        # Ensure correct data types and shapes
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Shape: (1, window_size, sequence_dim)
        static_tensor = torch.FloatTensor(static).unsqueeze(0)  # Shape: (1, static_dim)

        # Predict lap time
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(sequence_tensor, static_tensor)
            lap_time_normalized = prediction.item()

        # Inverse transform to get actual lap time
        lap_time = self.preprocessor.lap_time_scaler.inverse_transform([[lap_time_normalized]])[0][0]

        # Adjust lap time during safety car periods
        if driver.dynamic_features['TrackStatus'] == '4':
            lap_time *= 1.3  # Increase lap time by 30% during safety car

        return lap_time

    
    def update_driver_sequence(self, driver, lap_time):
        # Update the sequence with the new lap data
        driver.sequence = np.roll(driver.sequence, -1, axis=0)

        # Prepare new sequence entry
        dynamic_features_to_scale = [
            col for col in self.preprocessor.race_features.dynamic_features
            if col not in ['tire_compound']
        ]
        tire_compound_value = driver.dynamic_features['tire_compound']
        other_dynamic_feature_values = np.array([
            driver.dynamic_features[feature] for feature in dynamic_features_to_scale
        ])
        other_dynamic_features_scaled = self.preprocessor.dynamic_scaler.transform(
            other_dynamic_feature_values.reshape(1, -1)
        ).flatten()

        dynamic_features_combined = np.hstack((tire_compound_value, other_dynamic_features_scaled))

        lap_time_normalized = self.preprocessor.lap_time_scaler.transform([[lap_time]]).flatten()

        # Combine lap time and dynamic features
        new_sequence_entry = np.hstack((lap_time_normalized, dynamic_features_combined))

        # Place new entry at the end of the sequence
        driver.sequence[-1] = new_sequence_entry

    def update_driver_sequence(self, driver, lap_time):
        # Update the sequence with the new lap data
        driver.sequence = np.roll(driver.sequence, -1, axis=0)
        
        # Prepare new sequence entry
        dynamic_features_to_scale = [
            col for col in self.preprocessor.race_features.dynamic_features 
            if col not in ['tire_compound']
        ]
        tire_compound_value = driver.dynamic_features['tire_compound']
        other_dynamic_feature_values = np.array([
            driver.dynamic_features[feature] for feature in dynamic_features_to_scale
        ])
        other_dynamic_features_scaled = self.preprocessor.dynamic_scaler.transform(
            other_dynamic_feature_values.reshape(1, -1)
        ).flatten()
        
        dynamic_features_combined = np.hstack((tire_compound_value, other_dynamic_features_scaled))
        
        lap_time_normalized = self.preprocessor.lap_time_scaler.transform([[lap_time]]).flatten()
        
        # Combine lap time and dynamic features
        new_sequence_entry = np.hstack((lap_time_normalized, dynamic_features_combined))
        
        # Place new entry at the end of the sequence
        driver.sequence[-1] = new_sequence_entry

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
        race_length = race.total_laps
        race_lap_data = []

        for lap in range(1, race_length + 1):
            lap_data = []
            for driver in race.drivers:
                # Prepare input data for prediction
                if self.model_type == 'lstm':
                    # Use the existing LSTM prediction code
                    sequence_tensor = torch.FloatTensor(driver.sequence).unsqueeze(0)
                    static_tensor = torch.FloatTensor(driver.static_features).unsqueeze(0)
                    predictions = self.model(sequence_tensor, static_tensor)
                    lap_time_pred = self.preprocessor.inverse_transform_lap_times(predictions.cpu().numpy())[0]
                else:
                    # Prepare input data for other model types
                    input_data = np.concatenate((driver.sequence.flatten(), driver.static_features))
                    input_data = input_data.reshape(1, -1)  # Reshape to 2D array

                    if self.model_type == 'linear_regression':
                        # Prepare input data for linear regression
                        sequence_flattened = driver.sequence.flatten()
                        static_features = driver.static_features
                        
                        # Select only the features used during training
                        selected_features = [...]  # List the feature names used for training the linear regression model
                        
                        # Create a DataFrame with the selected features
                        input_data = pd.DataFrame({feature: [value] for feature, value in zip(selected_features, np.concatenate((sequence_flattened, static_features)))})
                        
                        # Apply one-hot encoding to categorical features
                        input_data = pd.get_dummies(input_data, columns=[...])  # Specify the categorical feature names
                        
                        # Ensure the input data has the same features as the training data
                        missing_cols = set(X.columns) - set(input_data.columns)
                        for col in missing_cols:
                            input_data[col] = 0
                        input_data = input_data[X.columns]
                        
                        # Scale the input data using the same scaler used during training
                        input_data_scaled = scaler.transform(input_data)
                        
                        # Apply PCA transformation
                        input_data_pca = pca.transform(input_data_scaled)
                        
                        # Make predictions using the linear regression model
                        lap_time_pred = self.model.predict(input_data_pca)[0]
                    elif self.model_type == 'xgboost':
                        # XGBoost can handle raw input data
                        lap_time_pred = self.model.predict(input_data)[0]
                    elif self.model_type == 'random_forest':
                        # Random forest can handle raw input data
                        lap_time_pred = self.model.predict(input_data)[0]
                    else:
                        raise ValueError(f"Unsupported model type: {self.model_type}")

                # Update driver's lap time and race state
                driver.lap_times.append(lap_time_pred)
                driver.race_state['lap_number'] = lap
                driver.race_state['lap_time'] = lap_time_pred

                # Collect lap data for the driver
                lap_data.append({
                    'driver': driver.name,
                    'lap': lap,
                    'lap_time': lap_time_pred,
                    **driver.race_state
                })

                # Update driver's sequence for the next lap
                driver.sequence[:-1] = driver.sequence[1:]
                driver.sequence[-1] = np.append(lap_time_pred, list(driver.race_state.values()))

            # Sort drivers based on total elapsed time
            race.drivers.sort(key=lambda driver: sum(driver.lap_times))

            # Update race standings
            for i, driver in enumerate(race.drivers, start=1):
                driver.race_state['position'] = i

            # Collect lap data for all drivers
            race_lap_data.append(lap_data)

        return race_lap_data

def plot_race_positions(race):
    plt.figure(figsize=(12, 6))

    for driver in race.drivers:
        positions = race.lap_data[driver.driver_id]['positions']
        plt.plot(range(1, race.total_laps + 1), positions, label=driver.name)

    plt.gca().invert_yaxis()  # Position 1 at the top
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
        plt.plot(range(1, race.total_laps + 1), lap_times, label=driver.name)

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


