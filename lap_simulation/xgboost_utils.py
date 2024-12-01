# xgboost_utils.py

import pickle
import logging
import pandas as pd

def load_pipeline(path: str):
    """
    Load the entire pipeline (preprocessor + model) from a pickle file.

    Args:
        path (str): Path to the pickle file.

    Returns:
        Pipeline: The loaded scikit-learn pipeline.
    """
    try:
        with open(path, 'rb') as f:
            pipeline = pickle.load(f)
        logging.info(f"Pipeline loaded from {path}")
        return pipeline
    except Exception as e:
        logging.error(f"Error loading pipeline from {path}: {e}")
        raise

def save_pipeline(pipeline, path: str):
    """
    Save the entire pipeline (preprocessor + model) to a pickle file.

    Args:
        pipeline (Pipeline): The scikit-learn pipeline to save.
        path (str): Path to save the pickle file.
    """
    try:
        with open(path, 'wb') as f:
            pickle.dump(pipeline, f)
        logging.info(f"Pipeline saved to {path}")
    except Exception as e:
        logging.error(f"Error saving pipeline to {path}: {e}")
        raise

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