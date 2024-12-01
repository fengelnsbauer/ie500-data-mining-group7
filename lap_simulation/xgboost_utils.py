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


def load_model_with_preprocessor(path: str):
    """
    Load the XGBoost model and preprocessor from a pickle file.

    Args:
        path (str): Path to the pickle file.

    Returns:
        tuple: (model, preprocessor)
    """
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = data['model']
        preprocessor = data['preprocessor']
        logging.info(f"Model and preprocessor loaded from {path}")
        return model, preprocessor
    except Exception as e:
        logging.error(f"Error loading model and preprocessor from {path}: {e}")
        raise

def save_model_with_preprocessor(model, preprocessor, path: str):
    """
    Save the model and preprocessor to a pickle file.

    Args:
        model: Trained XGBoost model.
        preprocessor: Preprocessing pipeline.
        path (str): Path to save the pickle file.
    """
    try:
        with open(path, 'wb') as f:
            pickle.dump({'model': model, 'preprocessor': preprocessor}, f)
        logging.info(f"Model and preprocessor saved to {path}")
    except Exception as e:
        logging.error(f"Error saving model and preprocessor to {path}: {e}")
        raise

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