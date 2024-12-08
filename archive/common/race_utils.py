# common/race_utils.py

import pandas as pd
from typing import Dict, List, Tuple
import logging

def extract_pit_strategies(lap_times_df: pd.DataFrame, race_id: int) -> Dict:
    """
    Extract pit strategies with better error handling.
    """
    pit_strategies = {}
    
    # Filter for the specific race
    race_df = lap_times_df[lap_times_df['raceId'] == race_id]
    
    # Get unique drivers in the race
    drivers = race_df['driverId'].unique()
    
    for driver_id in drivers:
        try:
            # Get driver's laps
            driver_df = race_df[race_df['driverId'] == driver_id]
            
            # Find pit stops
            pit_laps = driver_df[driver_df['is_pit_lap'] == 1]['lap'].tolist()
            
            # Get tire compound information
            try:
                # Try to get starting compound from first lap
                first_lap_compound = driver_df[driver_df['lap'] == 1]['tire_compound'].iloc[0]
                starting_compound = first_lap_compound if not pd.isna(first_lap_compound) else 2
            except (IndexError, KeyError):
                starting_compound = 2  # Default to medium if missing
            
            # Get compounds for each pit stop
            pit_strategy = []
            for pit_lap in pit_laps:
                try:
                    # Try to get compound after pit stop
                    next_lap_df = driver_df[driver_df['lap'] == pit_lap + 1]
                    if not next_lap_df.empty:
                        new_compound = next_lap_df['tire_compound'].iloc[0]
                        if not pd.isna(new_compound):
                            pit_strategy.append((pit_lap, int(new_compound)))
                        else:
                            pit_strategy.append((pit_lap, 2))  # Default to medium
                except (IndexError, KeyError):
                    pit_strategy.append((pit_lap, 2))  # Default to medium
            
            pit_strategies[driver_id] = {
                'starting_compound': starting_compound,
                'pit_strategy': pit_strategy
            }
            
        except Exception as e:
            logging.warning(f"Error extracting pit strategy for driver {driver_id} in race {race_id}: {str(e)}")
            # Use default strategy
            pit_strategies[driver_id] = {
                'starting_compound': 2,
                'pit_strategy': []
            }
    
    return pit_strategies

# Extract safety car periods using the updated function
def extract_safety_car_periods(lap_times_df: pd.DataFrame, race_id: int) -> List[Tuple[int, int]]:
    """
    Extracts safety car periods for a given race based on TrackStatus.

    Args:
        lap_times_df (pd.DataFrame): DataFrame containing lap data.
        race_id (int): Identifier for the race.

    Returns:
        List[Tuple[int, int]]: A list of (start_lap, end_lap) tuples indicating safety car periods.
    """
    safety_car_periods = []
    
    # Filter data for the specific race
    race_df = lap_times_df[lap_times_df['raceId'] == race_id].sort_values('lap')
    
    # Identify all laps with safety car (TrackStatus == 4)
    safety_car_laps = race_df[race_df['TrackStatus'] == 4]['lap'].tolist()
    
    if not safety_car_laps:
        return safety_car_periods  # No safety car periods in this race
    
    # Initialize the first period
    start_lap = safety_car_laps[0]
    end_lap = safety_car_laps[0]
    
    for lap in safety_car_laps[1:]:
        if lap == end_lap + 1:
            # Consecutive lap, extend the current period
            end_lap = lap
        else:
            # Non-consecutive lap, finalize the current period and start a new one
            safety_car_periods.append((start_lap, end_lap))
            start_lap = lap
            end_lap = lap
    
    # Append the last period
    safety_car_periods.append((start_lap, end_lap))
    
    return safety_car_periods

# Function to get race length (you can adjust this based on your data)
def get_race_length(race_id: int, lap_times_df: pd.DataFrame) -> int:
    """
    Get the actual race length for a given race ID from historical data.
    """
    race_laps = lap_times_df[lap_times_df['raceId'] == race_id]['lap'].max()
    if pd.isna(race_laps):
        # Fallback to a default length if race not found
        return 50
    return int(race_laps)

