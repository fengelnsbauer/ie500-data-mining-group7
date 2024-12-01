import pandas as pd
from typing import Dict, List, Tuple

def extract_pit_strategies(lap_times_df: pd.DataFrame, race_id: int) -> Dict[int, Dict]:
    """
    Extracts pit strategies for each driver in a given race.

    Args:
        lap_times_df (pd.DataFrame): DataFrame containing lap data.
        race_id (int): Identifier for the race.

    Returns:
        Dict[int, Dict]: A dictionary where each key is a driverId and the value is another
                         dictionary containing the starting compound and a list of pit stops.
                         Each pit stop is a tuple (lap_number, new_compound).
    """
    pit_strategies = {}
    
    # Filter data for the specific race
    race_df = lap_times_df[lap_times_df['raceId'] == race_id]
    
    # Get unique drivers in the race
    drivers = race_df['driverId'].unique()
    
    for driver_id in drivers:
        driver_df = race_df[race_df['driverId'] == driver_id].sort_values('lap')
        
        # Identify pit laps
        pit_laps = driver_df[driver_df['is_pit_lap'] == 1]['lap'].tolist()
        
        # Get tire compounds on pit laps (assuming compound after pit stop)
        compounds_on_pit = driver_df[driver_df['is_pit_lap'] == 1]['tire_compound'].tolist()
        
        # Starting compound: the compound before the first pit stop
        if not pit_laps:
            # No pit stops; assume the first compound is the starting compound
            starting_compound = driver_df.iloc[0]['tire_compound']
        else:
            first_pit_lap = pit_laps[0]
            # Compound before first pit lap is the compound used on the previous lap
            if first_pit_lap == 1:
                starting_compound = driver_df.iloc[0]['tire_compound']
            else:
                starting_compound = driver_df[race_df['lap'] == first_pit_lap - 1]['tire_compound'].values[0]
        
        # Compile pit strategy
        pit_strategy = list(zip(pit_laps, compounds_on_pit))
        
        pit_strategies[driver_id] = {
            'starting_compound': starting_compound,
            'pit_strategy': pit_strategy
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