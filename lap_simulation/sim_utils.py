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
