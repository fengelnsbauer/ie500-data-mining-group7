import pandas as pd
import numpy as np
import re
from typing import Tuple, List
from sklearn.model_selection import GroupShuffleSplit
from thefuzz import process  # For fuzzy matching
import logging
import os


from meteostat import Daily, Stations, Hourly
from datetime import datetime, time  # Import datetime and time classes here
from common.requiredFeatures import F1RaceFeatures

# Configure logging
logging.basicConfig(level=logging.INFO)

na_values = ['\\N', 'NaN', '']

# Preprocessing function for strings
def preprocess_string(s):
    if isinstance(s, str):
        s = s.lower().strip()
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'grand prix', 'gp', s)
        s = re.sub(r'[^\w\s]', '', s)
        return s
    return s

# Data preparation functions
def load_raw_data():
    print(os.getcwd())
    try:
        lap_times = pd.read_csv('../../data/raw_data/lap_times.csv', na_values=na_values)
        drivers = pd.read_csv('../../data/raw_data/drivers.csv', na_values=na_values)
        races = pd.read_csv('../../data/raw_data/races.csv', na_values=na_values)
        circuits = pd.read_csv('../../data/raw_data/circuits.csv', na_values=na_values)
        pit_stops = pd.read_csv('../../data/raw_data/pit_stops.csv', na_values=na_values)
        pit_stops.rename(columns={'milliseconds': 'pitstop_milliseconds'}, inplace=True)
        results = pd.read_csv('../../data/raw_data/results.csv', na_values=na_values)
        results.rename(columns={'milliseconds': 'racetime_milliseconds'}, inplace=True)
        print(results.columns)
        qualifying = pd.read_csv('../../data/raw_data/qualifying.csv', na_values=na_values)
        status = pd.read_csv('../../data/raw_data/status.csv', na_values=na_values)
        weather_data = pd.read_csv('../../data/raw_data/ff1_weather.csv', na_values=na_values)
        practice_sessions = pd.read_csv('../../data/raw_data/ff1_laps_intervals.csv', na_values=na_values)
        tire_data = pd.read_csv('../../data/raw_data/ff1_laps_intervals.csv', na_values=na_values)
        constructors = pd.read_csv('../../data/raw_data/constructors.csv', na_values=na_values)
        constructor_standings = pd.read_csv('../../data/raw_data/constructor_standings.csv', na_values=na_values)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e.filename}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading data: {str(e)}")
        raise

    return {
        'lap_times': lap_times,
        'drivers': drivers,
        'races': races,
        'circuits': circuits,
        'pit_stops': pit_stops,
        'results': results,
        'qualifying': qualifying,
        'status': status,
        'weather_data': weather_data,
        'practice_sessions': practice_sessions,
        'tire_data': tire_data,
        'constructors': constructors,
        'constructor_standings': constructor_standings
    }

def preprocess_data() -> pd.DataFrame:
    data = load_raw_data()
    
    # Extract individual DataFrames and print initial shapes
    lap_times = data['lap_times']
    print(f"\nInitial data sizes:")
    print(f"Lap times: {lap_times.shape}")
    drivers = data['drivers']
    races = data['races']
    circuits = data['circuits']
    pit_stops = data['pit_stops']
    results = data['results']
    qualifying = data['qualifying']
    status = data['status']
    weather_data = data['weather_data']
    practice_sessions = data['practice_sessions']
    tire_data = data['tire_data']
    constructors = data['constructors']
    constructor_standings = data['constructor_standings']

    # Convert intervals to ms
    time_columns = ['CumRaceTime', 'GapToLeader', 'IntervalToPositionAhead']
    practice_sessions = convert_time_intervals_to_milliseconds(practice_sessions, time_columns)
    tire_data = convert_time_intervals_to_milliseconds(tire_data, time_columns)

    # Track data transformations
    laps = merge_dataframes(lap_times, drivers, races, circuits, results, status, pit_stops)
    print(f"\nAfter initial merge: {laps.shape} - All lap data merged")
    
    laps = filter_laps_by_year(laps, start_year=2018)
    print(f"After year filtering (>=2018): {laps.shape}")
    
    laps = add_constructor_info(laps, constructors, constructor_standings, results)
    print(f"After adding constructor info: {laps.shape}")
    
    laps = add_circuit_info(laps, circuits)
    print(f"After adding circuit info: {laps.shape}")
    
    laps.sort_values(['raceId', 'driverId', 'lap'], inplace=True)
    laps['cumulative_milliseconds'] = laps.groupby(['raceId', 'driverId'])['milliseconds'].cumsum()
    laps['seconds_from_start'] = laps['cumulative_milliseconds'] / 1000.0
    
    laps = add_weather_info(laps, races, weather_data)
    print(f"After adding weather info: {laps.shape}")

    # Handle missing weather data
    missing_race_ids = get_races_with_missing_weather(laps)
    if missing_race_ids:
        print(f"\nFetching weather for {len(missing_race_ids)} races: {missing_race_ids}")
        fetched_weather_df = fetch_missing_weather_data(races, circuits, missing_race_ids)
        laps = merge_fetched_weather(laps, fetched_weather_df)
        
    laps['TrackTemp'].fillna(25.0, inplace=True)
    laps['AirTemp'].fillna(20.0, inplace=True)
    laps['Humidity'].fillna(50.0, inplace=True)

    laps = add_tire_info(laps, tire_data, drivers, races)
    print(f"After adding tire info: {laps.shape}")
    
    laps = add_practice_session_info(laps, practice_sessions, drivers, races)
    print(f"After adding practice info: {laps.shape}")
    
    laps = clean_time_intervals(laps)
    print(f"After cleaning time intervals: {laps.shape}")
    
    laps = enhance_driver_attributes(laps, results, races)
    print(f"After enhancing driver attributes: {laps.shape}")
    
    laps = add_dynamic_features(laps)
    print(f"After adding dynamic features: {laps.shape}")
    
    # Add new interval calculations
    print("\nCalculating race intervals...")
    laps = calculate_race_intervals(laps)
    laps = validate_intervals(laps)
    print(f"After calculating race intervals: {laps.shape}")
    
    print("\nBefore outlier removal:")
    
    print("\nBefore outlier removal:")
    print(f"Unique races: {laps['raceId'].nunique()}")
    print(f"Unique drivers per race: {laps.groupby('raceId')['driverId'].nunique().describe()}")
    
    laps = remove_lap_time_outliers(laps)
    #laps, special_laps = remove_lap_time_outliers(laps)
    print(f"\nAfter removing outliers: {laps.shape}")
    #print(f"Special laps removed: {special_laps.shape}")
    
    laps = drop_unnecessary_columns(laps)
    print(f"After dropping unnecessary columns: {laps.shape}")
    
    laps = remove_duplicate_columns(laps, ['is_pit_lap', 'driver_circuit_skill', 'driver_risk_taking', 
                                         'track_position', 'constructor_performance', 
                                         'IntervalToPositionAhead', 'GapToLeader'])
    print(f"After removing duplicate columns: {laps.shape}")
    
    print("\nBefore handling missing values:")
    print(f"Unique races: {laps['raceId'].nunique()}")
    print(f"Unique drivers per race: {laps.groupby('raceId')['driverId'].nunique().describe()}")
    
    laps = handle_missing_values(laps)
    print(f"\nAfter handling missing values: {laps.shape}")
    print(f"Final unique races: {laps['raceId'].nunique()}")
    print(f"Final unique drivers per race: {laps.groupby('raceId')['driverId'].nunique().describe()}")
    
    #laps = save_auxiliary_data(laps, drivers, races, special_laps)
    laps = save_auxiliary_data(laps, drivers, races)
    print(f"\nFinal dataset: {laps.shape}")

    return laps 


def merge_dataframes(
    lap_times: pd.DataFrame,
    drivers: pd.DataFrame,
    races: pd.DataFrame,
    circuits: pd.DataFrame,
    results: pd.DataFrame,
    status: pd.DataFrame,
    pit_stops: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges the primary DataFrames to form the base laps DataFrame.

    Returns:
        pd.DataFrame: The merged laps DataFrame.
    """
    # Convert date columns to datetime
    races['date'] = pd.to_datetime(races['date'])
    results['date'] = results['raceId'].map(races.set_index('raceId')['date'])
    lap_times['date'] = lap_times['raceId'].map(races.set_index('raceId')['date'])

    # Merge DataFrames
    laps = lap_times.merge(drivers, on='driverId', how='left')
    laps = laps.merge(races, on='raceId', how='left', suffixes=('', '_race'))
    laps.rename(columns={'quali_time' : 'quali_date_time'}, inplace=True)
    laps = laps.merge(circuits, on='circuitId', how='left')
    
    # **Include 'constructorId' in the merge from results**
    laps = laps.merge(
        results[['raceId', 'driverId', 'positionOrder', 'grid', 'fastestLap', 'statusId']],
        on=['raceId', 'driverId'],
        how='left'
    )
    
    laps = laps.merge(status[['statusId', 'status']], on='statusId', how='left')
    laps = laps.merge(
        pit_stops[['raceId', 'driverId', 'lap', 'pitstop_milliseconds']],
        on=['raceId', 'driverId', 'lap'],
        how='left'
    )
    # Fill missing pit stop durations with 0 (no pit stop)
    laps['pitstop_milliseconds'].fillna(0, inplace=True)
    return laps


def filter_laps_by_year(laps: pd.DataFrame, start_year: int = 2018) -> pd.DataFrame:
    """
    Filters laps DataFrame to include data from the specified start year onwards.
    
    Returns:
        pd.DataFrame: The filtered laps DataFrame.
    """
    laps = laps[laps['year'] >= start_year]
    return laps

def add_constructor_info(
    laps: pd.DataFrame,
    constructors: pd.DataFrame,
    constructor_standings: pd.DataFrame,
    results: pd.DataFrame  # Include 'results' as a parameter
) -> pd.DataFrame:
    """
    Adds constructor information and performance metrics to the laps DataFrame.
    
    Returns:
        pd.DataFrame: The laps DataFrame with constructor information.
    """
    # Map driverId to constructorId from 'results'
    driver_constructor = results[['raceId', 'driverId', 'constructorId']].drop_duplicates()
    laps = laps.merge(driver_constructor, on=['raceId', 'driverId'], how='left')

    # Merge constructors with 'laps' to get constructor information
    constructors_info = constructors[['constructorId', 'name', 'nationality']]
    constructors_info.rename(columns={'name': 'constructor_name', 'nationality': 'constructor_nationality'}, inplace=True)
    laps = laps.merge(constructors_info, on='constructorId', how='left')

    # Add constructor performance metrics
    constructor_standings_latest = (
        constructor_standings.sort_values('raceId', ascending=False)
        .drop_duplicates('constructorId')
    )
    constructor_standings_latest = constructor_standings_latest[['constructorId', 'points', 'position']]
    constructor_standings_latest.rename(
        columns={'points': 'constructor_points', 'position': 'constructor_position'}, inplace=True
    )
    laps = laps.merge(constructor_standings_latest, on='constructorId', how='left')

    # Fill missing constructor performance data
    laps['constructor_points'].fillna(laps['constructor_points'].mean(), inplace=True)
    laps['constructor_position'].fillna(laps['constructor_position'].max(), inplace=True)

    # Add constructor performance as a static feature
    laps['constructor_performance'] = laps['constructor_points']
    return laps


def add_circuit_info(laps: pd.DataFrame, circuits: pd.DataFrame) -> pd.DataFrame:
    """
    Adds circuit characteristics to the laps DataFrame using circuitId.
    """
    # Circuit length (in kilometers) and type mapping using circuitId
    circuit_data = {
        1: {'length': 5.278, 'type': 'Street'},        # Albert Park
        2: {'length': 5.543, 'type': 'Permanent'},     # Sepang
        3: {'length': 5.412, 'type': 'Permanent'},     # Bahrain
        4: {'length': 4.675, 'type': 'Permanent'},     # Catalunya
        5: {'length': 5.338, 'type': 'Permanent'},     # Istanbul
        6: {'length': 3.337, 'type': 'Street'},        # Monaco
        7: {'length': 4.361, 'type': 'Hybrid'},        # Gilles Villeneuve
        8: {'length': 4.411, 'type': 'Permanent'},     # Magny-Cours
        9: {'length': 5.891, 'type': 'Permanent'},     # Silverstone
        10: {'length': 4.574, 'type': 'Permanent'},    # Hockenheim
        11: {'length': 4.381, 'type': 'Permanent'},    # Hungaroring
        12: {'length': 5.419, 'type': 'Street'},       # Valencia
        13: {'length': 7.004, 'type': 'Permanent'},    # Spa
        14: {'length': 5.793, 'type': 'Permanent'},    # Monza
        15: {'length': 5.063, 'type': 'Street'},       # Marina Bay
        16: {'length': 4.563, 'type': 'Permanent'},    # Fuji
        17: {'length': 5.451, 'type': 'Permanent'},    # Shanghai
        18: {'length': 4.309, 'type': 'Permanent'},    # Interlagos
        19: {'length': 4.192, 'type': 'Permanent'},    # Indianapolis
        20: {'length': 5.148, 'type': 'Permanent'},    # Nurburgring
        21: {'length': 4.909, 'type': 'Permanent'},    # Imola
        22: {'length': 5.807, 'type': 'Permanent'},    # Suzuka
        24: {'length': 5.281, 'type': 'Permanent'},    # Yas Marina
        32: {'length': 4.304, 'type': 'Permanent'},    # Rodriguez
        34: {'length': 5.842, 'type': 'Permanent'},    # Paul Ricard
        39: {'length': 4.259, 'type': 'Permanent'},    # Zandvoort
        69: {'length': 5.513, 'type': 'Permanent'},    # Circuit of the Americas
        70: {'length': 4.318, 'type': 'Permanent'},    # Red Bull Ring
        71: {'length': 5.848, 'type': 'Hybrid'},       # Sochi
        73: {'length': 6.003, 'type': 'Street'},       # Baku
        75: {'length': 4.653, 'type': 'Permanent'},    # Portimao
        76: {'length': 5.245, 'type': 'Permanent'},    # Mugello
        77: {'length': 6.175, 'type': 'Street'},       # Jeddah
        78: {'length': 5.380, 'type': 'Permanent'},    # Losail
        79: {'length': 5.412, 'type': 'Street'},       # Miami
        80: {'length': 6.201, 'type': 'Street'},       # Las Vegas
    }
    
    # Create series for circuit length and type
    circuits['circuit_length'] = circuits['circuitId'].map(
        {k: v['length'] for k, v in circuit_data.items()}
    ).fillna(5.000)  # Default length of 5km for any missing circuits
    
    circuits['circuit_type'] = circuits['circuitId'].map(
        {k: v['type'] for k, v in circuit_data.items()}
    ).fillna('Permanent')  # Default type of Permanent for any missing circuits
    
    # Merge with laps DataFrame
    laps = laps.merge(
        circuits[['circuitId', 'circuit_length', 'circuit_type']],
        on='circuitId',
        how='left'
    )
    
    # Encode circuit_type as categorical
    circuit_type_mapping = {'Permanent': 0, 'Street': 1, 'Hybrid': 2}
    laps['circuit_type_encoded'] = laps['circuit_type'].map(circuit_type_mapping)
    
    return laps

def get_races_with_missing_weather(laps: pd.DataFrame) -> List[int]:
    """
    Identifies races where weather data is missing.
    """
    # Assuming 'TrackTemp' is a key weather feature
    races_with_weather = laps.groupby('raceId')['TrackTemp'].apply(lambda x: x.notnull().all())
    missing_race_ids = races_with_weather[races_with_weather == False].index.tolist()
    logging.info(f"Races with missing weather data: {missing_race_ids}")
    return missing_race_ids



def fetch_missing_weather_data(races: pd.DataFrame, circuits: pd.DataFrame, missing_race_ids: List[int]) -> pd.DataFrame:
    """
    Fetches weather data for races where it's missing.
    """
    # Filter races to only those with missing weather data
    missing_races = races[races['raceId'].isin(missing_race_ids)]

    # Get race locations
    race_locations = missing_races.merge(
        circuits[['circuitId', 'lat', 'lng', 'alt']],
        on='circuitId',
        how='left'
    )
    race_locations.rename(columns={'lat': 'latitude', 'lng': 'longitude'}, inplace=True)

    weather_data_list = []
    for _, row in race_locations.iterrows():
        race_id = row['raceId']
        date = pd.to_datetime(row['date'])
        latitude = row['latitude']
        longitude = row['longitude']

        weather_data = fetch_weather_data_for_race(race_id, date, latitude, longitude)
        if weather_data:
            weather_data_list.append(weather_data)

    fetched_weather_df = pd.DataFrame(weather_data_list)
    return fetched_weather_df


def fetch_weather_data_for_race(race_id: int, date: pd.Timestamp, latitude: float, longitude: float) -> dict:
    # Find the nearest weather station
    stations = Stations()
    stations = stations.nearby(latitude, longitude)
    station = stations.fetch(1)
    
    if station.empty:
        print(f"No weather station found for race {race_id}")
        return None
    
    station_id = station.index[0]
    
    # Fetch hourly weather data for the race date
    start = datetime.combine(date.date(), time(12, 0))  # Adjust time as needed
    end = datetime.combine(date.date(), time(18, 0))    # Adjust time as needed
    data = Hourly(station_id, start, end)
    data = data.fetch()
    
    if data.empty:
        print(f"No weather data available for race {race_id}")
        return None
    
    # Calculate average values
    weather_data = {
        'raceId': race_id,
        'AirTemp': data['temp'].mean(),
        'Humidity': data['rhum'].mean(),
        'TrackTemp': data['temp'].mean() + 10  # Approximate
    }
    
    return weather_data

def convert_time_intervals_to_milliseconds(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Converts time interval columns to milliseconds.
    
    Parameters:
        df (pd.DataFrame): The dataframe containing time interval columns.
        columns (List[str]): List of column names to convert.

    Returns:
        pd.DataFrame: The dataframe with new '_ms' columns containing time in milliseconds.
    """
    for col in columns:
        if col in df.columns:
            df[col + '_ms'] = pd.to_timedelta(df[col], errors='coerce').dt.total_seconds() * 1000
    return df

def merge_fetched_weather(laps: pd.DataFrame, fetched_weather_df: pd.DataFrame) -> pd.DataFrame:
    # Merge fetched weather data
    laps = laps.merge(
        fetched_weather_df[['raceId', 'AirTemp', 'Humidity', 'TrackTemp']],
        on='raceId',
        how='left',
        suffixes=('', '_fetched')
    )
    
    # Fill missing values in laps with fetched data
    for col in ['AirTemp', 'Humidity', 'TrackTemp']:
        laps[col] = laps[col].fillna(laps[f'{col}_fetched'])
        laps.drop(columns=[f'{col}_fetched'], inplace=True)
    
    return laps

def fill_qualifying_times(laps: pd.DataFrame, qualifying: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing qualifying times based on F1's elimination format:
    - Q1: All drivers participate, bottom 5 eliminated
    - Q2: 15 drivers participate, bottom 5 eliminated
    - Q3: Top 10 drivers set final times
    
    Returns:
        pd.DataFrame: DataFrame with properly filled qualifying times
    """
    # Process qualifying data for each race
    quali_times = []
    
    for race_id in laps['raceId'].unique():
        race_quali = qualifying[qualifying['raceId'] == race_id]
        
        if race_quali.empty:
            continue
            
        # Sort drivers by their best times in each session
        q1_times = race_quali[race_quali['q1'].notna()].sort_values('q1')
        q2_times = race_quali[race_quali['q2'].notna()].sort_values('q2')
        q3_times = race_quali[race_quali['q3'].notna()].sort_values('q3')
        
        # Get cutoff times for each session
        q1_cutoff = q1_times['q1'].iloc[-6] if len(q1_times) > 15 else None  # P16 time
        q2_cutoff = q2_times['q2'].iloc[-6] if len(q2_times) > 10 else None  # P11 time
        
        for driver_id in race_quali['driverId'].unique():
            driver_quali = race_quali[race_quali['driverId'] == driver_id]
            
            if driver_quali.empty:
                continue
                
            # Determine final qualifying time based on elimination rules
            if not pd.isna(driver_quali['q3'].iloc[0]):
                quali_time = driver_quali['q3'].iloc[0]
                quali_session = 'Q3'
            elif not pd.isna(driver_quali['q2'].iloc[0]):
                quali_time = driver_quali['q2'].iloc[0]
                quali_session = 'Q2'
            elif not pd.isna(driver_quali['q1'].iloc[0]):
                quali_time = driver_quali['q1'].iloc[0]
                quali_session = 'Q1'
            else:
                # No time set - use appropriate estimation
                if q1_cutoff is not None:
                    # Estimate time as slightly slower than Q1 cutoff
                    quali_time = q1_cutoff * 1.02  # 2% slower than cutoff
                    quali_session = 'Q1_EST'
                else:
                    # No reference times available - use teammate or session median
                    teammate_time = race_quali[
                        (race_quali['constructorId'] == driver_quali['constructorId'].iloc[0]) &
                        (race_quali['driverId'] != driver_id)
                    ]['q1'].median()
                    
                    if not pd.isna(teammate_time):
                        quali_time = teammate_time * 1.01  # Slightly slower than teammate
                    else:
                        quali_time = race_quali['q1'].median()  # Session median as last resort
                    quali_session = 'EST'
            
            quali_times.append({
                'raceId': race_id,
                'driverId': driver_id,
                'quali_time': quali_time,
                'quali_session': quali_session
            })
    
    # Convert to DataFrame and merge with laps
    quali_times_df = pd.DataFrame(quali_times)
    laps = laps.merge(
        quali_times_df,
        on=['raceId', 'driverId'],
        how='left'
    )
    
    # Handle any remaining missing values
    median_time = laps['quali_time'].median()
    laps['quali_time'] = laps['quali_time'].fillna(median_time)
    laps['quali_session'] = laps['quali_session'].fillna('MISSING')
    
    return laps


def add_weather_info(laps: pd.DataFrame, races: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merges weather data with laps data.
    
    Returns:
        pd.DataFrame: The laps DataFrame with weather information.
    """
    # Process weather data for race sessions
    weather_data = weather_data[weather_data['SessionName'] == 'R'].copy()
    weather_data['seconds_from_start'] = pd.to_timedelta(weather_data['Time'], errors='coerce').dt.total_seconds()

    # Preprocess event names
    weather_data['EventName_clean'] = weather_data['EventName'].apply(preprocess_string)
    races['name_clean'] = races['name'].apply(preprocess_string)

    # Merge with races to get raceId
    weather_data = weather_data.merge(
        races[['raceId', 'year', 'name_clean']],
        left_on=['EventName_clean', 'Year'],
        right_on=['name_clean', 'year'],
        how='left',
        indicator=True
    )

    # Handle unmatched races
    unmatched_weather = weather_data[weather_data['_merge'] == 'left_only']
    if not unmatched_weather.empty:
        logging.warning("Unmatched races in weather data:")
        logging.warning(unmatched_weather[['EventName', 'Year']].drop_duplicates())

        # Use fuzzy matching to find possible matches
        unmatched_event_names = unmatched_weather['EventName_clean'].unique()
        races_event_names = races['name_clean'].unique()
        event_name_mapping = {}
        for event in unmatched_event_names:
            match, score = process.extractOne(event, races_event_names)
            if score >= 80:  # Adjust the threshold as needed
                event_name_mapping[event] = match
                logging.info(f"Mapping '{event}' to '{match}' with score {score}")
            else:
                logging.warning(f"No suitable match found for '{event}'")

        # Apply the mappings
        for event, match in event_name_mapping.items():
            weather_data.loc[weather_data['EventName_clean'] == event, 'EventName_clean'] = match

        # Retry the merge after applying mappings
        weather_data = weather_data.merge(
            races[['raceId', 'year', 'name_clean']],
            left_on=['EventName_clean', 'Year'],
            right_on=['name_clean', 'year'],
            how='left',
            indicator=False
        )

    # Drop unnecessary columns
    weather_data.drop(['EventName_clean', 'name_clean', '_merge'], axis=1, inplace=True, errors='ignore')

    # Clean up columns
    weather_cols = ['raceId', 'seconds_from_start', 'TrackTemp', 'AirTemp', 'Humidity']
    weather_data = weather_data[weather_cols]

    # Match weather data to laps
    laps = match_weather_to_laps(laps, races, weather_data)
    return laps

def match_weather_to_laps(laps: pd.DataFrame, races: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
    """
    Matches weather data to each lap based on time from race start.
    """
    merged_laps_list = []

    for race_id in laps['raceId'].unique():
        race_laps = laps[laps['raceId'] == race_id].copy()
        race_weather = weather_data[weather_data['raceId'] == race_id]

        if not race_weather.empty:
            # Convert 'seconds_from_start' to numeric
            race_laps['seconds_from_start'] = pd.to_numeric(race_laps['seconds_from_start'], errors='coerce')
            race_weather['seconds_from_start'] = pd.to_numeric(race_weather['seconds_from_start'], errors='coerce')
            
            # Drop NaNs
            race_laps.dropna(subset=['seconds_from_start'], inplace=True)
            race_weather.dropna(subset=['seconds_from_start'], inplace=True)
            
            if not race_weather.empty and not race_laps.empty:
                # Perform asof merge
                merged = pd.merge_asof(
                    race_laps.sort_values('seconds_from_start'),
                    race_weather[['seconds_from_start', 'TrackTemp', 'AirTemp', 'Humidity']].sort_values('seconds_from_start'),
                    on='seconds_from_start',
                    direction='nearest'
                )
                merged_laps_list.append(merged)
            else:
                # If either is empty after dropping NaNs, keep the laps without weather data
                merged_laps_list.append(race_laps)
        else:
            # No weather data for this race, keep the laps without weather data
            merged_laps_list.append(race_laps)

    # Concatenate all race laps
    laps_with_weather = pd.concat(merged_laps_list, ignore_index=True)

    return laps_with_weather

def calculate_race_intervals(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate GapToLeader and IntervalToPositionAhead with enhanced edge case handling.
    
    Args:
        laps (pd.DataFrame): DataFrame containing lap data with cumulative_milliseconds and positionOrder
        
    Returns:
        pd.DataFrame: DataFrame with calculated GapToLeader_ms and IntervalToPositionAhead_ms
    """
    laps = laps.copy()
    
    # Process each race separately
    for race_id in laps['raceId'].unique():
        race_laps = laps[laps['raceId'] == race_id]
        
        # Get the total number of laps for this race
        max_laps = race_laps['lap'].max()
        
        # Process each lap separately
        for lap_num in range(1, max_laps + 1):
            lap_data = race_laps[race_laps['lap'] == lap_num].copy()
            
            if lap_data.empty:
                continue
                
            # Sort by position to ensure correct calculations
            lap_data = lap_data.sort_values('positionOrder')
            
            # Handle race start (Lap 1)
            if lap_num == 1:
                # Use grid positions to estimate gaps at race start
                grid_gaps = estimate_start_gaps(lap_data)
                lap_data['GapToLeader_ms'] = grid_gaps['GapToLeader_ms']
                lap_data['IntervalToPositionAhead_ms'] = grid_gaps['IntervalToPositionAhead_ms']
            else:
                # Calculate standard gaps for regular racing laps
                leader_time = lap_data.iloc[0]['cumulative_milliseconds']
                lap_data['GapToLeader_ms'] = lap_data['cumulative_milliseconds'] - leader_time
                
                # Calculate intervals with safety checks
                lap_data['IntervalToPositionAhead_ms'] = calculate_safe_intervals(lap_data)
            
            # Handle pit stops and special cases
            lap_data = handle_special_cases(lap_data)
            
            # Update the main DataFrame
            laps.loc[(laps['raceId'] == race_id) & (laps['lap'] == lap_num), 
                    ['GapToLeader_ms', 'IntervalToPositionAhead_ms']] = lap_data[
                        ['GapToLeader_ms', 'IntervalToPositionAhead_ms']].values
    
    # Apply final cleaning and validation
    laps = clean_and_validate_intervals(laps)
    
    return laps

def estimate_start_gaps(lap_data: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate gaps for the race start based on grid positions.
    """
    gaps = pd.DataFrame(index=lap_data.index)
    
    # Estimate ~2 seconds gap between each grid position
    # This is a simplification but provides reasonable initial values
    gaps['GapToLeader_ms'] = (lap_data['grid'] - 1) * 2000
    
    # Calculate intervals between consecutive positions
    gaps['IntervalToPositionAhead_ms'] = 2000  # Standard gap between positions
    gaps.loc[lap_data['grid'] == 1, 'IntervalToPositionAhead_ms'] = 0  # No gap for pole position
    
    return gaps

def calculate_safe_intervals(lap_data: pd.DataFrame) -> pd.Series:
    """
    Calculate intervals with safety checks for anomalies.
    """
    intervals = lap_data['cumulative_milliseconds'].diff()
    
    # Handle anomalies (e.g., very large gaps or negative intervals)
    median_interval = intervals.median()
    max_reasonable_gap = median_interval * 3  # 3x median as maximum reasonable gap
    
    # Clean unreasonable values
    intervals = intervals.clip(lower=0, upper=max_reasonable_gap)
    
    # Set interval to 0 for race leader
    intervals.iloc[0] = 0
    
    return intervals

def handle_special_cases(lap_data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle special cases like pit stops, blue flags, and accidents.
    """
    # Handle pit stops
    pit_stop_mask = lap_data['is_pit_lap'] == 1
    if pit_stop_mask.any():
        # Adjust intervals for pitting cars based on typical pit loss
        typical_pit_loss = 20000  # 20 seconds in milliseconds
        lap_data.loc[pit_stop_mask, 'GapToLeader_ms'] += typical_pit_loss
        
        # Recalculate intervals around pitting cars
        pit_indices = pit_stop_mask[pit_stop_mask].index
        for idx in pit_indices:
            if idx > 0:  # Not first position
                position = lap_data.index.get_loc(idx)
                # Adjust interval to car ahead
                if position > 0:
                    gap_ahead = lap_data.iloc[position]['cumulative_milliseconds'] - lap_data.iloc[position-1]['cumulative_milliseconds']
                    lap_data.at[idx, 'IntervalToPositionAhead_ms'] = max(0, gap_ahead)
                # Adjust interval to car behind
                if position < len(lap_data) - 1:
                    gap_behind = lap_data.iloc[position+1]['cumulative_milliseconds'] - lap_data.iloc[position]['cumulative_milliseconds']
                    lap_data.iloc[position+1, lap_data.columns.get_loc('IntervalToPositionAhead_ms')] = max(0, gap_behind)
    
    return lap_data

def clean_and_validate_intervals(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleaning and validation of interval data.
    """
    # Remove physically impossible values
    laps.loc[laps['GapToLeader_ms'] < 0, 'GapToLeader_ms'] = 0
    laps.loc[laps['IntervalToPositionAhead_ms'] < 0, 'IntervalToPositionAhead_ms'] = 0
    
    # Handle remaining missing values through interpolation within each driver's race
    for (race_id, driver_id), group in laps.groupby(['raceId', 'driverId']):
        mask = (laps['raceId'] == race_id) & (laps['driverId'] == driver_id)
        
        # Linear interpolation for small gaps
        laps.loc[mask, 'GapToLeader_ms'] = group['GapToLeader_ms'].interpolate(
            method='linear', limit=3)  # Only interpolate up to 3 consecutive missing values
        laps.loc[mask, 'IntervalToPositionAhead_ms'] = group['IntervalToPositionAhead_ms'].interpolate(
            method='linear', limit=3)
        
        # Forward fill any remaining gaps
        laps.loc[mask, 'GapToLeader_ms'] = group['GapToLeader_ms'].ffill()
        laps.loc[mask, 'IntervalToPositionAhead_ms'] = group['IntervalToPositionAhead_ms'].ffill()
    
    return laps

def validate_intervals(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the interval data by removing physically impossible values.
    
    Args:
        laps (pd.DataFrame): DataFrame with calculated intervals
        
    Returns:
        pd.DataFrame: DataFrame with validated intervals
    """
    laps = laps.copy()
    
    # Remove negative intervals (physically impossible)
    laps.loc[laps['GapToLeader_ms'] < 0, 'GapToLeader_ms'] = np.nan
    laps.loc[laps['IntervalToPositionAhead_ms'] < 0, 'IntervalToPositionAhead_ms'] = np.nan
    
    # Remove unreasonably large gaps (e.g., more than 2 laps behind)
    median_lap_time = laps.groupby('circuitId')['milliseconds'].median()
    for circuit_id in laps['circuitId'].unique():
        max_gap = 2 * median_lap_time[circuit_id]  # 2 laps worth of time
        circuit_mask = laps['circuitId'] == circuit_id
        
        laps.loc[circuit_mask & (laps['GapToLeader_ms'] > max_gap), 'GapToLeader_ms'] = np.nan
        laps.loc[circuit_mask & (laps['IntervalToPositionAhead_ms'] > max_gap), 'IntervalToPositionAhead_ms'] = np.nan
    
    # Fill remaining NaN values with interpolation within each race/driver combination
    for (race_id, driver_id), group in laps.groupby(['raceId', 'driverId']):
        mask = (laps['raceId'] == race_id) & (laps['driverId'] == driver_id)
        
        # Interpolate within the group
        laps.loc[mask, 'GapToLeader_ms'] = group['GapToLeader_ms'].interpolate(method='linear')
        laps.loc[mask, 'IntervalToPositionAhead_ms'] = group['IntervalToPositionAhead_ms'].interpolate(method='linear')
        
        # Fill any remaining NaN values at the start/end of groups
        laps.loc[mask, 'GapToLeader_ms'] = laps.loc[mask, 'GapToLeader_ms'].ffill().bfill()
        laps.loc[mask, 'IntervalToPositionAhead_ms'] = laps.loc[mask, 'IntervalToPositionAhead_ms'].ffill().bfill()
    
    return laps

def validate_gap_calculations(laps: pd.DataFrame) -> dict:
    """
    Validate calculated gaps and cumulative laptimes against known values.
    
    Args:
        laps (pd.DataFrame): DataFrame containing both original and calculated gaps
    
    Returns:
        dict: Validation metrics and statistics
    """
    validation_df = laps.copy()
    results = {}
    
    # Process each race separately
    for race_id in validation_df['raceId'].unique():
        race_laps = validation_df[validation_df['raceId'] == race_id].copy()
        
        # Calculate gaps for each lap
        for lap_num in race_laps['lap'].unique():
            lap_data = race_laps[race_laps['lap'] == lap_num].sort_values('positionOrder')
            
            # Calculate gaps
            leader_time = lap_data['cumulative_milliseconds'].iloc[0]
            lap_data['calculated_gap'] = lap_data['cumulative_milliseconds'] - leader_time
            lap_data['calculated_interval'] = lap_data['cumulative_milliseconds'].diff()
            lap_data.loc[lap_data['positionOrder'] == 1, 'calculated_interval'] = 0
            
            # Calculate cumulative time differences
            lap_data['calculated_cumtime'] = lap_data.groupby('driverId')['milliseconds'].cumsum()
            
            # Store calculations
            validation_df.loc[lap_data.index, 'calculated_gap'] = lap_data['calculated_gap']
            validation_df.loc[lap_data.index, 'calculated_interval'] = lap_data['calculated_interval']
            validation_df.loc[lap_data.index, 'calculated_cumtime'] = lap_data['calculated_cumtime']
    
    # Compare with original non-null values
    comparisons = {
        'gap': ('GapToLeader_ms', 'calculated_gap'),
        'interval': ('IntervalToPositionAhead_ms', 'calculated_interval'),
        'cumtime': ('cumulative_milliseconds', 'calculated_cumtime')
    }
    
    for comparison_name, (original_col, calculated_col) in comparisons.items():
        valid_data = validation_df[validation_df[original_col].notna()][[original_col, calculated_col]]
        
        if not valid_data.empty:
            diff = valid_data[original_col] - valid_data[calculated_col]
            
            results[f'{comparison_name}_validation'] = {
                'mean_difference_ms': diff.mean(),
                'std_difference_ms': diff.std(),
                'max_difference_ms': diff.abs().max(),
                'median_difference_ms': diff.median(),
                'within_100ms': (diff.abs() <= 100).mean() * 100,  # % within 100ms
                'within_1000ms': (diff.abs() <= 1000).mean() * 100,  # % within 1s
                'total_comparisons': len(diff),
                'correlation': valid_data[original_col].corr(valid_data[calculated_col])
            }
    
    # Add consistency checks
    results['consistency_checks'] = {
        'gap_equals_cumtime_diff': (
            abs(validation_df['calculated_gap'] - 
                (validation_df['calculated_cumtime'] - validation_df.groupby('raceId')['calculated_cumtime'].transform('min')))
            .mean()
        ),
        'intervals_sum_to_gap': (
            abs(validation_df.groupby(['raceId', 'lap'])['calculated_interval'].cumsum() - 
                validation_df['calculated_gap'])
            .mean()
        )
    }
    
    # Generate validation plots
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Gap comparison plot
        axes[0,0].scatter(validation_df['GapToLeader_ms'], validation_df['calculated_gap'], alpha=0.5)
        axes[0,0].plot([0, validation_df['GapToLeader_ms'].max()], 
                      [0, validation_df['GapToLeader_ms'].max()], 'r--')
        axes[0,0].set_xlabel('Original Gap (ms)')
        axes[0,0].set_ylabel('Calculated Gap (ms)')
        axes[0,0].set_title('Gap to Leader: Original vs Calculated')
        
        # Interval comparison plot
        axes[0,1].scatter(validation_df['IntervalToPositionAhead_ms'], 
                         validation_df['calculated_interval'], alpha=0.5)
        axes[0,1].plot([0, validation_df['IntervalToPositionAhead_ms'].max()], 
                      [0, validation_df['IntervalToPositionAhead_ms'].max()], 'r--')
        axes[0,1].set_xlabel('Original Interval (ms)')
        axes[0,1].set_ylabel('Calculated Interval (ms)')
        axes[0,1].set_title('Interval: Original vs Calculated')
        
        # Cumulative time comparison plot
        axes[1,0].scatter(validation_df['cumulative_milliseconds'], 
                         validation_df['calculated_cumtime'], alpha=0.5)
        axes[1,0].plot([0, validation_df['cumulative_milliseconds'].max()], 
                      [0, validation_df['cumulative_milliseconds'].max()], 'r--')
        axes[1,0].set_xlabel('Original Cumulative Time (ms)')
        axes[1,0].set_ylabel('Calculated Cumulative Time (ms)')
        axes[1,0].set_title('Cumulative Time: Original vs Calculated')
        
        # Consistency check plot
        axes[1,1].hist(validation_df['calculated_gap'] - 
                      (validation_df['calculated_cumtime'] - 
                       validation_df.groupby('raceId')['calculated_cumtime'].transform('min')),
                      bins=50)
        axes[1,1].set_xlabel('Difference (ms)')
        axes[1,1].set_ylabel('Count')
        axes[1,1].set_title('Gap vs Cumulative Time Difference Check')
        
        plt.tight_layout()
        plt.savefig('validation_plots.png')
        plt.close()
        
    except ImportError:
        print("Matplotlib not available for plotting")
    
    return results

def analyze_validation_results(validation_results: dict):
    """
    Print analysis of validation results with recommendations.
    """
    print("\nValidation Analysis:")
    
    metrics = {
        'gap': 'Gap to Leader',
        'interval': 'Interval to Position Ahead',
        'cumtime': 'Cumulative Time'
    }
    
    for metric_key, metric_name in metrics.items():
        validation_key = f'{metric_key}_validation'
        if validation_key in validation_results:
            print(f"\n{metric_name} Validation:")
            print(f"- Average difference: {validation_results[validation_key]['mean_difference_ms']:.2f}ms")
            print(f"- Standard deviation: {validation_results[validation_key]['std_difference_ms']:.2f}ms")
            print(f"- Maximum difference: {validation_results[validation_key]['max_difference_ms']:.2f}ms")
            print(f"- % within 100ms: {validation_results[validation_key]['within_100ms']:.1f}%")
            print(f"- % within 1s: {validation_results[validation_key]['within_1000ms']:.1f}%")
            print(f"- Correlation: {validation_results[validation_key]['correlation']:.4f}")
            print(f"- Total comparisons: {validation_results[validation_key]['total_comparisons']}")
    
    if 'consistency_checks' in validation_results:
        print("\nConsistency Checks:")
        print(f"- Average gap vs cumtime difference: {validation_results['consistency_checks']['gap_equals_cumtime_diff']:.2f}ms")
        print(f"- Average intervals sum to gap difference: {validation_results['consistency_checks']['intervals_sum_to_gap']:.2f}ms")
    
    print("\nRecommendations:")
    
    for metric_key, metric_name in metrics.items():
        validation_key = f'{metric_key}_validation'
        if validation_key in validation_results:
            accuracy = validation_results[validation_key]['within_1000ms']
            if accuracy < 95:
                print(f"- {metric_name} calculations may need adjustment - accuracy below 95% within 1 second")
            elif accuracy < 98:
                print(f"- {metric_name} calculations acceptable but could be improved")
            else:
                print(f"- {metric_name} calculations look good (>98% within 1 second)")

def add_tire_info(
    laps: pd.DataFrame,
    tire_data: pd.DataFrame,
    drivers: pd.DataFrame,
    races: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds tire compound and track status information to laps DataFrame, now using TyreLife from tire_data as tire_age.
    """
    # Standardize text data
    tire_data['Compound'] = tire_data['Compound'].str.upper()
    tire_data['EventName_clean'] = tire_data['EventName'].apply(preprocess_string)
    races['name_clean'] = races['name'].apply(preprocess_string)
    drivers['code'] = drivers['code'].str.strip().str.upper()
    tire_data['Driver'] = tire_data['Driver'].str.strip().str.upper()

    # Filter for race sessions
    tire_data = tire_data[tire_data['SessionName'] == 'R']

    # Merge with races to get raceId
    tire_data = tire_data.merge(
        races[['raceId', 'year', 'name_clean']],
        left_on=['Year', 'EventName_clean'],
        right_on=['year', 'name_clean'],
        how='left',
        indicator=True
    )

    # Handle unmatched races (fuzzy matching already in original code)
    unmatched_tire = tire_data[tire_data['_merge'] == 'left_only']
    if not unmatched_tire.empty:
        logging.warning("Unmatched races in tire data:")
        logging.warning(unmatched_tire[['EventName', 'Year']].drop_duplicates())

        unmatched_event_names = unmatched_tire['EventName_clean'].unique()
        races_event_names = races['name_clean'].unique()
        event_name_mapping = {}
        for event in unmatched_event_names:
            match, score = process.extractOne(event, races_event_names)
            if score >= 80:
                event_name_mapping[event] = match
                logging.info(f"Mapping '{event}' to '{match}' with score {score}")
            else:
                logging.warning(f"No suitable match found for '{event}'")

        for event, match in event_name_mapping.items():
            tire_data.loc[tire_data['EventName_clean'] == event, 'EventName_clean'] = match

        tire_data = tire_data.merge(
            races[['raceId', 'year', 'name_clean']],
            left_on=['Year', 'EventName_clean'],
            right_on=['year', 'name_clean'],
            how='left',
            indicator=False
        )

    # Map driver codes to driverId
    driver_code_to_id = drivers.set_index('code')['driverId'].to_dict()
    tire_data['driverId'] = tire_data['Driver'].map(driver_code_to_id)

    # Rename and ensure integer type for 'lap'
    tire_data.rename(columns={'LapNumber': 'lap'}, inplace=True)
    tire_data['lap'] = tire_data['lap'].astype(int, errors='ignore')
    laps['lap'] = laps['lap'].astype(int, errors='ignore')

    # Create compound mapping
    compound_mapping = {
        'UNKNOWN': 0, np.nan: 0,
        'HARD': 1, 'MEDIUM': 2, 'SOFT': 3,
        'SUPERSOFT': 3, 'ULTRASOFT': 3, 'HYPERSOFT': 3,
        'INTERMEDIATE': 4, 'WET': 5
    }

    # Merge tire_data with laps, now including TyreLife
    laps = laps.merge(
        tire_data[['raceId', 'driverId', 'lap', 'Compound', 'TrackStatus', 
                   'CumRaceTime_ms', 'GapToLeader_ms', 'IntervalToPositionAhead_ms', 'TyreLife']], 
        on=['raceId', 'driverId', 'lap'],
        how='left'
    )

    # Handle missing compounds
    laps['Compound'].fillna('UNKNOWN', inplace=True)
    laps['tire_compound'] = laps['Compound'].map(compound_mapping)
    laps.drop('Compound', axis=1, inplace=True)

    # Use TyreLife as tire_age, fill NaN with 0
    laps['tire_age'] = laps['TyreLife'].fillna(0)
    laps.drop('TyreLife', axis=1, inplace=True)

    return laps

def clean_time_intervals(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and harmonizes time interval columns in the laps DataFrame.
    Handles negative values, missing values for leader, and retired drivers.
    
    Assumes columns: 'GapToLeader_ms', 'IntervalToPositionAhead_ms'
    and a 'positionOrder' column that indicates the driver's position.
    """
    # Convert negative values to NaN
    for col in ['GapToLeader_ms', 'IntervalToPositionAhead_ms', 'CumRaceTime_ms']:
        if col in laps.columns:
            laps.loc[laps[col] < 0, col] = np.nan

    # Fill the race leader's gap columns with 0 if NaN
    # Leader identified by positionOrder == 1
    leader_mask = (laps['positionOrder'] == 1)
    if 'GapToLeader_ms' in laps.columns:
        laps.loc[leader_mask & laps['GapToLeader_ms'].isna(), 'GapToLeader_ms'] = 0
    if 'IntervalToPositionAhead_ms' in laps.columns:
        laps.loc[leader_mask & laps['IntervalToPositionAhead_ms'].isna(), 'IntervalToPositionAhead_ms'] = 0

    # Optional: If you want to handle missing intervals for non-leaders:
    # For now, let's just leave them as NaN to signify unknown interval.
    # If you have information about retirement, you can handle it here:
    # Example: If 'status' indicates DNF, we leave these values as NaN.
    # If 'status' column is present and indicates retirement:
    if 'status' in laps.columns:
        # Create a retirement mask if needed (depends on how 'status' is encoded)
        # For demonstration, assume 'status' = 'Retired' means not in race
        retired_mask = (laps['status'] == 'Retired')
        # Retired drivers keep NaN values, so no further action needed here.

    return laps

def add_practice_session_info(
    laps: pd.DataFrame,
    practice_sessions: pd.DataFrame,
    drivers: pd.DataFrame,
    races: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds practice session median lap times to the laps DataFrame.
    
    Returns:
        pd.DataFrame: The laps DataFrame with practice session information.
    """
    # Standardize names
    practice_sessions['EventName_clean'] = practice_sessions['EventName'].apply(preprocess_string)
    races['name_clean'] = races['name'].apply(preprocess_string)
    drivers['code'] = drivers['code'].str.strip().str.upper()
    practice_sessions['Driver'] = practice_sessions['Driver'].str.strip().str.upper()

    # First merge attempt with races
    practice_sessions_merged = practice_sessions.merge(
        races[['raceId', 'year', 'name_clean']],
        left_on=['Year', 'EventName_clean'],
        right_on=['year', 'name_clean'],
        how='left'
    )

    # Identify unmatched races
    unmatched_races = practice_sessions_merged[practice_sessions_merged['raceId'].isna()]
    if not unmatched_races.empty:
        logging.warning("Unmatched races in practice session data:")
        logging.warning(unmatched_races[['EventName', 'Year']].drop_duplicates())

        # Use fuzzy matching for unmatched races
        unmatched_event_names = unmatched_races['EventName_clean'].unique()
        races_event_names = races['name_clean'].unique()
        
        # Create mapping dictionary for fuzzy matches
        event_name_mapping = {}
        for event in unmatched_event_names:
            match, score = process.extractOne(event, races_event_names)
            if score >= 80:  # 80% similarity threshold
                event_name_mapping[event] = match
                logging.info(f"Mapping '{event}' to '{match}' with score {score}")
            else:
                logging.warning(f"No suitable match found for '{event}'")

        # Apply mappings to unmatched events
        for event, match in event_name_mapping.items():
            practice_sessions.loc[
                practice_sessions['EventName_clean'] == event, 
                'EventName_clean'
            ] = match

        # Retry merge with updated event names
        practice_sessions_merged = practice_sessions.merge(
            races[['raceId', 'year', 'name_clean']],
            left_on=['Year', 'EventName_clean'],
            right_on=['year', 'name_clean'],
            how='left'
        )

    # Map driver codes to driverId
    driver_code_to_id = drivers.set_index('code')['driverId'].to_dict()
    practice_sessions_merged['driverId'] = practice_sessions_merged['Driver'].map(driver_code_to_id)

    # Convert LapTime to milliseconds
    practice_sessions_merged['LapTime_ms'] = practice_sessions_merged['LapTime'].apply(
        lambda x: pd.to_timedelta(x, errors='coerce').total_seconds() * 1000
    )

    # Exclude sprint shootout and sprints
    practice_sessions_filtered = practice_sessions_merged[
        ~practice_sessions_merged['SessionName'].isin(['S', 'SS'])
    ]

    # Calculate median lap times per driver per session at each race
    session_medians = practice_sessions_filtered.groupby(
        ['raceId', 'driverId', 'SessionName']
    )['LapTime_ms'].median().reset_index()

    # Pivot sessions into columns
    session_medians_pivot = session_medians.pivot_table(
        index=['raceId', 'driverId'],
        columns='SessionName',
        values='LapTime_ms'
    ).reset_index()

    # Rename columns
    session_medians_pivot.rename(columns={
        'FP1': 'fp1_median_time',
        'FP2': 'fp2_median_time',
        'FP3': 'fp3_median_time',
        'Q': 'quali_time'
    }, inplace=True)

    # Merge into laps
    laps = laps.merge(
        session_medians_pivot,
        on=['raceId', 'driverId'],
        how='left'
    )

    # Ensure 'constructorId' is available in 'laps'
    if 'constructorId' not in laps.columns:
        driver_constructor = laps[['raceId', 'driverId', 'constructorId']].drop_duplicates()
        laps = laps.merge(driver_constructor, on=['raceId', 'driverId'], how='left')

    sessions = ['fp1_median_time', 'fp2_median_time', 'fp3_median_time', 'quali_time']

    # Fill missing values with teammate's median time
    for session in sessions:
        missing_mask = laps[session].isna()
        if missing_mask.any():
            teammate_times = laps[~laps[session].isna()][['raceId', 'constructorId', session]]
            teammate_times = teammate_times.groupby(['raceId', 'constructorId'])[session].median().reset_index()
            teammate_times.rename(columns={session: f"{session}_teammate"}, inplace=True)
            
            laps = laps.merge(
                teammate_times,
                on=['raceId', 'constructorId'],
                how='left'
            )
            laps.loc[missing_mask, session] = laps.loc[missing_mask, f"{session}_teammate"]
            laps.drop(columns=[f"{session}_teammate"], inplace=True)

    # Fill remaining missing values with race session median
    for session in sessions:
        missing_mask = laps[session].isna()
        if missing_mask.any():
            laps[session] = laps.groupby('raceId')[session].transform(
                lambda x: x.fillna(x.median())
            )

    # Fill with driver's median time across all races
    for session in sessions:
        missing_mask = laps[session].isna()
        if missing_mask.any():
            driver_session_medians = laps.groupby('driverId')[session].transform('median')
            laps.loc[missing_mask, session] = driver_session_medians[missing_mask]

    # Fill any remaining missing values with global median
    for session in sessions:
        global_median = laps[session].median()
        laps[session].fillna(global_median, inplace=True)

    return laps

def remove_duplicate_columns(df: pd.DataFrame, exceptions: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove duplicate columns (based on values) while keeping the first occurrence.
    Also track and return the names of removed columns, excluding those in the exceptions list.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        exceptions (List[str], optional): Columns to exclude from removal, even if duplicates.
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: Cleaned DataFrame and list of removed columns.
    """
    # Handle the case where exceptions are not provided
    exceptions = exceptions or []
    
    # Transpose the DataFrame to make columns into rows for comparison
    duplicates = df.T.duplicated()
    
    # Find duplicate columns, excluding those in the exceptions list
    columns_to_remove = [col for col, is_duplicate in zip(df.columns, duplicates) 
                         if is_duplicate and col not in exceptions]
    
    # Create a cleaned DataFrame excluding the duplicate columns
    df_cleaned = df.drop(columns=columns_to_remove)
    
    # Print the removed columns
    print("\nRemoved Columns:")
    print(columns_to_remove)
    
    return df_cleaned

def get_first_lap_positions(lap_times: pd.DataFrame) -> pd.DataFrame:
    # Filter for the first lap
    first_lap = lap_times[lap_times['lap'] == 1]
    
    # Select necessary columns
    first_lap_positions = first_lap[['raceId', 'driverId', 'position']].copy()
    
    # Rename the 'position' column to 'position_lap_1'
    first_lap_positions.rename(columns={'position': 'position_lap_1'}, inplace=True)
    
    return first_lap_positions

def determine_wet_races(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Determines wet races based on precipitation and tire compounds.

    Returns:
        pd.DataFrame: DataFrame with 'raceId' and 'wet_race' columns.
    """
    # Use tire compounds
    wet_races_from_tires = laps.groupby('raceId')['tire_compound'].apply(
        lambda x: x.isin([4, 5]).any()
    ).reset_index()
    wet_races_from_tires.rename(columns={'tire_compound': 'wet_race_tires'}, inplace=True)

    # Use precipitation data
    if 'prcp' in laps.columns:
        wet_races_from_weather = laps.groupby('raceId')['prcp'].mean().reset_index()
        wet_races_from_weather['wet_race_weather'] = wet_races_from_weather['prcp'] > 0.0

        # Merge both
        wet_races = wet_races_from_tires.merge(
            wet_races_from_weather[['raceId', 'wet_race_weather']],
            on='raceId',
            how='outer'
        )
        # Final wet_race determination
        wet_races['wet_race'] = wet_races[['wet_race_tires', 'wet_race_weather']].any(axis=1)
    else:
        wet_races = wet_races_from_tires.rename(columns={'wet_race_tires': 'wet_race'})

    wet_races['wet_race'].fillna(False, inplace=True)
    return wet_races[['raceId', 'wet_race']]




def calculate_driver_metrics(driver_id: int, results: pd.DataFrame, lap_times: pd.DataFrame, races: pd.DataFrame, window_races: int = 10) -> dict:
    """
    Calculate enhanced driver metrics based on historical performance.
    Uses a rolling window of races to ensure metrics are time-relevant.
    """
    # Get driver's recent results
    driver_results = results[results['driverId'] == driver_id].copy()
    
    # Merge with races DataFrame
    driver_results = driver_results.merge(
        races[['raceId', 'date', 'year', 'round', 'circuitId']],
        on='raceId',
        how='left',
        suffixes=('', '_race')
    )
    
    # Rename columns for clarity
    driver_results.rename(columns={
        'date': 'race_date',
        'year': 'race_year',
        'round': 'race_round',
        'circuitId': 'circuitId'
    }, inplace=True)
    
    # Get first lap positions
    first_lap_positions = get_first_lap_positions(lap_times)
    
    # Merge first lap positions into driver_results
    driver_results = driver_results.merge(
        first_lap_positions,
        on=['raceId', 'driverId'],
        how='left'
    )
    
    # Fill missing 'position_lap_1' with grid position
    driver_results['position_lap_1'].fillna(driver_results['grid'], inplace=True)
    
    # Now 'wet_race' is available in driver_results
    # Proceed with calculations
    metrics = {}

    # Calculate Skill (0-1)
    def calculate_skill(race_window: pd.DataFrame) -> float:
        """
        Calculate driver skill with proper handling of edge cases.
        """
        if len(race_window) <= 1:  # Check for insufficient data
            return 0.5  # Return default value for insufficient data
            
        # Weight recent races more heavily
        weights = np.linspace(0.5, 1.0, len(race_window))
        
        try:
            # Safe calculations with error handling
            finishing_positions = np.array(race_window['positionOrder'].values, dtype=float)
            starting_positions = np.array(race_window['grid'].values, dtype=float)
            points_scored = np.array(race_window['points'].values, dtype=float)
            
            # Handle zeros and invalid values
            finishing_positions = np.clip(finishing_positions, 1, 20)
            starting_positions = np.clip(starting_positions, 1, 20)
            
            # Normalize positions (reverse scale - lower is better)
            norm_finish = 1 - (finishing_positions / 20)
            norm_start = 1 - (starting_positions / 20)
            
            # Safe position improvements calculation
            pos_improvements = starting_positions - finishing_positions
            norm_improvements = np.clip((pos_improvements + 20) / 40, 0, 1)
            
            # Safe points calculation
            norm_points = np.clip(points_scored / 25, 0, 1)
            
            # Weighted components with safe averaging
            skill_score = (
                0.35 * np.average(norm_finish, weights=weights) +
                0.25 * np.average(norm_start, weights=weights) +
                0.25 * np.average(norm_points, weights=weights) +
                0.15 * np.average(norm_improvements, weights=weights)
            )
            
            return np.clip(skill_score, 0.1, 1.0)
            
        except Exception as e:
            logging.warning(f"Error calculating skill: {str(e)}")
            return 0.5
    
    # Calculate Consistency (0-1)
    def calculate_consistency(race_window: pd.DataFrame, driver_laps: pd.DataFrame) -> float:
        if race_window.empty:
            return 0.5
            
        consistency_scores = []
        
        for _, race in race_window.iterrows():
            race_laps = driver_laps[driver_laps['raceId'] == race['raceId']]
            
            if len(race_laps) < 5:  # Skip races with too few laps
                continue
                
            # Calculate lap time consistency excluding outliers
            lap_times = race_laps['milliseconds'].values
            q1, q3 = np.percentile(lap_times, [25, 75])
            iqr = q3 - q1
            valid_laps = lap_times[(lap_times >= q1 - 1.5*iqr) & (lap_times <= q3 + 1.5*iqr)]
            
            if len(valid_laps) < 3:
                continue
                
            # Use coefficient of variation as consistency measure
            cv = np.std(valid_laps) / np.mean(valid_laps)
            consistency = 1 - np.clip(cv, 0, 0.1) / 0.1  # Scale CV to 0-1
            consistency_scores.append(consistency)
        
        if not consistency_scores:
            return 0.5
            
        return np.mean(consistency_scores)
    
    # Calculate Aggression (0-1)
    def calculate_aggression(race_window: pd.DataFrame) -> float:
        try:
            if race_window.empty:
                return 0.5
                
            # Track aggressive moves
            dnf_statuses = [4, 5, 6, 20, 82]  # Status IDs for accidents/collisions
            incident_rate = (race_window['statusId'].isin(dnf_statuses)).mean()
            
            # Overtaking aggression
            position_changes = race_window['grid'] - race_window['positionOrder']
            overtaking_rate = (position_changes > 0).mean()
            avg_positions_gained = position_changes[position_changes > 0].mean() or 0
            
            # First lap aggression
            first_lap_gains = race_window['grid'] - race_window['position_lap_1']
            first_lap_aggression = (first_lap_gains > 0).mean()
            
            # Combine metrics
            aggression_score = (
                0.3 * np.clip(incident_rate * 2, 0, 1) +  # Incident rate (weighted higher)
                0.3 * np.clip(overtaking_rate, 0, 1) +    # Overtaking frequency
                0.2 * np.clip(avg_positions_gained / 5, 0, 1) +  # Average positions gained
                0.2 * np.clip(first_lap_aggression * 1.5, 0, 1)  # First lap aggression
            )
            
            return np.clip(aggression_score, 0.1, 1.0)
        except Exception as e:
            logging.warning(f"Error calculating skill: {str(e)}")
            return 0.5

    
    # Calculate Adaptability (0-1) - New metric
    def calculate_adaptability(race_window: pd.DataFrame) -> float:
        if race_window.empty:
            return 0.5
            
        # Compare qualifying to race performance in different conditions
        qual_race_correlation = np.corrcoef(
            race_window['grid'].values,
            race_window['positionOrder'].values
        )[0, 1]
        
        # Performance variation across different tracks
        track_performance_std = race_window.groupby('circuitId')['positionOrder'].mean().std()
        norm_track_std = 1 - np.clip(track_performance_std / 5, 0, 1)
        
        # Wet race performance vs dry
        wet_races = race_window[race_window['wet_race'] == True]
        dry_races = race_window[race_window['wet_race'] == False]
        
        if not wet_races.empty and not dry_races.empty:
            wet_performance = wet_races['positionOrder'].mean()
            dry_performance = dry_races['positionOrder'].mean()
            weather_adaptability = 1 - abs(wet_performance - dry_performance) / 20
        else:
            weather_adaptability = 0.5
        
        adaptability_score = (
            0.4 * (1 - abs(qual_race_correlation)) +  # Lower correlation = more adaptable
            0.3 * norm_track_std +                    # Consistent across different tracks
            0.3 * weather_adaptability                # Consistent in different conditions
        )
        
        return np.clip(adaptability_score, 0.1, 1.0)
    
    # Calculate Reliability (0-1)
    def calculate_reliability(race_window: pd.DataFrame) -> float:
        if race_window.empty:
            return 0.5
            
        # Calculate finish rate
        finish_rate = (race_window['statusId'] == 'Finished').mean()
        
        # Calculate mechanical failure rate (excluding accidents)
        mechanical_dnfs = race_window['statusId'].isin([3, 4, 5, 6, 7, 8, 9, 10, 11])
        mechanical_reliability = 1 - mechanical_dnfs.mean()
        
        # Points finishing consistency
        points_consistency = (race_window['points'] > 0).mean()
        
        reliability_score = (
            0.4 * finish_rate +
            0.4 * mechanical_reliability +
            0.2 * points_consistency
        )
        
        return np.clip(reliability_score, 0.1, 1.0)
    
    # Calculate metrics using rolling window
    for idx in range(len(driver_results)):
        race_date = driver_results.iloc[idx]['race_date']
        
        # Get window of previous races
        historical_races = driver_results[
            driver_results['race_date'] < race_date
        ].tail(window_races)
        
        # Get relevant lap times
        historical_race_ids = historical_races['raceId'].values
        historical_laps = lap_times[
            (lap_times['driverId'] == driver_id) &
            (lap_times['raceId'].isin(historical_race_ids))
        ]
        
        # Calculate metrics for this race
        race_id = driver_results.iloc[idx]['raceId']
        if race_id not in metrics:
            metrics[race_id] = {
                'driver_overall_skill': calculate_skill(historical_races),
                'driver_consistency': calculate_consistency(historical_races, historical_laps),
                'driver_aggression': calculate_aggression(historical_races),
                'driver_adaptability': calculate_adaptability(historical_races),
                'driver_reliability': calculate_reliability(historical_races)
            }
            
            # Calculate circuit-specific skill
            circuit_id = driver_results.iloc[idx]['circuitId']
            circuit_history = historical_races[historical_races['circuitId'] == circuit_id]
            if not circuit_history.empty:
                metrics[race_id]['driver_circuit_skill'] = calculate_skill(circuit_history)
            else:
                metrics[race_id]['driver_circuit_skill'] = metrics[race_id]['driver_overall_skill']
            
            # Calculate risk-taking as a function of aggression and adaptability
            metrics[race_id]['driver_risk_taking'] = (
                0.7 * metrics[race_id]['driver_aggression'] +
                0.3 * (1 - metrics[race_id]['driver_adaptability'])
            )
    
    return metrics

def enhance_driver_attributes(laps: pd.DataFrame, results: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """
    Enhance driver attributes in the laps DataFrame using the new metrics calculation.
    """
    print("Calculating enhanced driver metrics...")

    # **Determine wet races using the updated weather data**
    wet_races_df = determine_wet_races(laps)

    # Merge wet_race information into results DataFrame
    results = results.merge(wet_races_df, on='raceId', how='left')
    results['wet_race'].fillna(False, inplace=True)

    # Calculate metrics for each driver
    all_metrics = {}
    unique_drivers = laps['driverId'].unique()

    for driver_id in unique_drivers:
        driver_metrics = calculate_driver_metrics(
            driver_id,
            results,
            laps,
            races
        )
        all_metrics[driver_id] = driver_metrics
    
    # Apply metrics to laps DataFrame
    metric_columns = [
        'driver_overall_skill',
        'driver_circuit_skill',
        'driver_consistency',
        'driver_aggression',
        'driver_adaptability',
        'driver_reliability',
        'driver_risk_taking'
    ]
    
    # Create new columns for enhanced metrics
    for column in metric_columns:
        laps[f'{column}_enhanced'] = np.nan
    
    # Fill in the enhanced metrics
    for driver_id in unique_drivers:
        driver_metrics = all_metrics[driver_id]
        for race_id in driver_metrics:
            mask = (laps['driverId'] == driver_id) & (laps['raceId'] == race_id)
            for column in metric_columns:
                laps.loc[mask, f'{column}_enhanced'] = driver_metrics[race_id][column]
    
    # Replace original columns with enhanced versions
    for column in metric_columns:
        laps[column] = laps[f'{column}_enhanced']
        laps.drop(f'{column}_enhanced', axis=1, inplace=True)
    
    return laps

def add_dynamic_features(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Adds dynamic features like fuel_load, track_position, and is_pit_lap.
    The tire_age calculation is now handled by TyreLife from tire_data, so we remove the old calculation.
    """
    # Remove the old tire_age calculation line
    # laps['tire_age'] = laps.groupby(['raceId', 'driverId'])['lap'].cumcount()

    laps['fuel_load'] = laps.groupby(['raceId', 'driverId'])['lap'].transform(lambda x: x.max() - x + 1)
    #laps['track_position'] = laps['positionOrder']
    laps['is_pit_lap'] = laps['pitstop_milliseconds'].apply(lambda x: 1 if x > 0 else 0)
    laps['TrackStatus'].fillna(1, inplace=True)  # 1 = regular racing status
    return laps

def remove_lap_time_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Only removes clearly invalid lap times (>150 seconds).
    Keeps all other racing conditions and variations.
    """
    df = df.copy()
    filtered_df = df[df['milliseconds'] < 150000]
    return filtered_df

# def remove_lap_time_outliers(
#     df: pd.DataFrame,
#     iqr_multiplier: float = 1.5
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Removes lap time outliers using the IQR method.
    
#     Returns:
#         Tuple[pd.DataFrame, pd.DataFrame]: Cleaned laps DataFrame and special laps DataFrame.
#     """
#     df = df.copy()
#     normal_racing_mask = (
#         (df['TrackStatus'] == 1) &
#         (df['is_pit_lap'] == 0) &
#         (df['milliseconds'] < 150000)
#     )
#     special_laps = df[~normal_racing_mask]
#     normal_laps = df[normal_racing_mask]

#     def remove_outliers_group(group):
#         Q1 = group['milliseconds'].quantile(0.25)
#         Q3 = group['milliseconds'].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - iqr_multiplier * IQR
#         upper_bound = Q3 + iqr_multiplier * IQR
#         return group[(group['milliseconds'] >= lower_bound) & (group['milliseconds'] <= upper_bound)]

#     cleaned_normal_laps = normal_laps.groupby('circuitId').apply(remove_outliers_group).reset_index(drop=True)
#     return cleaned_normal_laps, special_laps

def drop_unnecessary_columns(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Drops unnecessary columns from the laps DataFrame.
    
    Returns:
        pd.DataFrame: The laps DataFrame with unnecessary columns dropped.
    """
    columns_to_drop = ['EventFormat', 'EventName', 'S', 'SessionName', 'R', 'Time', 'Year', 'WindDirection', 
        'WindSpeed', 'WindSpeed', 'circuitRef', 'constructor_name', 
        'date', 'date_race', 'dob', 'driverRef', 'fastestLap', 'forename', 'fp1_date', 
        'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date', 'fp3_time', 'location', 
        'name', 'name_x', 'name_y', 'number', 'quali_date', 'CumRaceTime_ms'
        'quali_date_time', 'rainfall', 'raceId_x', 'raceId_y', 'raceTime', 'positionOrder', 'cumulative_milliseconds'
        'RoundNumber', 'sprint_date', 'sprint_time', 'surname', 'time', 'time_race', 
        'url_race', 'url_x', 'url_y', 'year_x', 'year_y', 'statusId', 'constructor_points', 'constructor_points']
    laps.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')
    return laps

def handle_missing_values(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in the laps DataFrame.
    
    Returns:
        pd.DataFrame: The laps DataFrame with missing values handled.
    """
    # Print initial state
    print(f"\nInitial shape: {laps.shape}")
    print(f"Initial unique races: {laps['raceId'].nunique()}")
    print(f"Initial unique drivers: {laps['driverId'].nunique()}")
    
    # Get required columns
    race_features = F1RaceFeatures()
    required_columns = race_features.static_features + race_features.dynamic_features
    
    # Check missing values before any operations
    missing_counts = laps[required_columns].isnull().sum()
    missing_columns = missing_counts[missing_counts > 0]
    
    if not missing_columns.empty:
        print("\nColumns with missing values:")
        for col in missing_columns.index:
            n_missing = missing_counts[col]
            pct_missing = (n_missing / len(laps)) * 100
            missing_races = laps[laps[col].isnull()]['raceId'].nunique()
            missing_drivers = laps[laps[col].isnull()]['driverId'].nunique()
            
            print(f"\nColumn: {col}")
            print(f"- Missing values: {n_missing} ({pct_missing:.2f}%)")
            print(f"- Affects {missing_races} races and {missing_drivers} drivers")
            print("- Example races affected:", laps[laps[col].isnull()]['raceId'].unique()[:5].tolist())
            print("- Example drivers affected:", laps[laps[col].isnull()]['driverId'].unique()[:5].tolist())
    
    # Track which races lose drivers
    before_counts = laps.groupby('raceId')['driverId'].nunique()
    
    # Drop rows with missing values
    laps.dropna(subset=required_columns, inplace=True)
    
    # After dropping, check which races lost drivers
    after_counts = laps.groupby('raceId')['driverId'].nunique()
    driver_losses = before_counts - after_counts
    
    problematic_races = driver_losses[driver_losses > 5]
    if not problematic_races.empty:
        print("\nRaces losing more than 5 drivers:")
        for race_id, loss in problematic_races.items():
            print(f"\nRace {race_id}:")
            print(f"- Started with {before_counts[race_id]} drivers")
            print(f"- Ended with {after_counts[race_id]} drivers")
            print(f"- Lost {loss} drivers")
            
            # Show which drivers were lost
            before_drivers = set(laps[laps['raceId'] == race_id]['driverId'].unique())
            after_drivers = set(laps[laps['raceId'] == race_id]['driverId'].unique())
            lost_drivers = before_drivers - after_drivers
            print(f"- Lost drivers: {list(lost_drivers)}")
    
    print(f"\nFinal shape: {laps.shape}")
    print(f"Final unique races: {laps['raceId'].nunique()}")
    print(f"Final unique drivers: {laps['driverId'].nunique()}")
    
    return laps

def save_auxiliary_data(
    laps: pd.DataFrame,
    drivers: pd.DataFrame,
    races: pd.DataFrame,
) -> pd.DataFrame:
    """
    Saves auxiliary data such as driver attributes, circuit attributes,
    special laps, and weather data.
    
    Returns:
        pd.DataFrame: The laps DataFrame.
    """
    # Create driver-specific attributes
    drivers_df = laps.groupby(['driverId', 'raceId']).agg({
        'driver_overall_skill': 'last',
        'driver_circuit_skill': 'last',
        'driver_consistency': 'last',
        'driver_reliability': 'last',
        'driver_aggression': 'last',
        'driver_risk_taking': 'last',
        'constructor_performance': 'last',
        'fp1_median_time': 'last',
        'fp2_median_time': 'last',
        'fp3_median_time': 'last',
        'quali_time': 'last',
        'constructorId': 'last',
        'code': 'last'
    }).reset_index()
    drivers_df.to_csv('../data/util/drivers_attributes.csv', index=False)

    # Create circuit-specific attributes
    circuit_attributes_df = laps.groupby('circuitId').agg({
        'circuit_length': 'first',
        'circuit_type_encoded': 'first',
        'lat': 'first',
        'lng': 'first',
        'alt': 'first'
    }).reset_index()
    circuit_attributes_df.to_csv('../data/util/circuit_attributes.csv', index=False)

    # Save weather data by raceId and cumulated lap time
    weather_df = laps[['raceId', 'cumulative_milliseconds', 'Humidity', 'TrackTemp', 'AirTemp']].copy()
    weather_df = weather_df.drop_duplicates()
    weather_df.sort_values(['raceId', 'cumulative_milliseconds'], inplace=True)
    weather_df.to_csv('../data/util/weather_data.csv', index=False)

    # Save special laps
    # special_laps.to_csv('../data/SPECIAL_LAPS.csv', index=False)

    # Save processed laps
    laps.to_csv('../data/LAPS.csv', index=False)
    
    return laps


def load_and_preprocess_data():
    # Ensure RaceFeatures is available
    race_features = F1RaceFeatures()

    # Preprocess data
    df = preprocess_data()

    # Validate that all required columns are present
    required_columns = race_features.static_features + race_features.dynamic_features
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Return the processed DataFrame
    return df