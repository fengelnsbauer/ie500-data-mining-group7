# data_preparation.py

import pandas as pd
import numpy as np
import re
from typing import Tuple
from sklearn.model_selection import GroupShuffleSplit
from thefuzz import process  # For fuzzy matching
import logging

from features import RaceFeatures

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
    try:
        lap_times = pd.read_csv('../data/raw_data/lap_times.csv', na_values=na_values)
        drivers = pd.read_csv('../data/raw_data/drivers.csv', na_values=na_values)
        races = pd.read_csv('../data/raw_data/races.csv', na_values=na_values)
        circuits = pd.read_csv('../data/raw_data/circuits.csv', na_values=na_values)
        pit_stops = pd.read_csv('../data/raw_data/pit_stops.csv', na_values=na_values)
        pit_stops.rename(columns={'milliseconds': 'pitstop_milliseconds'}, inplace=True)
        results = pd.read_csv('../data/raw_data/results.csv', na_values=na_values)
        results.rename(columns={'milliseconds': 'racetime_milliseconds'}, inplace=True)
        print(results.columns)
        qualifying = pd.read_csv('../data/raw_data/qualifying.csv', na_values=na_values)
        status = pd.read_csv('../data/raw_data/status.csv', na_values=na_values)
        weather_data = pd.read_csv('../data/raw_data/ff1_weather.csv', na_values=na_values)
        practice_sessions = pd.read_csv('../data/raw_data/ff1_laps.csv', na_values=na_values)
        tire_data = pd.read_csv('../data/raw_data/ff1_laps.csv', na_values=na_values)
        constructors = pd.read_csv('../data/raw_data/constructors.csv', na_values=na_values)
        constructor_standings = pd.read_csv('../data/raw_data/constructor_standings.csv', na_values=na_values)
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
    """
    Preprocesses raw F1 racing data to prepare it for model training.
    
    Returns:
        pd.DataFrame: The preprocessed DataFrame containing lap data with merged features.
    """
    # Load raw data
    data = load_raw_data()

    # Extract individual DataFrames
    lap_times = data['lap_times']
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

    # Preprocess data
    # Preprocess data
    laps = merge_dataframes(lap_times, drivers, races, circuits, results, status, pit_stops)
    laps = filter_laps_by_year(laps, start_year=2018)
    laps = add_constructor_info(laps, constructors, constructor_standings, results)
    laps = add_circuit_info(laps, circuits)

    # Compute cumulative time from the start of the race for each driver
    laps.sort_values(['raceId', 'driverId', 'lap'], inplace=True)
    laps['cumulative_milliseconds'] = laps.groupby(['raceId', 'driverId'])['milliseconds'].cumsum()
    laps['seconds_from_start'] = laps['cumulative_milliseconds'] / 1000.0
    laps = add_weather_info(laps, races, weather_data)
    laps = add_tire_info(laps, tire_data, drivers, races)
    laps = add_practice_session_info(laps, practice_sessions, drivers, races)
    laps = add_driver_metrics(laps, drivers, results, lap_times, status, races)
    laps = add_dynamic_features(laps)
    laps, special_laps = remove_lap_time_outliers(laps)
    laps = drop_unnecessary_columns(laps)
    laps = handle_missing_values(laps)
    laps = save_auxiliary_data(laps, drivers, races, special_laps)

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
    Adds circuit characteristics to the laps DataFrame.
    
    Returns:
        pd.DataFrame: The laps DataFrame with circuit information.
    """
    # Assume circuit_length and circuit_type are available
    circuits['circuit_length'] = 5.0  # Placeholder, replace with actual data
    circuits['circuit_type'] = 'Permanent'  # Options: 'Permanent', 'Street', 'Hybrid'
    laps = laps.merge(
        circuits[['circuitId', 'circuit_length', 'circuit_type']],
        on='circuitId',
        how='left'
    )
    # Encode circuit_type as categorical
    circuit_type_mapping = {'Permanent': 0, 'Street': 1, 'Hybrid': 2}
    laps['circuit_type_encoded'] = laps['circuit_type'].map(circuit_type_mapping)
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
    
    Returns:
        pd.DataFrame: The laps DataFrame with matched weather data.
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
            # Perform asof merge
            merged = pd.merge_asof(
                race_laps.sort_values('seconds_from_start'),
                race_weather.sort_values('seconds_from_start'),
                on='seconds_from_start',
                direction='nearest'
            )
            merged_laps_list.append(merged)
        else:
            # Assign default weather values
            race_laps['TrackTemp'] = 25.0
            race_laps['AirTemp'] = 20.0
            race_laps['Humidity'] = 50.0
            merged_laps_list.append(race_laps)

    # Concatenate all race laps
    laps_with_weather = pd.concat(merged_laps_list, ignore_index=True)
    # Fill any remaining missing weather data
    laps_with_weather['TrackTemp'].fillna(25.0, inplace=True)
    laps_with_weather['AirTemp'].fillna(20.0, inplace=True)
    laps_with_weather['Humidity'].fillna(50.0, inplace=True)
    return laps_with_weather

def add_tire_info(
    laps: pd.DataFrame,
    tire_data: pd.DataFrame,
    drivers: pd.DataFrame,
    races: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds tire compound and track status information to laps DataFrame.
    
    Returns:
        pd.DataFrame: The laps DataFrame with tire information.
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

    # Handle unmatched races
    unmatched_tire = tire_data[tire_data['_merge'] == 'left_only']
    if not unmatched_tire.empty:
        logging.warning("Unmatched races in tire data:")
        logging.warning(unmatched_tire[['EventName', 'Year']].drop_duplicates())

        # Use fuzzy matching to find possible matches
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

        # Apply the mappings
        for event, match in event_name_mapping.items():
            tire_data.loc[tire_data['EventName_clean'] == event, 'EventName_clean'] = match

        # Retry the merge after applying mappings
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
    # Rename and ensure integer type
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
    # Merge tire_data with laps
    laps = laps.merge(
        tire_data[['raceId', 'driverId', 'lap', 'Compound', 'TrackStatus']],
        on=['raceId', 'driverId', 'lap'],
        how='left'
    )
    # Handle missing compounds
    laps['Compound'].fillna('UNKNOWN', inplace=True)
    laps['tire_compound'] = laps['Compound'].map(compound_mapping)
    laps.drop('Compound', axis=1, inplace=True)
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

    # Merge with races to get raceId
    practice_sessions = practice_sessions.merge(
        races[['raceId', 'year', 'name_clean']],
        left_on=['Year', 'EventName_clean'],
        right_on=['year', 'name_clean'],
        how='left',
        indicator=True
    )

    # Handle unmatched races
    unmatched_practice = practice_sessions[practice_sessions['_merge'] == 'left_only']
    if not unmatched_practice.empty:
        logging.warning("Unmatched races in practice session data:")
        logging.warning(unmatched_practice[['EventName', 'Year']].drop_duplicates())

        # Use fuzzy matching to find possible matches
        unmatched_event_names = unmatched_practice['EventName_clean'].unique()
        races_event_names = races['name_clean'].unique()
        event_name_mapping = {}
        for event in unmatched_event_names:
            match, score = process.extractOne(event, races_event_names)
            if score >= 80:
                event_name_mapping[event] = match
                logging.info(f"Mapping '{event}' to '{match}' with score {score}")
            else:
                logging.warning(f"No suitable match found for '{event}'")

        # Apply the mappings
        for event, match in event_name_mapping.items():
            practice_sessions.loc[practice_sessions['EventName_clean'] == event, 'EventName_clean'] = match

        # Retry the merge after applying mappings
        practice_sessions = practice_sessions.merge(
            races[['raceId', 'year', 'name_clean']],
            left_on=['Year', 'EventName_clean'],
            right_on=['year', 'name_clean'],
            how='left',
            indicator=False
        )

    # Map driver codes to driverId
    driver_code_to_id = drivers.set_index('code')['driverId'].to_dict()
    practice_sessions['driverId'] = practice_sessions['Driver'].map(driver_code_to_id)
    # Convert LapTime to milliseconds
    practice_sessions['LapTime_ms'] = practice_sessions['LapTime'].apply(
        lambda x: pd.to_timedelta(x, errors='coerce').total_seconds() * 1000
    )
    # Calculate median lap times per driver per session
    session_medians = practice_sessions.groupby(
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
    laps = laps.merge(session_medians_pivot, on=['raceId', 'driverId'], how='left')
    # Fill missing session times with global medians
    for session in ['fp1_median_time', 'fp2_median_time', 'fp3_median_time', 'quali_time']:
        global_median = laps[session].median()
        laps[session].fillna(global_median, inplace=True)
    return laps

def add_driver_metrics(
    laps: pd.DataFrame,
    drivers: pd.DataFrame,
    results: pd.DataFrame,
    lap_times: pd.DataFrame,
    status: pd.DataFrame,
    races: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculates and adds driver aggression, skill, consistency, and reliability metrics.
    
    Returns:
        pd.DataFrame: The laps DataFrame with driver metrics.
    """
    # Merge necessary data
    drivers['driver_name'] = drivers['forename'] + ' ' + drivers['surname']
    results = results.merge(
        races[['raceId', 'circuitId']],
        on='raceId',
        how='left'
    )
    status_dict = status.set_index('statusId')['status'].to_dict()
    results['status'] = results['statusId'].map(status_dict)
    lap_times_df = lap_times.copy()

    # Calculate driver metrics
    driver_aggression = {}
    driver_skill = {}
    driver_consistency = {}
    driver_reliability = {}

    for driver_id in drivers['driverId'].unique():
        driver_results = results[results['driverId'] == driver_id]
        aggression = calculate_aggression(driver_results)
        driver_aggression[driver_id] = aggression

        skill = calculate_skill(driver_results)
        driver_skill[driver_id] = skill

        consistency = calculate_consistency(lap_times_df, driver_id)
        driver_consistency[driver_id] = consistency

        reliability = calculate_reliability(driver_results)
        driver_reliability[driver_id] = reliability

    # Map metrics back to laps DataFrame
    laps['driver_aggression'] = laps['driverId'].map(driver_aggression)
    laps['driver_overall_skill'] = laps['driverId'].map(driver_skill)
    laps['driver_circuit_skill'] = laps['driver_overall_skill']  # For simplicity
    laps['driver_consistency'] = laps['driverId'].map(driver_consistency)
    laps['driver_reliability'] = laps['driverId'].map(driver_reliability)
    laps['driver_risk_taking'] = laps['driver_aggression']
    return laps

def calculate_aggression(driver_results: pd.DataFrame) -> float:
    """
    Calculates driver aggression based on overtaking and risk metrics.
    
    Returns:
        float: Aggression score between 0 and 1.
    """
    if driver_results.empty:
        return 0.5  # Default aggression

    recent_results = driver_results.sort_values('date', ascending=False).head(20)
    positions_gained = recent_results['grid'] - recent_results['positionOrder']
    dnf_rate = (recent_results['status'] != 'Finished').mean()
    incidents = (recent_results['statusId'].isin([4, 5, 6, 20, 82])).mean()
    positive_overtakes = (positions_gained > 0).sum()
    negative_overtakes = (positions_gained < 0).sum()
    total_overtakes = positive_overtakes + negative_overtakes
    overtake_success_rate = positive_overtakes / total_overtakes if total_overtakes > 0 else 0.5
    avg_positions_gained = positions_gained[positions_gained > 0].mean() or 0
    normalized_gains = np.clip(avg_positions_gained / 20, 0, 1)
    normalized_dnf = np.clip(dnf_rate, 0, 1)
    normalized_incidents = np.clip(incidents, 0, 1)
    overtaking_component = (normalized_gains * 0.6 + overtake_success_rate * 0.4)
    risk_component = (normalized_dnf * 0.5 + normalized_incidents * 0.5)
    aggression = (
        overtaking_component * 0.4 +
        risk_component * 0.5 +
        0.5 * 0.1  # Baseline aggression
    )
    # Add small random variation
    aggression = np.clip(aggression + np.random.normal(0, 0.02), 0, 1)
    return aggression

def calculate_skill(driver_results: pd.DataFrame) -> float:
    """
    Calculates driver skill based on performance metrics.
    
    Returns:
        float: Skill score between 0 and 1.
    """
    recent_results = driver_results.sort_values('date', ascending=False).head(10)
    if recent_results.empty:
        return 0.5  # Default skill

    avg_finish_pos = recent_results['positionOrder'].mean()
    avg_quali_pos = recent_results['grid'].mean()
    points_per_race = recent_results['points'].mean()
    fastest_laps = (recent_results['rank'] == 1).mean()
    normalized_finish_pos = np.exp(-avg_finish_pos / 5)
    normalized_quali_pos = np.exp(-avg_quali_pos / 5)
    max_points_per_race = 26  # Max possible points
    normalized_points = points_per_race / max_points_per_race
    skill = (
        0.35 * normalized_finish_pos +
        0.25 * normalized_quali_pos +
        0.25 * normalized_points +
        0.15 * fastest_laps
    )
    skill = np.clip(skill + np.random.normal(0, 0.05), 0.1, 1.0)
    return skill

def calculate_consistency(lap_times_df: pd.DataFrame, driver_id: int) -> float:
    """
    Calculates driver consistency based on lap time variability.
    
    Returns:
        float: Consistency score between 0 and 1.
    """
    driver_lap_times = lap_times_df[lap_times_df['driverId'] == driver_id]
    recent_races = sorted(driver_lap_times['raceId'].unique(), reverse=True)[:5]
    recent_lap_times = driver_lap_times[driver_lap_times['raceId'].isin(recent_races)]
    if recent_lap_times.empty or len(recent_lap_times) < 5:
        return 0.5  # Default consistency

    lap_time_variance = recent_lap_times['milliseconds'].var()
    max_variance = lap_times_df['milliseconds'].var()
    consistency_score = 1 - (lap_time_variance / max_variance)
    consistency_score = np.clip(consistency_score, 0, 1)
    return consistency_score

def calculate_reliability(driver_results: pd.DataFrame) -> float:
    """
    Calculates driver reliability based on race finishes.
    
    Returns:
        float: Reliability score between 0 and 1.
    """
    recent_results = driver_results.sort_values('date', ascending=False).head(10)
    if recent_results.empty:
        return 0.5  # Default reliability

    races_finished = (recent_results['status'] == 'Finished').sum()
    total_races = len(recent_results)
    reliability_score = races_finished / total_races
    reliability_score = np.clip(reliability_score, 0, 1)
    return reliability_score

def add_dynamic_features(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Adds dynamic features such as tire age, fuel load, and track position.
    
    Returns:
        pd.DataFrame: The laps DataFrame with dynamic features.
    """
    laps['tire_age'] = laps.groupby(['raceId', 'driverId'])['lap'].cumcount()
    laps['fuel_load'] = laps.groupby(['raceId', 'driverId'])['lap'].transform(lambda x: x.max() - x + 1)
    laps['track_position'] = laps['positionOrder']
    laps['is_pit_lap'] = laps['pitstop_milliseconds'].apply(lambda x: 1 if x > 0 else 0)
    laps['TrackStatus'].fillna(1, inplace=True)  # 1 = regular racing status
    return laps

def remove_lap_time_outliers(
    df: pd.DataFrame,
    iqr_multiplier: float = 1.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Removes lap time outliers using the IQR method.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Cleaned laps DataFrame and special laps DataFrame.
    """
    df = df.copy()
    normal_racing_mask = (
        (df['TrackStatus'] == 1) &
        (df['is_pit_lap'] == 0) &
        (df['milliseconds'] < 150000)
    )
    special_laps = df[~normal_racing_mask]
    normal_laps = df[normal_racing_mask]

    def remove_outliers_group(group):
        Q1 = group['milliseconds'].quantile(0.25)
        Q3 = group['milliseconds'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        return group[(group['milliseconds'] >= lower_bound) & (group['milliseconds'] <= upper_bound)]

    cleaned_normal_laps = normal_laps.groupby('circuitId').apply(remove_outliers_group).reset_index(drop=True)
    return cleaned_normal_laps, special_laps

def drop_unnecessary_columns(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Drops unnecessary columns from the laps DataFrame.
    
    Returns:
        pd.DataFrame: The laps DataFrame with unnecessary columns dropped.
    """
    columns_to_drop = ['EventFormat', 'EventName', 'S', 'SessionName', 'R', 'Time', 'Year', 'WindDirection', 
        'WindSpeed', 'WindSpeed', 'circuitRef', 'constructor_name', 'cumulative_milliseconds', 
        'date', 'date_race', 'dob', 'driverRef', 'fastestLap', 'forename', 'fp1_date', 
        'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date', 'fp3_time', 'location', 
        'name', 'name_x', 'name_y', 'number', 'quali_date', 
        'quali_date_time', 'rainfall', 'raceId_x', 'raceId_y', 'raceTime', 
        'RoundNumber', 'sprint_date', 'sprint_time', 'surname', 'time', 'time_race', 
        'url_race', 'url_x', 'url_y', 'year_x', 'year_y']
    laps.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')
    return laps

def handle_missing_values(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in the laps DataFrame.
    
    Returns:
        pd.DataFrame: The laps DataFrame with missing values handled.
    """
    # Ensure required columns are present
    race_features = RaceFeatures()
    required_columns = race_features.static_features + race_features.dynamic_features
    missing_columns = set(required_columns) - set(laps.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Drop rows with missing values in required columns
    laps.dropna(subset=required_columns, inplace=True)
    return laps

def save_auxiliary_data(
    laps: pd.DataFrame,
    drivers: pd.DataFrame,
    races: pd.DataFrame,
    special_laps: pd.DataFrame
) -> pd.DataFrame:
    """
    Saves auxiliary data such as driver attributes and race attributes.
    
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
    drivers_df.to_csv('data/util/drivers_attributes.csv', index=False)

    # Create race-specific attributes
    race_attributes_df = laps.groupby('raceId').agg({
        'circuitId': 'first',
        'TrackTemp': 'mean',
        'AirTemp': 'mean',
        'Humidity': 'mean'
    }).reset_index()
    race_attributes_df.to_csv('data/util/race_attributes.csv', index=False)

    # Save special laps
    special_laps.to_csv('data/SPECIAL_LAPS.csv', index=False)

    # Save processed laps
    laps.to_csv('data/LAPS.csv', index=False)
    return laps

def load_and_preprocess_data():
    # Ensure RaceFeatures is available
    race_features = RaceFeatures()

    # Preprocess data
    df = preprocess_data()

    # Validate that all required columns are present
    required_columns = race_features.static_features + race_features.dynamic_features
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Return the processed DataFrame
    return df

def save_data_splits(train_df, test_df):
    train_df.to_csv('data/train/train_data.csv', index=False)
    test_df.to_csv('data/test/test_data.csv', index=False)

def split_data_by_race(df, test_size=0.2, random_state=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df['raceId']))
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    return train_df, test_df

def load_data_splits() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the saved train/test splits."""
    train_df = pd.read_csv('data/train/train_data.csv')
    test_df = pd.read_csv('data/test/test_data.csv')
    return train_df, test_df

def prepare_regression_data(df):
    X = df[[
        'driver_overall_skill', 'driver_circuit_skill', 'driver_consistency',
        'driver_reliability', 'driver_aggression', 'driver_risk_taking',
        'fp1_median_time', 'fp2_median_time', 'fp3_median_time', 'quali_time',
        'tire_age', 'fuel_load', 'track_position', 'TrackTemp',
        'AirTemp', 'Humidity', 'tire_compound', 'TrackStatus', 'is_pit_lap'
    ]]
    y = df['milliseconds']
    return X, y

# data_preparation.py

import numpy as np

def prepare_sequence_data(df, race_features, window_size=3):
    """
    Prepare sequential data with a sliding window approach.
    Args:
    - df (DataFrame): Preprocessed DataFrame containing the race data.
    - race_features: Instance of the RaceFeatures class.
    - window_size (int): Number of laps to include in each sequence.
    
    Returns:
    - sequences: 3D numpy array of shape (num_sequences, window_size, num_features).
    - static_features: 2D numpy array of shape (num_sequences, num_static_features).
    - targets: 1D numpy array of shape (num_sequences,) containing target lap times.
    """
    sequences = []
    static_features = []
    targets = []

    # Sort the dataframe to ensure consistent ordering
    df = df.sort_values(['raceId', 'driverId', 'lap'])

    # Group by race and driver
    for (race_id, driver_id), group in df.groupby(['raceId', 'driverId']):
        group = group.sort_values('lap')

        # Extract static features (assumed to be constant per driver per race)
        static = group[race_features.static_features].iloc[0].values

        # Extract dynamic features and target
        lap_times = group[race_features.target].values.reshape(-1, 1)
        dynamic_features = group[race_features.dynamic_features].values

        # Create sequences using the sliding window
        
        if len(lap_times) > window_size:
            for i in range(len(lap_times) - window_size):
                # Sequence of lap times and dynamic features
                sequence_lap_times = lap_times[i:i + window_size]
                sequence_dynamic = dynamic_features[i:i + window_size]
                sequence = np.hstack((sequence_lap_times, sequence_dynamic))

                # Append the sequence, static features, and target
                sequences.append(sequence)
                static_features.append(static)
                targets.append(lap_times[i + window_size][0])  # Target is the lap time after the sequence
        else:
            # Handle cases where there are not enough laps
            continue

    return np.array(sequences), np.array(static_features), np.array(targets)

