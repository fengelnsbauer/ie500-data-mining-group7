# data_preparation.py
# Imports
import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Tuple
from sklearn.model_selection import GroupShuffleSplit

from features import RaceFeatures

na_values = ['\\N', 'NaN', '']
# Data preparation functions
def load_raw_data():
    lap_times = pd.read_csv('../data/raw_data/lap_times.csv', na_values=na_values)
    drivers = pd.read_csv('../data/raw_data/drivers.csv', na_values=na_values)
    races = pd.read_csv('../data/raw_data/races.csv', na_values=na_values)
    circuits = pd.read_csv('../data/raw_data/circuits.csv', na_values=na_values)
    pit_stops = pd.read_csv('../data/raw_data/pit_stops.csv', na_values=na_values)
    pit_stops.rename(columns={'milliseconds': 'pitstop_milliseconds'}, inplace=True)
    results = pd.read_csv('../data/raw_data/results.csv', na_values=na_values)
    results.rename(columns={'milliseconds': 'racetime_milliseconds'}, inplace=True)
    qualifying = pd.read_csv('../data/raw_data/qualifying.csv', na_values=na_values)
    status = pd.read_csv('../data/raw_data/status.csv', na_values=na_values)
    weather_data = pd.read_csv('../data/raw_data/ff1_weather.csv', na_values=na_values)
    practice_sessions = pd.read_csv('../data/raw_data/ff1_laps.csv', na_values=na_values)
    tire_data = pd.read_csv('../data/raw_data/ff1_laps.csv', na_values=na_values)
    
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
        'tire_data': tire_data
    }

def preprocess_data():
    # Load raw data
    data = load_raw_data()

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
    
    # Implement data preprocessing steps 
    # These include merging dataframes, feature engineering, and handling missing values
    # Convert date columns to datetime
    races['date'] = pd.to_datetime(races['date'])
    results['date'] = results['raceId'].map(races.set_index('raceId')['date'])
    lap_times['date'] = lap_times['raceId'].map(races.set_index('raceId')['date'])
    
    # Merge dataframes
    laps = lap_times.merge(drivers, on='driverId', how='left')
    print(laps.shape)
    laps = laps.merge(races, on='raceId', how='left', suffixes=('', '_race'))
    laps.rename(columns={'quali_time' : 'quali_date_time'}, inplace=True)
    print(laps.shape)
    laps = laps.merge(circuits, on='circuitId', how='left')
    print(laps.shape)
    laps = laps.merge(results[['raceId', 'driverId', 'positionOrder', 'grid', 'racetime_milliseconds', 'fastestLap', 'statusId']], on=['raceId', 'driverId'], how='left')
    print(laps.shape)
    laps = laps.merge(status, on='statusId', how='left')
    print(laps.shape)
    laps = laps.merge(pit_stops[['raceId', 'driverId', 'lap', 'pitstop_milliseconds']], on=['raceId', 'driverId', 'lap'], how='left')
    print(laps.shape)
    laps['pitstop_milliseconds'].fillna(0, inplace=True)  # Assuming 0 if no pit stop
    print(laps.shape)

    laps = laps[laps['year'] >= 2018]

    # Load additional data
    constructors = pd.read_csv('../data/raw_data/constructors.csv', na_values=na_values)
    constructor_results = pd.read_csv('../data/raw_data/constructor_results.csv', na_values=na_values)
    constructor_standings = pd.read_csv('../data/raw_data/constructor_standings.csv', na_values=na_values)
    
    # Merge constructors with drivers
    results = results.merge(constructors[['constructorId', 'name', 'nationality']], on='constructorId', how='left')
    results.rename(columns={'name': 'constructor_name', 'nationality': 'constructor_nationality'}, inplace=True)
    
    # Map driverId to constructorId
    driver_constructor = results[['raceId', 'driverId', 'constructorId']].drop_duplicates()
    
    # Merge driver_constructor into laps
    laps = laps.merge(driver_constructor, on=['raceId', 'driverId'], how='left')
    
    # Add constructor performance metrics
    # For simplicity, we'll use the constructor standings position as a performance metric
    constructor_standings_latest = constructor_standings.sort_values('raceId', ascending=False).drop_duplicates('constructorId')
    constructor_standings_latest = constructor_standings_latest[['constructorId', 'points', 'position']]
    constructor_standings_latest.rename(columns={'points': 'constructor_points', 'position': 'constructor_position'}, inplace=True)
    
    laps = laps.merge(constructor_standings_latest, on='constructorId', how='left')
    
    # Fill missing constructor performance data
    laps['constructor_points'].fillna(laps['constructor_points'].mean(), inplace=True)
    laps['constructor_position'].fillna(laps['constructor_position'].max(), inplace=True)
    
    # Add constructor performance as a static feature
    laps['constructor_performance'] = laps['constructor_points']
    
    # Add circuit characteristics
    # For simplicity, let's assume circuit length and type are available in circuits.csv
    circuits['circuit_length'] = 5.0  # Placeholder value, replace with actual data if available
    circuits['circuit_type'] = 'Permanent'  # Options could be 'Permanent', 'Street', 'Hybrid'
    
    # Merge circuit data into laps
    laps = laps.merge(circuits[['circuitId', 'circuit_length', 'circuit_type']], on='circuitId', how='left')
    
    # Encode circuit_type as a categorical variable
    circuit_type_mapping = {'Permanent': 0, 'Street': 1, 'Hybrid': 2}
    laps['circuit_type_encoded'] = laps['circuit_type'].map(circuit_type_mapping)
    
    # Add weather information
    # Filter weather data to include only the Race session
    weather_data = weather_data[weather_data['SessionName'] == 'R']
    
    # Merge weather data with races to get raceId
    weather_data = weather_data.merge(
        races[['raceId', 'year', 'name']], 
        left_on=['EventName', 'Year'],
        right_on=['name', 'year'],
        how='left'
    )
    
    # Compute cumulative time from the start of the race for each driver
    laps.sort_values(['raceId', 'driverId', 'lap'], inplace=True)
    laps['cumulative_milliseconds'] = laps.groupby(['raceId', 'driverId'])['milliseconds'].cumsum()
    laps['seconds_from_start'] = laps['cumulative_milliseconds'] / 1000
    print(laps.shape)
    
    # Use 'Time' in weather_data as 'seconds_from_start'
    weather_data['seconds_from_start'] = weather_data['Time']
    # Add weather information
    # Filter weather data to include only the Race session and process timestamps first
    weather_data = weather_data[weather_data['SessionName'] == 'R'].copy()  # Make a copy to avoid SettingWithCopyWarning

    # Process the Time column to get seconds from start
    weather_data['seconds_from_start'] = weather_data['Time'].astype(str).str.strip()
    weather_data['seconds_from_start'] = pd.to_timedelta(weather_data['seconds_from_start'], errors='coerce')
    weather_data['seconds_from_start'] = weather_data['seconds_from_start'].dt.total_seconds()

    # Clean up weather_data before merging with races
    weather_cols_to_keep = ['Time', 'AirTemp', 'Humidity', 'TrackTemp', 'EventName', 'Year', 'SessionName', 'seconds_from_start']
    weather_data = weather_data[weather_cols_to_keep]

    # Merge with races to get raceId
    weather_data = weather_data.merge(
        races[['raceId', 'year', 'name']], 
        left_on=['EventName', 'Year'],
        right_on=['name', 'year'],
        how='left'
    )

    # Clean up columns after merge
    final_cols = [
        'Time', 'AirTemp', 'Humidity', 'TrackTemp',
        'raceId', 'SessionName', 'seconds_from_start', 'EventName', 'Year'  # Added seconds_from_start to keep
    ]

    cols_to_drop = [col for col in weather_data.columns 
                    if col not in final_cols and col in weather_data.columns]
    weather_data = weather_data.drop(cols_to_drop, axis=1)
    
    # Standardize text data
    tire_data['Compound'] = tire_data['Compound'].str.upper()
    tire_data['EventName'] = tire_data['EventName'].str.strip().str.upper()
    races['name'] = races['name'].str.strip().str.upper()
    
    # Filter for race sessions only
    tire_data = tire_data[tire_data['SessionName'] == 'R']
    
    # Merge with races to get raceId
    tire_data = tire_data.merge(
        races[['raceId', 'year', 'name']],
        left_on=['Year', 'EventName'],
        right_on=['year', 'name'],
        how='left'
    )
    
    # Map driver codes to driverId
    tire_data['Driver'] = tire_data['Driver'].str.strip().str.upper()
    drivers['code'] = drivers['code'].str.strip().str.upper()
    driver_code_to_id = drivers.set_index('code')['driverId'].to_dict()
    tire_data['driverId'] = tire_data['Driver'].map(driver_code_to_id)
    
    # Rename 'LapNumber' to 'lap' and ensure integer type
    tire_data.rename(columns={'LapNumber': 'lap'}, inplace=True)
    tire_data['lap'] = tire_data['lap'].astype(int)
    laps['lap'] = laps['lap'].astype(int)
    
    # Create compound mapping (ordered from hardest to softest)
    compound_mapping = {
        'UNKNOWN': 0,
        np.nan: 0,
        'HARD': 1,
        'MEDIUM': 2,
        'SOFT': 3,
        'SUPERSOFT': 3,    # Treat as "Soft"
        'ULTRASOFT': 3,    # Treat as "Soft"
        'HYPERSOFT': 3,    # Treat as "Soft"
        'INTERMEDIATE': 4,
        'WET': 5
    }
    # Merge tire_data with laps
    laps = laps.merge(
        tire_data[['raceId', 'driverId', 'lap', 'Compound', 'TrackStatus']],
        on=['raceId', 'driverId', 'lap'],
        how='left'
    )

    
    # Handle missing compounds and apply numeric encoding
    laps['Compound'].fillna('UNKNOWN', inplace=True)
    laps['tire_compound'] = laps['Compound'].map(compound_mapping)
    
    # Drop the original Compound column if desired
    laps.drop('Compound', axis=1, inplace=True)
    
    # Standardize names
    practice_sessions['EventName'] = practice_sessions['EventName'].str.strip().str.upper()
    races['name'] = races['name'].str.strip().str.upper()
    
    # Merge practice_sessions with races to get raceId
    practice_sessions = practice_sessions.merge(
        races[['raceId', 'year', 'name']],
        left_on=['Year', 'EventName'],
        right_on=['year', 'name'],
        how='left'
    )
    
    # Map driver codes to driverId
    practice_sessions['Driver'] = practice_sessions['Driver'].str.strip().str.upper()
    drivers['code'] = drivers['code'].str.strip().str.upper()
    driver_code_to_id = drivers.set_index('code')['driverId'].to_dict()
    practice_sessions['driverId'] = practice_sessions['Driver'].map(driver_code_to_id)
    
    # Convert LapTime to milliseconds
    practice_sessions['LapTime_ms'] = practice_sessions['LapTime'].apply(lambda x: pd.to_timedelta(x).total_seconds() * 1000)
    
    # Calculate median lap times for each driver in each session (per race)
    session_medians = practice_sessions.groupby(['raceId', 'driverId', 'SessionName'])['LapTime_ms'].median().reset_index()

    # Pivot the data to have sessions as columns (preserves raceId and driverId context)
    session_medians_pivot = session_medians.pivot_table(
        index=['raceId', 'driverId'],  # Keep raceId and driverId as indices
        columns='SessionName',
        values='LapTime_ms'
    ).reset_index()

    # Rename columns for clarity
    session_medians_pivot.rename(columns={
        'FP1': 'fp1_median_time',
        'FP2': 'fp2_median_time',
        'FP3': 'fp3_median_time',
        'Q': 'quali_time'
    }, inplace=True)


    laps = laps.merge(
        session_medians_pivot,
        on=['raceId', 'driverId'],  # Merge on raceId and driverId
        how='left'  # Ensure all laps data is retained
    )

    # Calculate global median times for each FP session within each race
    global_medians = session_medians.groupby('SessionName')['LapTime_ms'].median()

    # Fill missing FP times for each session
    laps['fp1_median_time'].fillna(global_medians.get('FP1', 0), inplace=True)
    laps['fp2_median_time'].fillna(global_medians.get('FP2', 0), inplace=True)
    laps['fp3_median_time'].fillna(global_medians.get('FP3', 0), inplace=True)
    laps['quali_time'].fillna(global_medians.get('Q', 0), inplace=True)


    # For missing session times, fill with global means for that session within the same race
    for session in ['fp1_median_time', 'fp2_median_time', 'fp3_median_time', 'quali_time']:
        global_mean_column = f'global_mean_{session}'
        laps[global_mean_column] = laps.groupby('raceId')[session].transform('mean')  # Calculate global mean for the session
        laps[session] = laps[session].fillna(laps[global_mean_column])  # Fill missing session times with global means
        laps.drop(columns=[global_mean_column], inplace=True)  # Remove intermediate column

    # Ensure session times are now part of the driver-specific attributes


    
    # Create a binary indicator for pit stops
    laps['is_pit_lap'] = laps['pitstop_milliseconds'].apply(lambda x: 1 if x > 0 else 0)

    
    # Define a function to match weather data to laps
    def match_weather_to_lap(race_laps, race_weather):
        try:
            # Copy DataFrames to avoid modifying originals
            race_laps = race_laps.copy()
            race_weather = race_weather.copy()
            
            # Convert 'seconds_from_start' to numeric
            race_laps['seconds_from_start'] = pd.to_numeric(race_laps['seconds_from_start'], errors='coerce')
            race_weather['seconds_from_start'] = pd.to_numeric(race_weather['seconds_from_start'], errors='coerce')
            
            # Drop rows with NaN in 'seconds_from_start'
            race_laps.dropna(subset=['seconds_from_start'], inplace=True)
            race_weather.dropna(subset=['seconds_from_start'], inplace=True)
            
            # **Select only necessary columns from race_weather**
            race_weather = race_weather[['seconds_from_start', 'TrackTemp', 'AirTemp', 'Humidity']]
            
            # Sort DataFrames by 'seconds_from_start'
            race_laps.sort_values('seconds_from_start', inplace=True)
            race_weather.sort_values('seconds_from_start', inplace=True)
            
            # Perform the asof merge
            merged = pd.merge_asof(
                race_laps,
                race_weather,
                on='seconds_from_start',
                direction='nearest'
            )

            # Debugging output
            print(merged[['seconds_from_start', 'TrackTemp', 'AirTemp', 'Humidity']].head())
            
            if merged.empty:
                print("Warning: Merged result is empty. Check if there is overlapping time data.")
                
            return merged
        
        except Exception as e:
            print(f"Error in matching weather to lap data: {str(e)}")
            print("\nRace laps seconds_from_start dtype:", race_laps['seconds_from_start'].dtype)
            print("Race weather seconds_from_start dtype:", race_weather['seconds_from_start'].dtype)
            print("\nSample of race_laps seconds_from_start:", race_laps['seconds_from_start'].head())
            print("\nSample of race_weather seconds_from_start:", race_weather['seconds_from_start'].head())
            raise


    def match_weather_to_laps(laps, races, weather_data):
        # Initialize list to store merged laps data
        merged_laps_list = []

        for race_id in laps['raceId'].unique():
            race_laps = laps[laps['raceId'] == race_id].copy()
            race_info = races[races['raceId'] == race_id][['year', 'name']].iloc[0]
            race_year = race_info['year']
            race_name = race_info['name']
            race_weather = weather_data[weather_data['raceId'] == race_id]

            print(f"\nProcessing race {race_id} ({race_year} {race_name})")

            if not race_weather.empty:
                try:
                    # Perform the matching
                    merged = match_weather_to_lap(race_laps, race_weather)
                    print(f"Successfully matched weather data - Found {len(race_weather)} weather records")
                    merged_laps_list.append(merged)
                except Exception as e:
                    print(f"Error matching weather data for race {race_id}: {str(e)}")
                    # If there's an error, append race_laps without weather data
                    merged_laps_list.append(race_laps)
            else:
                print(f"No weather data found for race - Assigning default weather values")
                # Assign default weather values to race_laps
                race_laps['TrackTemp'] = 25.0
                race_laps['AirTemp'] = 20.0
                race_laps['Humidity'] = 50.0
                merged_laps_list.append(race_laps)

        # Concatenate all race laps back into a single DataFrame
        laps_with_weather = pd.concat(merged_laps_list, ignore_index=True)

        # Fill any remaining missing weather data with defaults
        laps_with_weather['TrackTemp'].fillna(25.0, inplace=True)
        laps_with_weather['AirTemp'].fillna(20.0, inplace=True)
        laps_with_weather['Humidity'].fillna(50.0, inplace=True)

        print("Columns in laps DataFrame:", laps_with_weather.columns.tolist())
        print("Sample data after merging weather data:")
        print(laps_with_weather[['raceId', 'lap', 'TrackTemp', 'AirTemp', 'Humidity']].head())


        # **Drop any duplicated columns resulting from the merge**
        # For example, if 'lap' appears as 'lap_x' and 'lap_y'
        columns_to_drop = [col for col in laps_with_weather.columns if col.endswith('_x') or col.endswith('_y')]
        laps_with_weather.drop(columns=columns_to_drop, inplace=True, errors='ignore')

        print("Final weather columns added and cleaned:")
        print(laps_with_weather[['raceId', 'lap', 'TrackTemp', 'AirTemp', 'Humidity']].head())

        return laps_with_weather


    print(f"Shape of laps before merging weather data: {laps.shape}")
    laps = match_weather_to_laps(laps, races, weather_data)
    print(f"Shape of laps after merging weather data: {laps.shape}")

    # Check for duplicates after merging weather data
    duplicates = laps[laps.duplicated()]
    if not duplicates.empty:
        print(f"Warning: {len(duplicates)} duplicate rows found after merging weather data.")
        print("Sample of duplicate rows:")
        print(duplicates.head())
    else:
        print("No duplicate rows found after merging weather data.")

    # Calculate driver aggression and skill
    # Create driver names
    drivers['driver_name'] = drivers['forename'] + ' ' + drivers['surname']
    driver_mapping = drivers[['driverId', 'driver_name']].copy()
    driver_mapping.set_index('driverId', inplace=True)
    driver_names = driver_mapping['driver_name'].to_dict()
    
    # Map statusId to status descriptions
    status_dict = status.set_index('statusId')['status'].to_dict()
    results['status'] = results['statusId'].map(status_dict)
    
    # Calculate driver aggression and skill
    def calculate_aggression(driver_results):
        if len(driver_results) == 0:
            return 0.5  # Default aggression for new drivers
        
        # Only consider recent races for more current behavior
        recent_results = driver_results.sort_values('date', ascending=False).head(20)
        
        # Calculate overtaking metrics
        positions_gained = recent_results['grid'] - recent_results['positionOrder']
        
        # Calculate risk metrics
        dnf_rate = (recent_results['status'] != 'Finished').mean()
        incidents = (recent_results['statusId'].isin([
            4,  # Collision
            5,  # Spun off
            6,  # Accident
            20, # Collision damage
            82, # Collision with another driver
        ])).mean()
        
        # Calculate overtaking success rate (normalized between 0-1)
        positive_overtakes = (positions_gained > 0).sum()
        negative_overtakes = (positions_gained < 0).sum()
        total_overtake_attempts = positive_overtakes + negative_overtakes
        overtake_success_rate = positive_overtakes / total_overtake_attempts if total_overtake_attempts > 0 else 0.5
        
        # Normalize average positions gained (0-1)
        avg_positions_gained = positions_gained[positions_gained > 0].mean() if len(positions_gained[positions_gained > 0]) > 0 else 0
        max_possible_gain = 20  # Maximum grid positions that could be gained
        normalized_gains = np.clip(avg_positions_gained / max_possible_gain, 0, 1)
        
        # Normalize risk factors (0-1)
        normalized_dnf = np.clip(dnf_rate, 0, 1)
        normalized_incidents = np.clip(incidents, 0, 1)
        
        # Calculate component scores (each between 0-1)
        overtaking_component = (normalized_gains * 0.6 + overtake_success_rate * 0.4)
        risk_component = (normalized_dnf * 0.5 + normalized_incidents * 0.5)
        
        # Combine components with weights (ensuring sum of weights = 1)
        weights = {
            'overtaking': 0.4,  # Aggressive overtaking
            'risk': 0.5,       # Risk-taking behavior
            'baseline': 0.1    # Baseline aggression
        }
        
        aggression = (
            overtaking_component * weights['overtaking'] +
            risk_component * weights['risk'] +
            0.5 * weights['baseline']  # Baseline aggression factor
        )
        
        # Add small random variation while maintaining 0-1 bounds
        variation = np.random.normal(0, 0.02)
        aggression = np.clip(aggression + variation, 0, 1)
        
        return aggression
    
    def calculate_consistency(lap_times_df, driver_id, num_recent_races=5):
        """
        Calculate driver consistency based on lap time variability over recent races.
        Args:
            lap_times_df (DataFrame): DataFrame containing lap times.
            driver_id (int): Driver ID.
            num_recent_races (int): Number of recent races to consider.
        Returns:
            float: Consistency score between 0 and 1.
        """
        # Filter lap times for the driver
        driver_lap_times = lap_times_df[lap_times_df['driverId'] == driver_id]
        
        # Get recent race IDs
        recent_races = driver_lap_times['raceId'].unique()
        recent_races = sorted(recent_races, reverse=True)[:num_recent_races]
        
        # Filter lap times for recent races
        recent_lap_times = driver_lap_times[driver_lap_times['raceId'].isin(recent_races)]
        
        if recent_lap_times.empty or len(recent_lap_times) < 5:
            return 0.5  # Default consistency for insufficient data
        
        # Calculate lap time variance
        lap_time_variance = recent_lap_times['milliseconds'].var()
        
        # Normalize variance (lower variance means higher consistency)
        # Invert and normalize between 0 and 1
        max_variance = lap_times_df['milliseconds'].var()
        consistency_score = 1 - (lap_time_variance / max_variance)
        consistency_score = np.clip(consistency_score, 0, 1)
        
        return consistency_score

    def calculate_reliability(driver_results, num_recent_races=10):
        """
        Calculate driver reliability based on race finishes over recent races.
        Args:
            driver_results (DataFrame): DataFrame containing driver results.
            num_recent_races (int): Number of recent races to consider.
        Returns:
            float: Reliability score between 0 and 1.
        """
        # Get recent races
        recent_results = driver_results.sort_values('date', ascending=False).head(num_recent_races)
        
        if recent_results.empty or len(recent_results) == 0:
            return 0.5  # Default reliability for new drivers
        
        # Calculate proportion of races finished
        races_finished = (recent_results['status'] == 'Finished').sum()
        total_races = len(recent_results)
        reliability_score = races_finished / total_races
        reliability_score = np.clip(reliability_score, 0, 1)
        
        return reliability_score

    
    # Modify calculate_skill function
    def calculate_skill(driver_data, results_data, circuit_id, constructor_performance):
        driver_results = results_data[
            (results_data['driverId'] == driver_data['driverId']) & 
            (results_data['circuitId'] == circuit_id)
        ].sort_values('date', ascending=False).head(10)  # Use last 10 races at circuit
        
        if len(driver_results) == 0:
            return 0.5  # Default skill
        
        # Calculate performance metrics
        avg_finish_pos = driver_results['positionOrder'].mean()
        avg_quali_pos = driver_results['grid'].mean()
        points_per_race = driver_results['points'].mean()
        fastest_laps = (driver_results['rank'] == 1).mean()
        
        # Include constructor performance
        constructor_factor = np.exp(-constructor_performance / 100)
        
        # Improved normalization (exponential decay for positions)
        normalized_finish_pos = np.exp(-avg_finish_pos/5) # Better spread of values
        normalized_quali_pos = np.exp(-avg_quali_pos/5)
        
        # Points normalization with improved scaling
        max_points_per_race = 26  # Maximum possible points (25 + 1 fastest lap)
        normalized_points = points_per_race / max_points_per_race
        
        # Weighted combination with more factors
        weights = {
            'finish': 0.35,
            'quali': 0.25,
            'points': 0.25,
            'fastest_laps': 0.15
        }
        
        skill = (
            weights['finish'] * normalized_finish_pos +
            weights['quali'] * normalized_quali_pos +
            weights['points'] * normalized_points +
            weights['fastest_laps'] * fastest_laps +
            0.1 * constructor_factor  # Adjust weight as needed
        )
        
        # Add random variation to prevent identical skills
        skill = np.clip(skill + np.random.normal(0, 0.05), 0.1, 1.0)
    
        return skill
    
    # First merge results with races to get circuitId
    results = results.merge(
        races[['raceId', 'circuitId']], 
        on='raceId',
        how='left'
    )

    # Map statusId to status descriptions
    status_dict = status.set_index('statusId')['status'].to_dict()
    results['status'] = results['statusId'].map(status_dict)
    
    # Copy lap_times DataFrame for consistency calculations
    lap_times_df = lap_times.copy()
    
    # Now calculate driver aggression, skill, consistency, and reliability
    driver_aggression = {}
    driver_skill = {}
    driver_consistency = {}
    driver_reliability = {}
    for driver_id in drivers['driverId'].unique():
        driver_results = results[results['driverId'] == driver_id]
        
        # Calculate aggression
        aggression = calculate_aggression(driver_results)
        driver_aggression[driver_id] = aggression
        
        # Calculate skill
        recent_race = driver_results.sort_values('date', ascending=False).head(1)
        if not recent_race.empty:
            circuit_id = recent_race['circuitId'].iloc[0]
            constructor_performance = laps.loc[laps['driverId'] == driver_id, 'constructor_performance'].mean()
            skill = calculate_skill({'driverId': driver_id}, results, circuit_id, constructor_performance)
            driver_skill[driver_id] = skill
        else:
            driver_skill[driver_id] = 0.5  # Default skill for new drivers
        
        # Calculate consistency
        consistency = calculate_consistency(lap_times_df, driver_id, num_recent_races=5)
        driver_consistency[driver_id] = consistency
        
        # Calculate reliability
        reliability = calculate_reliability(driver_results, num_recent_races=10)
        driver_reliability[driver_id] = reliability
    
    # Map calculated metrics back to laps DataFrame
    laps['driver_aggression'] = laps['driverId'].map(driver_aggression)
    laps['driver_overall_skill'] = laps['driverId'].map(driver_skill)
    laps['driver_circuit_skill'] = laps['driver_overall_skill']  # For simplicity, using overall skill
    laps['driver_consistency'] = laps['driverId'].map(driver_consistency)
    laps['driver_reliability'] = laps['driverId'].map(driver_reliability)
    laps['driver_risk_taking'] = laps['driver_aggression']  # Assuming similar to aggression
    
    
    # Dynamic features
    laps['tire_age'] = laps.groupby(['raceId', 'driverId'])['lap'].cumcount()
    laps['fuel_load'] = laps.groupby(['raceId', 'driverId'])['lap'].transform(lambda x: x.max() - x + 1)
    laps['track_position'] = laps['position']  # Assuming 'position' is available in laps data
    
    # Ensure that all required columns are present
    # Create an instance of RaceFeatures
    race_features = RaceFeatures()

    
    laps['TrackStatus'].fillna(1, inplace=True)  # 1 = regular racing status

    # Continue with outlier removal and other preprocessing steps
    def remove_lap_time_outliers(df, iqr_multiplier=1.5):
        """
        Remove lap time outliers using IQR method, considering only normal racing laps.
        
        Args:
            df: DataFrame containing lap times and related features
            iqr_multiplier: IQR multiplier for outlier threshold (default 1.5)
        
        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()
        print(f"Shape before filtering and outlier removal: {df.shape}")
        
        # Filter out special laps
        normal_racing_mask = (
            (df['TrackStatus'] == 1) &      # Normal racing conditions
            (df['is_pit_lap'] == 0) &       # No pit stops
            #(df['lap'] > 1) &
            (df['milliseconds'] < 120000)   # Exclude first lap
        )
        
        special_laps = df[~normal_racing_mask]
        normal_laps = df[normal_racing_mask]
        
        print(f"Normal racing laps: {normal_laps.shape}")
        print(f"Special laps (pit stops, safety car, etc.): {special_laps.shape}")
        
        def remove_outliers_group(group):
            Q1 = group['milliseconds'].quantile(0.25)
            Q3 = group['milliseconds'].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            
            return group[(group['milliseconds'] >= lower_bound) & 
                        (group['milliseconds'] <= upper_bound)]
        
        # Remove outliers only from normal racing laps
        cleaned_normal_laps = normal_laps.groupby('circuitId').apply(remove_outliers_group)
        cleaned_normal_laps.reset_index(drop=True, inplace=True)
        
        # Combine cleaned normal laps with special laps
        special = pd.concat([cleaned_normal_laps, special_laps], ignore_index=True)
        
        print(f"Final shape after outlier removal: {special.shape}")
        
        return cleaned_normal_laps, special

    # Apply outlier removal
    laps, special_laps = remove_lap_time_outliers(laps)
    
    # Update required columns
    race_features = RaceFeatures()
    required_columns = race_features.static_features + race_features.dynamic_features
    missing_columns = set(required_columns) - set(laps.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Drop rows with missing values in required columns
    laps = laps[laps['year'] >= 2018]
    laps = laps.dropna(subset=required_columns)

    # Create driver-specific attributes per race
    drivers_df = laps.groupby(['driverId', 'raceId']).agg({
        'driver_overall_skill': 'last',
        'driver_circuit_skill': 'last',
        'driver_consistency': 'last',
        'driver_reliability': 'last',
        'driver_aggression': 'last',
        'driver_risk_taking': 'last',
        'constructor_performance': 'last',  # Most recent constructor performance
        'fp1_median_time': 'last',  # Include session times
        'fp2_median_time': 'last',
        'fp3_median_time': 'last',
        'quali_time': 'last',
    }).reset_index()

    # Save the driver-specific attributes
    drivers_df.to_csv('data/util/drivers_attributes.csv', index=False)

    # For race_attributes_df - aggregate race-specific attributes
    race_attributes_df = laps.groupby('raceId').agg({
        'circuitId': 'first',
        'TrackTemp': 'mean',
        'AirTemp': 'mean',
        'Humidity': 'mean'
    }).reset_index()

    # Save the race-specific attributes
    race_attributes_df.to_csv('data/util/race_attributes.csv', index=False)

    # Drop unnecessary columns
    columns_to_drop = [
        'raceId_x', 'year_y', 'Time', 'Pressure', 'Rainfall', 'WindDirection', 'WindSpeed', 'Year', 'year_x', 'EventName', 
        'SessionName', 'fp1_date', 'fp2_date', 'fp3_date', 'fp1_time', 'fp2_time',
        'fp3_time', 'racetime_milliseconds', 'raceId_y', 'RoundNumber', 'name', 
        'R', 'S', 'EventFormat', 'quali_date', 'quali_date_time', 'sprint_date', 
        'sprint_time', 'url_y', 'fastestLap'
    ]
    laps = laps.drop(columns=columns_to_drop, axis=1, errors='ignore')

    # Final race_attributes_df
    race_attributes_df = laps[['raceId', 'circuitId', 'fp1_median_time', 'fp2_median_time',
                               'fp3_median_time', 'quali_time', 'TrackTemp', 'AirTemp', 'Humidity']].drop_duplicates()

    race_attributes_df.to_csv('data/util/race_attributes.csv', index=False)

    special_laps.to_csv('data/SPECIAL_LAPS.csv', index=False)

    laps.to_csv('data/LAPS.csv', index=False)

    # Return the preprocessed DataFrame
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
        for i in range(len(lap_times) - window_size):
            # Sequence of lap times and dynamic features
            sequence_lap_times = lap_times[i:i + window_size]
            sequence_dynamic = dynamic_features[i:i + window_size]
            sequence = np.hstack((sequence_lap_times, sequence_dynamic))

            # Append the sequence, static features, and target
            sequences.append(sequence)
            static_features.append(static)
            targets.append(lap_times[i + window_size][0])  # Target is the lap time after the sequence

    return np.array(sequences), np.array(static_features), np.array(targets)

