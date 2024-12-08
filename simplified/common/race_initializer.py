from typing import Dict, List, Tuple, Optional
import pandas as pd
from common.race import Race
from common.driver import Driver
import logging

class RaceInitializer:
    """Handles race initialization for all simulation models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def initialize_race(self, 
                       race_id: int,
                       lap_times_df: pd.DataFrame,
                       circuit_attributes_df: pd.DataFrame,
                       drivers_attributes_df: pd.DataFrame,
                       race_attributes_df: pd.DataFrame,
                       weather_data_df: pd.DataFrame) -> Race:
        """
        Initialize a race with all necessary data and configurations.
        
        Args:
            race_id: Identifier for the race
            lap_times_df: DataFrame containing lap-by-lap data
            circuit_attributes_df: DataFrame containing circuit information
            drivers_attributes_df: DataFrame containing driver attributes
            race_attributes_df: DataFrame containing race-specific attributes
            weather_data_df: DataFrame containing weather progression data
        
        Returns:
            Initialized Race object
        """
        # Get race-specific data
        race_data = lap_times_df[lap_times_df['raceId'] == race_id].iloc[0]
        circuit_data = circuit_attributes_df[
            circuit_attributes_df['circuitId'] == race_data['circuitId']
        ].iloc[0]
        
        # Create base race object
        race = Race(
            race_id=race_id,
            circuit_id=race_data['circuitId'],
            year=race_data['year'],
            total_laps=self.get_race_length(race_id, lap_times_df),
            circuit_length=circuit_data['circuit_length'],
            circuit_country=race_data.get('country'),
            circuit_lat=circuit_data['lat'],
            circuit_lng=circuit_data['lng']
        )
        
        # Initialize drivers
        self._initialize_drivers(race, race_id, lap_times_df, drivers_attributes_df)
        
        # Set up pit strategies
        pit_strategies = self.extract_pit_strategies(lap_times_df, race_id)
        self._apply_pit_strategies(race, pit_strategies)
        
        # Set up safety car periods
        safety_car_periods = self.extract_safety_car_periods(lap_times_df, race_id)
        race.safety_car_periods = safety_car_periods
        
        # Initialize weather data
        self._initialize_weather(race, race_id, weather_data_df)
        
        return race
    
    def _initialize_drivers(self, 
                          race: Race, 
                          race_id: int, 
                          lap_times_df: pd.DataFrame,
                          drivers_attributes_df: pd.DataFrame):
        """Initialize all drivers for the race with their attributes."""
        race_drivers = lap_times_df[
            (lap_times_df['raceId'] == race_id) & 
            (lap_times_df['lap'] == 1)
        ]
        
        for _, driver_data in race_drivers.iterrows():
            # Create driver instance
            driver = Driver(
                driver_id=driver_data['driverId'],
                name=driver_data['code'],  # Using code as name
                code=driver_data['code'],
                nationality=driver_data['nationality'],
                constructor_id=driver_data['constructorId'],
                constructor_name=driver_data['constructor_nationality']
            )
            
            # Get driver attributes
            driver_attrs = drivers_attributes_df[
                (drivers_attributes_df['driverId'] == driver_data['driverId']) &
                (drivers_attributes_df['raceId'] == race_id)
            ].iloc[0]
            
            # Set static features
            driver.static_features.update({
                'driver_overall_skill': driver_attrs['driver_overall_skill'],
                'driver_circuit_skill': driver_attrs['driver_circuit_skill'],
                'driver_consistency': driver_attrs['driver_consistency'],
                'driver_aggression': driver_attrs['driver_aggression'],
                'driver_reliability': driver_attrs['driver_reliability'],
                'driver_risk_taking': driver_attrs['driver_risk_taking'],
                'constructor_performance': driver_data['constructor_performance'],
                'constructor_position': driver_data['constructor_position'],
                'constructor_nationality': driver_data['constructor_nationality'],
                'fp1_median_time': driver_attrs['fp1_median_time'],
                'fp2_median_time': driver_attrs['fp2_median_time'],
                'fp3_median_time': driver_attrs['fp3_median_time'],
                'quali_time': driver_attrs['quali_time'],
                'grid_position': driver_data['grid']
            })
            
            # Set initial dynamic features
            driver.dynamic_features.update({
                'current_position': driver_data['grid'],
                'tire_compound': driver_data['tire_compound'],
                'tire_age': 0,
                'fuel_load': 100.0,
                'is_pit_lap': 0
            })
            
            race.drivers.append(driver)
    
    def _apply_pit_strategies(self, race: Race, pit_strategies: Dict):
        """Apply pit strategies to each driver."""
        for driver in race.drivers:
            if driver.driver_id in pit_strategies:
                strategy = pit_strategies[driver.driver_id]
                driver.static_features['starting_compound'] = strategy['starting_compound']
                driver.pit_strategy = strategy['pit_strategy']
    
    def _initialize_weather(self, race: Race, race_id: int, weather_data_df: pd.DataFrame):
        """Initialize weather data for the race."""
        race.weather_df = weather_data_df[weather_data_df['raceId'] == race_id].copy()
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
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