class Driver:
    def __init__(self, driver_id: int,
                name: str,
                code: str,
                nationality: str,
                constructor_id: int,
                constructor_name: str):

        # Core identification attributes that won't change
        self.driver_id = driver_id
        self.name = name
        self.code = code
        self.nationality = nationality
        self.constructor_id = constructor_id
        self.constructor_name = constructor_name

        # Static features - will be populated with actual values during initialization
        self.static_features = {
            # Driver characteristics
            'driver_overall_skill': 0.0,
            'driver_circuit_skill': 0.0,
            'driver_consistency': 0.0,
            'driver_aggression': 0.0,
            'driver_reliability': 0.0,
            'driver_risk_taking': 0.0,
            
            # Constructor characteristics
            'constructor_performance': 0.0,
            'constructor_position': 0,
            'constructor_nationality': '',
            
            # Historical performance
            'fp1_median_time': 0.0,
            'fp2_median_time': 0.0,
            'fp3_median_time': 0.0,
            'quali_time': 0.0,
            
            # Race start configuration
            'grid_position': 0,
            'starting_compound': 0
        }

        # Dynamic features - will be updated throughout the race
        self.dynamic_features = {
            # Race progress
            'current_position': 0,
            'lap': 1,
            'cumulative_race_time': 0.0,
            'lap_times': [],
            
            # Car state
            'tire_compound': 0,
            'tire_age': 0,
            'fuel_load': 100.0,
            'is_pit_lap': 0,
            
            # Race gaps
            'gaptoLeader_ms': 0,
            'intervaltopositionahead_ms': 0,
            
            # Driver status
            'status': 1
        }

    def update_race_progress(self, lap_time: float, circuit_length: float):
        self.dynamic_features['lap_times'].append(lap_time)
        self.dynamic_features['cumulative_race_time'] += lap_time