from typing import List

class F1RaceFeatures:
    """
    Class to store feature definitions for F1 race data preprocessing.
    Replaces the previous RaceFeatures class with explicit column definitions.
    """
    
    def __init__(self):
        # Static features that remain constant or change very slowly
        self.static_features: List[str] = [
            'driver_overall_skill',
            'driver_circuit_skill',
            'driver_consistency',
            'driver_reliability',
            'driver_aggression',
            'driver_risk_taking',
            'constructor_performance',
            'fp1_median_time',
            'fp2_median_time',
            'fp3_median_time',
            'quali_time',
            'circuit_length',
            'circuit_type_encoded',
            'alt'
        ]

        # Dynamic features that change during the race
        self.dynamic_features: List[str] = [
            'tire_age',
            'fuel_load',
            'track_position',
            'TrackTemp',
            'AirTemp',
            'Humidity',
            'tire_compound',
            'TrackStatus',
            'is_pit_lap',
            'GapToLeader_ms',
            'IntervalToPositionAhead_ms'
        ]

        # Required columns for data processing
        self.required_columns: List[str] = [
            'raceId',
            'driverId',
            'lap',
            'position',
            'milliseconds',
            'date',
            'code',
            'nationality',
            'year',
            'round',
            'circuitId',
            'lat',
            'lng',
            'alt',
            'positionOrder',
            'grid',
            'status',
            'pitstop_milliseconds',
            'constructorId',
            'constructor_performance',
            'circuit_length',
            'circuit_type',
            'circuit_type_encoded',
            'cumulative_milliseconds',
            'seconds_from_start'
        ] + self.static_features + self.dynamic_features

        # Target variable
        self.target: str = 'milliseconds'

    def get_training_features(self) -> List[str]:
        """Returns the combined list of features used for model training."""
        return self.static_features + self.dynamic_features

    def get_feature_counts(self) -> tuple:
        """Returns the count of static and dynamic features."""
        return len(self.static_features), len(self.dynamic_features)