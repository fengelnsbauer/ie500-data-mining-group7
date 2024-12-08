class RaceFeatures:
    def __init__(self):
        # Static features now include race-level info like year, round, plus circuit_length
        self.static_features = [
            'driver_overall_skill', 'driver_circuit_skill', 'driver_consistency',
            'driver_reliability', 'driver_aggression', 'driver_risk_taking',
            'constructor_performance', 'fp1_median_time', 'fp2_median_time',
            'fp3_median_time', 'quali_time', 'circuit_length', 'circuit_type_encoded', 'alt',
            'year', 'round'  # Add race-level static features
        ]

        # Dynamic features now include 'lap' and all other lap-varying columns
        self.dynamic_features = [
            'lap', 'tire_age', 'fuel_load', 'TrackTemp', 'AirTemp', 'Humidity',
            'tire_compound', 'track_position', 'GapToLeader_ms', 'IntervalToPositionAhead_ms',
            'TrackStatus', 'is_pit_lap'
        ]
        
        self.target = 'milliseconds'
