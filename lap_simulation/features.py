from dataclasses import dataclass, field
from typing import List

@dataclass
class RaceFeatures:
    def __init__(self):
        self.static_features = [
            'driver_overall_skill', 'driver_circuit_skill', 'driver_consistency',
            'driver_reliability', 'driver_aggression', 'driver_risk_taking',
            'constructor_performance', 'fp1_median_time', 'fp2_median_time',
            'fp3_median_time', 'quali_time'
        ]
        self.dynamic_features = [
            'tire_age', 'fuel_load', 'track_position', 'TrackTemp',
            'AirTemp', 'Humidity', 'tire_compound', 'TrackStatus', 'is_pit_lap'
        ]
        self.target = 'milliseconds'

