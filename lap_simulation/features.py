from dataclasses import dataclass, field
from typing import List

@dataclass
class RaceFeatures:
    static_features: List[str] = field(default_factory=lambda: [
        'driver_overall_skill', 'driver_circuit_skill', 'driver_consistency',
        'driver_reliability', 'driver_aggression', 'driver_risk_taking',
        'fp1_median_time', 'fp2_median_time', 'fp3_median_time', 'quali_time',
        'constructor_performance', 'circuitId'
    ])
    dynamic_features: List[str] = field(default_factory=lambda: [
        'tire_age', 'fuel_load', 'track_position', 'track_temp',
        'air_temp', 'humidity', 'tire_compound', 'TrackStatus', 'is_pit_lap'
    ])
    target: str = 'milliseconds'
