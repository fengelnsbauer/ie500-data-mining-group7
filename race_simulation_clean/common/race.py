# race.py
from typing import Optional

class Race:
    def __init__(self,
                race_id: int,
                circuit_id: int,
                year: int,
                total_laps: int,
                circuit_length: float,
                circuit_country: Optional[str] = None,
                circuit_lat: Optional[float] = None,
                circuit_lng: Optional[float] = None
            ):
        # Core race identification
        self.race_id = race_id
        self.circuit_id = circuit_id
        self.year = year
        self.total_laps = total_laps
        self.circuit_length = circuit_length
        
        # Circuit information
        self.circuit_country = circuit_country
        self.circuit_lat = circuit_lat
        self.circuit_lng = circuit_lng
        
        # Race conditions that affect all drivers
        self.current_conditions = {
            'track_status': 1,
            'track_temp': 0.0,
            'air_temp': 0.0,
            'humidity': 0.0
        }
        self.lap = 0
        
        # Safety car and other race events
        self.safety_car_periods = []
        
        # Driver management
        self.drivers = []
        self.lap_data = {}

    def check_safety_car(self, lap: int) -> bool:
        return any(start <= lap <= end for start, end in self.safety_car_periods)

    def update_race_conditions(self, lap: int, new_conditions: dict):
        self.lap = lap
        self.current_conditions.update(new_conditions)
        if self.check_safety_car(lap):
            self.current_conditions['track_status'] = 4