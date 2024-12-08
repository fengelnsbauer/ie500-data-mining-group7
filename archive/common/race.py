class Race:
    def __init__(
        self,
        race_id: int,
        circuit_id: int,
        total_laps: int,
        circuit_length: float,
        weather_conditions=None,
        safety_car_periods=None
    ):
        self.race_id = race_id
        self.circuit_id = circuit_id
        self.total_laps = total_laps
        self.circuit_length = circuit_length
        self.weather_conditions = weather_conditions or {}
        self.safety_car_periods = safety_car_periods or []
        self.drivers = []
        self.lap_data = {}

    def add_driver(self, driver: 'Driver'):
        self.drivers.append(driver)
