# concrete_simulator.py
import random
from common.base_simulator import BaseRaceSimulator
from common.race import Race
from common.driver import Driver

class SimpleRaceSimulator(BaseRaceSimulator):
    def __init__(self):
        super().__init__()
        # You can load or initialize additional resources here if needed

    def simulate_driver_lap(self, driver: Driver, lap: int, race: Race) -> float:
        """
        Simulates one lap for a driver using a simple model based on driver skills and car state.

        Args:
            driver: Driver object
            lap: Current lap number
            race: Race object

        Returns:
            Simulated lap time in milliseconds
        """
        # Base lap time derived from historical qualifying time
        base_lap_time = driver.static_features.get('quali_time', 90000)  # Default to 90,000 ms if missing

        # Adjust based on driver overall skill (higher skill reduces lap time)
        overall_skill = driver.static_features.get('driver_overall_skill', 1.0)
        lap_time = base_lap_time / overall_skill

        # Adjust based on driver consistency
        consistency = driver.static_features.get('driver_consistency', 1.0)
        lap_time *= (1.0 + (1.0 - consistency) * 0.01)  # Less consistency increases lap time slightly

        # Adjust based on tire condition
        tire_age = driver.dynamic_features.get('tire_age', 0)
        tire_degradation = self.get_tire_degradation(driver)
        lap_time += tire_degradation

        # Random variation based on driver reliability and risk-taking
        reliability = driver.static_features.get('driver_reliability', 1.0)
        risk_taking = driver.static_features.get('driver_risk_taking', 0.5)
        random_variation = random.uniform(-50 * reliability, 50 * (1 - reliability))
        lap_time += random_variation

        # Ensure lap time is within reasonable bounds
        lap_time = max(60000, lap_time)  # Minimum 60,000 ms
        lap_time = min(150000, lap_time)  # Maximum 150,000 ms

        # If it's a pit lap, add pit stop duration
        if driver.dynamic_features.get('is_pit_lap', 0):
            pitstop_ms = driver.dynamic_features.get('pitstop_milliseconds', self.pit_stop_duration)
            lap_time += pitstop_ms

        return lap_time

    def get_tire_degradation(self, driver: Driver) -> float:
        """
        Calculates tire degradation based on tire age and compound.

        Args:
            driver: Driver object

        Returns:
            Additional lap time due to tire degradation in milliseconds
        """
        tire_age = driver.dynamic_features.get('tire_age', 0)
        tire_compound = driver.dynamic_features.get('tire_compound', driver.starting_compound)
        compound_effect = self.TIRE_COMPOUND_EFFECTS.get(tire_compound, self.TIRE_COMPOUND_EFFECTS[1])
        degradation = compound_effect['degradation_per_lap'] * tire_age
        return degradation
