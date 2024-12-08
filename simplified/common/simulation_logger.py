# simulation_logger.py
import pandas as pd
import logging
from typing import Dict, List
from datetime import datetime

class SimulationLogger:
    def __init__(self, log_level=logging.INFO):
        self.simulation_data = []
        self.setup_logging(log_level)
        
    def setup_logging(self, log_level):
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=f'race_simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
    def log_lap_data(self, race_id: int, lap: int, driver_id: int, 
                     static_features: Dict, dynamic_features: Dict, 
                     race_conditions: Dict, lap_time: float):
        data_entry = {
            'race_id': race_id,
            'lap': lap,
            'driver_id': driver_id,
            'lap_time': lap_time,
            'timestamp': datetime.now(),
            **static_features,
            **dynamic_features,
            **race_conditions
        }
        self.simulation_data.append(data_entry)
        logging.info(f"Lap {lap} completed for driver {driver_id} - Time: {lap_time}")
        
    def export_simulation_data(self, filepath: str):
        df = pd.DataFrame(self.simulation_data)
        df.to_csv(filepath, index=False)
        logging.info(f"Simulation data exported to {filepath}")