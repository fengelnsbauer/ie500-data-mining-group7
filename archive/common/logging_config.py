# common/logging_config.py
import logging
import torch    

def log_simulation_inputs(lap: int, driver_id: int, sequence_tensor: torch.Tensor, static_tensor: torch.Tensor):
    logging.debug(f"Lap {lap}, Driver {driver_id} sequence input:\n{sequence_tensor.cpu().numpy()}")
    logging.debug(f"Lap {lap}, Driver {driver_id} static input:\n{static_tensor.cpu().numpy()}")

def log_dynamic_features(lap: int, driver_id: int, dynamic_features: dict):
    logging.debug(f"Lap {lap}, Driver {driver_id} dynamic features after update: {dynamic_features}")
