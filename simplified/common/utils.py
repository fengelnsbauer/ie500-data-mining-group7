# common/utils.py

import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import Dict, Optional
from common.race import Race
import pandas as pd

def evaluate_race_simulation(race, race_data):
    """
    Mock evaluation function. Returns dummy metrics.
    """
    # In a real scenario, compare predicted times/positions to actual.
    metrics = {
        'rmse': 0.0,
        'mae': 0.0,
        'positions_correct': 0.0,
        'position_changes_accuracy': 0.0,
        'num_drivers': len(race.drivers)
    }
    detailed_results = pd.DataFrame({
        'driverId': [d.driver_id for d in race.drivers],
        'final_position': [d.current_position for d in race.drivers],
        'cumulative_time': [d.cumulative_race_time for d in race.drivers]
    })
    return {'metrics': metrics, 'detailed_results': detailed_results}

def plot_race_positions(race, constructor_mapping, driver_code_mapping, TEAM_COLORS, save_path=None):
    """
    Mock plotting function. Plots random positions over laps.
    In reality, you'd plot race positions vs. lap here.
    """
    positions_data = []
    laps = list(range(1, race.total_laps+1))
    for driver in race.drivers:
        # Mock data: just invert order and add noise
        pos_data = [driver.start_position for _ in laps]
        positions_data.append(pos_data)

    plt.figure(figsize=(10,6))
    for i, driver in enumerate(race.drivers):
        plt.plot(laps, positions_data[i], label=driver.name, color='tab:blue')
    plt.xlabel('Lap')
    plt.ylabel('Position')
    plt.title(f'Race {race.race_id} Positions')
    plt.gca().invert_yaxis()
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_lap_times(race, constructor_mapping, driver_code_mapping, TEAM_COLORS, save_path=None):
    """
    Mock plotting function. Plots the lap times recorded during the simulation.
    """
    laps = list(range(1, race.total_laps+1))
    plt.figure(figsize=(10,6))
    for driver in race.drivers:
        plt.plot(laps, driver.lap_times, label=driver.name)
    plt.xlabel('Lap')
    plt.ylabel('Lap Time (ms)')
    plt.title(f'Race {race.race_id} Lap Times')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_actual_vs_predicted(race, actual_laps_df):
    """
    Plot actual vs predicted lap times for each driver in the race.
    
    Parameters:
    - race: Race object that contains race.lap_data with predicted_lap_times
    - actual_laps_df: DataFrame containing actual lap times with at least:
        'raceId', 'driverId', 'lap', 'milliseconds'
    """

    # Filter actual laps for the current race
    actual_data = actual_laps_df[actual_laps_df['raceId'] == race.race_id]

    # Build a predicted DataFrame from race.lap_data
    predicted_data = []
    for d in race.drivers:
        # race.lap_data[d.driver_id]['predicted_lap_times'] holds predicted times per lap
        predicted_times = race.lap_data[d.driver_id].get('predicted_lap_times', [])
        for lap_idx, pred_time in enumerate(predicted_times, start=1):
            predicted_data.append({
                'driverId': d.driver_id,
                'lap': lap_idx,
                'predicted_lap_time': pred_time
            })

    predicted_df = pd.DataFrame(predicted_data)

    # Merge predicted with actual
    # Actual laps must have 'driverId', 'lap', 'milliseconds' columns
    merged = predicted_df.merge(
        actual_data[['driverId', 'lap', 'milliseconds']],
        on=['driverId', 'lap'],
        how='inner'
    )

    # Plot for each driver
    plt.figure(figsize=(10, 6))
    for driver_id in merged['driverId'].unique():
        driver_data = merged[merged['driverId'] == driver_id]
        # Plot actual times
        plt.plot(driver_data['lap'], driver_data['milliseconds'], label=f'Driver {driver_id} Actual', linestyle='--')
        # Plot predicted times
        plt.plot(driver_data['lap'], driver_data['predicted_lap_time'], label=f'Driver {driver_id} Predicted')

    plt.xlabel('Lap')
    plt.ylabel('Lap Time (ms)')
    plt.title(f'Race {race.race_id}: Actual vs Predicted Lap Times')
    plt.legend()
    plt.tight_layout()
    plt.show()