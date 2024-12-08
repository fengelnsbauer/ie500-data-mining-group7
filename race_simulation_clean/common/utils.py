# common/utils.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from common.race import Race

def plot_race_positions(race: Race, team_colors: Dict[str, str], safety_car_periods: Optional[List[Tuple[int, int]]] = None, save_path: Optional[str] = None):
    """
    Plot driver positions over race duration with team colors and safety car periods.
    """
    plt.figure(figsize=(12, 6))

    for driver in race.drivers:
        positions = race.lap_data[driver.driver_id]['positions']
        constructor_id = driver.static_features.get('constructor_id', 'unknown')  # Ensure 'constructor_id' is parsed
        color = team_colors.get(constructor_id, '#CCCCCC')  # Default gray if team not found
        plt.plot(range(1, len(positions) + 1), positions, 
                label=driver.name, color=color, linewidth=2)

    plt.gca().invert_yaxis()  # Position 1 at the top
    plt.grid(True, alpha=0.3)
    plt.xlabel('Lap')
    plt.ylabel('Position')
    plt.title('Race Position Changes')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Shade safety car periods if provided
    if safety_car_periods:
        for start, end in safety_car_periods:
            plt.axvspan(start, end, color='yellow', alpha=0.2, label='Safety Car' if start == safety_car_periods[0][0] else "")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_lap_times(race: Race, team_colors: Dict[str, str], safety_car_periods: Optional[List[Tuple[int, int]]] = None, save_path: Optional[str] = None):
    """
    Plot lap times for all drivers with team colors and safety car periods.
    
    Args:
        race: Race object containing driver and lap data
        team_colors: Dictionary mapping constructor names to their colors
        safety_car_periods: List of tuples containing (start_lap, end_lap) for safety car deployments
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))

    for driver in race.drivers:
        lap_times = race.lap_data[driver.driver_id]['lap_times']
        color = team_colors.get(driver.constructor_id, '#CCCCCC')
        plt.plot(range(1, len(lap_times) + 1), lap_times, 
                label=driver.name, color=color, linewidth=2)

    plt.grid(True, alpha=0.3)
    plt.xlabel('Lap')
    plt.ylabel('Lap Time (ms)')
    plt.title('Driver Lap Times')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    if safety_car_periods:
        for start, end in safety_car_periods:
            plt.axvspan(start, end, color='yellow', alpha=0.2, label='Safety Car' if start == safety_car_periods[0][0] else "")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_actual_vs_predicted(race: Race, actual_results: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot actual vs predicted lap times for each driver.
    
    Args:
        race: Race object containing predicted lap times
        actual_results: DataFrame containing actual lap times with 'raceId', 'driverId', 'lap', 'milliseconds' columns
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for driver in race.drivers:
        driver_id = driver.driver_id
        predicted = race.lap_data[driver_id]['lap_times']
        actual_driver = actual_results[
            (actual_results['raceId'] == race.race_id) & 
            (actual_results['driverId'] == driver_id)
        ].sort_values('lap')

        if len(actual_driver) == 0 or 'milliseconds' not in actual_driver.columns:
            continue

        actual_times = actual_driver['milliseconds'].values
        laps = np.arange(1, min(len(predicted), len(actual_times)) + 1)

        if len(actual_times) >= len(laps):
            ax.plot(laps, actual_times[:len(laps)], '--', label=f"{driver.name} (Actual)")
            ax.plot(laps, predicted[:len(laps)], '-', label=f"{driver.name} (Predicted)")

    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap Time (ms)")
    ax.set_title("Actual vs Predicted Lap Times")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()