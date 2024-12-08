# common/evaluation.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from typing import Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def evaluate_model(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def plot_predictions(y_true, y_pred, model_name: str):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Lap Time')
    plt.ylabel('Predicted Lap Time')
    plt.title(f'{model_name} - Actual vs. Predicted Lap Times')
    plt.show()

def evaluate_race_simulation(race, actual_results):
    if not race.drivers or all(len(race.lap_data[d.driver_id]['positions']) == 0 for d in race.drivers):
        logging.warning("No laps were completed in this race. Cannot evaluate positions.")
        return {'metrics': {}, 'detailed_results': pd.DataFrame()}
    # Extract final positions from simulation
    sim_positions = []
    for driver in race.drivers:
        final_pos = race.lap_data[driver.driver_id]['positions'][-1]
        sim_positions.append({
            'driverId': driver.driver_id,
            'predicted_position': final_pos,
            'start_position': driver.start_position
        })
    
    sim_df = pd.DataFrame(sim_positions)
    
    # Get actual results for this race
    race_results = actual_results[
        actual_results['raceId'] == race.race_id
    ][['driverId', 'position', 'grid']].copy()
    
    # Merge predicted and actual results
    comparison = sim_df.merge(
        race_results,
        on='driverId',
        suffixes=('_pred', '_actual')
    )
    
    # Calculate metrics
    metrics = {
        'rmse': np.sqrt(mean_squared_error(comparison['position'], comparison['predicted_position'])),
        'mae': mean_absolute_error(comparison['position'], comparison['predicted_position']),
        'positions_correct': (comparison['position'] == comparison['predicted_position']).mean(),
        'position_changes_accuracy': np.corrcoef(
            comparison['position'] - comparison['grid'],
            comparison['predicted_position'] - comparison['start_position']
        )[0, 1],
        'num_drivers': len(comparison)
    }
    
    return {
        'metrics': metrics,
        'detailed_results': comparison
    }

def evaluate_direct_prediction(predictions: pd.Series, actual_results: pd.DataFrame) -> Dict:
    """
    Evaluate a direct position prediction model.
    
    Args:
        predictions: Series of predicted final positions
        actual_results: DataFrame containing actual results
        
    Returns:
        Dictionary containing evaluation metrics and detailed comparison
    """
    comparison = pd.DataFrame({
        'predicted_position': predictions,
        'position': actual_results['position'],
        'grid': actual_results['grid']
    })
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(comparison['position'], comparison['predicted_position'])),
        'mae': mean_absolute_error(comparison['position'], comparison['predicted_position']),
        'positions_correct': (comparison['position'] == comparison['predicted_position']).mean(),
        'position_changes_accuracy': np.corrcoef(
            comparison['position'] - comparison['grid'],
            comparison['predicted_position'] - comparison['grid']
        )[0, 1],
        'num_drivers': len(comparison)
    }
    
    return {
        'metrics': metrics,
        'detailed_results': comparison
    }

def plot_actual_vs_predicted(race, test_df):
    fig, ax = plt.subplots(figsize=(10,6))
    for d in race.drivers:
        driver_id = d.driver_id
        predicted = race.lap_data[driver_id]['predicted_lap_times']
        actual_driver = test_df[(test_df['raceId'] == race.race_id) & (test_df['driverId'] == driver_id)].sort_values('lap')

        if 'milliseconds' not in actual_driver.columns:
            logging.warning("Actual data does not have 'milliseconds' column.")
            continue

        actual_times = actual_driver['milliseconds'].values
        laps = np.arange(1, len(predicted)+1)

        if len(actual_times) == len(predicted):
            ax.plot(laps, actual_times, label=f"Driver {driver_id} Actual", linestyle='--')
        else:
            logging.warning(f"Driver {driver_id}: actual vs predicted length mismatch.")
        
        ax.plot(laps, predicted, label=f"Driver {driver_id} Predicted")

    ax.set_xlabel("Lap")
    ax.set_ylabel("Lap Time (ms)")
    ax.set_title("Actual vs. Predicted Lap Times")
    ax.legend()
    plt.tight_layout()
    plt.show()


def compare_models(sim_results: Dict, direct_results: Dict, race_id: int) -> pd.DataFrame:
    """
    Compare metrics between simulation and direct prediction models.
    
    Args:
        sim_results: Results dictionary from race simulation evaluation
        direct_results: Results dictionary from direct prediction evaluation
        race_id: ID of the race being compared
        
    Returns:
        DataFrame containing comparison metrics
    """
    comparison = pd.DataFrame({
        'Race ID': [race_id, race_id],
        'Model': ['Race Simulation', 'Direct Prediction'],
        'RMSE': [sim_results['metrics']['rmse'], direct_results['metrics']['rmse']],
        'MAE': [sim_results['metrics']['mae'], direct_results['metrics']['mae']],
        'Positions Correct (%)': [
            sim_results['metrics']['positions_correct'] * 100,
            direct_results['metrics']['positions_correct'] * 100
        ],
        'Position Changes Correlation': [
            sim_results['metrics']['position_changes_accuracy'],
            direct_results['metrics']['position_changes_accuracy']
        ],
        'Drivers': [
            sim_results['metrics']['num_drivers'],
            direct_results['metrics']['num_drivers']
        ]
    })
    
    return comparison