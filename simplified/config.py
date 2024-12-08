import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env
load_dotenv()

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent  # Root of the project
DATA_DIR = BASE_DIR / os.getenv('DATA_DIR', 'data')
MODEL_DIR = BASE_DIR / os.getenv('MODEL_DIR', 'models')
RESULTS_DIR = BASE_DIR / os.getenv('RESULTS_DIR', 'notebooks/results')

# Data Subdirectories
TRAIN_DATA_DIR = DATA_DIR / 'train'
TEST_DATA_DIR = DATA_DIR / 'test'
UTIL_DATA_DIR = DATA_DIR / 'util'

# Specific Files
CIRCUIT_ATTRS = UTIL_DATA_DIR / 'circuit_attributes.csv'
DRIVER_ATTRS = UTIL_DATA_DIR / 'drivers_attributes.csv'
RACE_ATTRS = UTIL_DATA_DIR / 'race_attributes.csv'
WEATHER_ATTRS = UTIL_DATA_DIR / 'race_weather_attributes.csv'
LAPS_FILE = DATA_DIR / 'LAPS.csv'
SPECIAL_LAPS_FILE = DATA_DIR / 'SPECIAL_LAPS.csv'

# Model Subdirectories
LSTM_DIR = MODEL_DIR / 'lstm'
LINEAR_REGRESSION_DIR = MODEL_DIR / 'linear_regression'
XGBOOST_DIR = MODEL_DIR / 'xgboost'

# Notebooks
LSTM_NOTEBOOK_DIR = BASE_DIR / 'notebooks/lstm'
LINEAR_NOTEBOOK_DIR = BASE_DIR / 'notebooks/linear_regression'
XGBOOST_NOTEBOOK_DIR = BASE_DIR / 'notebooks/xgboost'

# Output/Results
PLOTS_DIR = BASE_DIR / 'notebooks/plots'

# Example of dynamic configuration
PIT_STOP_DURATION = int(os.getenv('PIT_STOP_DURATION', 20000))
