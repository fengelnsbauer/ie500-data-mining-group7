# xgboost_utils.py

import pickle
import logging
import pandas as pd

def load_pipeline(path: str):
    """
    Load the entire pipeline (preprocessor + model) from a pickle file.

    Args:
        path (str): Path to the pickle file.

    Returns:
        Pipeline: The loaded scikit-learn pipeline.
    """
    try:
        with open(path, 'rb') as f:
            pipeline = pickle.load(f)
        logging.info(f"Pipeline loaded from {path}")
        return pipeline
    except Exception as e:
        logging.error(f"Error loading pipeline from {path}: {e}")
        raise

def save_pipeline(pipeline, path: str):
    """
    Save the entire pipeline (preprocessor + model) to a pickle file.

    Args:
        pipeline (Pipeline): The scikit-learn pipeline to save.
        path (str): Path to save the pickle file.
    """
    try:
        with open(path, 'wb') as f:
            pickle.dump(pipeline, f)
        logging.info(f"Pipeline saved to {path}")
    except Exception as e:
        logging.error(f"Error saving pipeline to {path}: {e}")
        raise


def load_model_with_preprocessor(path: str):
    """
    Load the XGBoost model and preprocessor from a pickle file.

    Args:
        path (str): Path to the pickle file.

    Returns:
        tuple: (model, preprocessor)
    """
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = data['model']
        preprocessor = data['preprocessor']
        logging.info(f"Model and preprocessor loaded from {path}")
        return model, preprocessor
    except Exception as e:
        logging.error(f"Error loading model and preprocessor from {path}: {e}")
        raise

def save_model_with_preprocessor(model, preprocessor, path: str):
    """
    Save the model and preprocessor to a pickle file.

    Args:
        model: Trained XGBoost model.
        preprocessor: Preprocessing pipeline.
        path (str): Path to save the pickle file.
    """
    try:
        with open(path, 'wb') as f:
            pickle.dump({'model': model, 'preprocessor': preprocessor}, f)
        logging.info(f"Model and preprocessor saved to {path}")
    except Exception as e:
        logging.error(f"Error saving model and preprocessor to {path}: {e}")
        raise