# models/linear_regression/linear_regression_utils.py

import pickle
from sklearn.pipeline import Pipeline
import logging

def save_pipeline(pipeline: Pipeline, path: str):
    """
    Save the pipeline to a pickle file.

    Args:
        pipeline (Pipeline): The pipeline to save.
        path (str): Path to save the pickle file.
    """
    try:
        with open(path, 'wb') as f:
            pickle.dump(pipeline, f)
        logging.info(f"Pipeline saved to {path}")
    except Exception as e:
        logging.error(f"Error saving pipeline to {path}: {e}")
        raise

def load_pipeline(path: str) -> Pipeline:
    """
    Load the pipeline from a pickle file.

    Args:
        path (str): Path to the pickle file.

    Returns:
        Pipeline: The loaded pipeline.
    """
    try:
        with open(path, 'rb') as f:
            pipeline = pickle.load(f)
        logging.info(f"Pipeline loaded from {path}")
        return pipeline
    except Exception as e:
        logging.error(f"Error loading pipeline from {path}: {e}")
        raise
