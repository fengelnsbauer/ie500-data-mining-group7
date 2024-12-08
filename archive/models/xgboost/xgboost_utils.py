import joblib
import logging

def save_pipeline(pipeline, path: str):
    joblib.dump(pipeline, path)
    logging.info(f"Pipeline saved to {path}")

def load_pipeline(path: str):
    pipeline = joblib.load(path)
    logging.info(f"Pipeline loaded from {path}")
    return pipeline
