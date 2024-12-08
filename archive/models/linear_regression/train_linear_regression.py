# models/linear_regression/train_linear_regression.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import pickle
import logging

from common.data_preparation import load_and_preprocess_data

def main():
    logging.info("Loading and preprocessing data...")
    df = load_and_preprocess_data()

    X = df.drop(columns=[
        "cumulative_milliseconds", "positionOrder", "date", "driverRef", "number", 
        "date_race", "time_race", "time", "forename", "surname", "dob", "url_race", 
        "location", "circuitRef", "milliseconds"
    ])
    y = df["milliseconds"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_features = ["code", "nationality", "status", "circuit_type", "country"]
    numerical_features = X.columns.drop(categorical_features)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=100)),
        ('regressor', LinearRegression())
    ])

    pipeline.fit(X_train, y_train)

    model_path = "models/linear_regression/linear_regression_pipeline.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)

    logging.info(f"Linear Regression pipeline saved to {model_path}")

if __name__ == "__main__":
    main()
