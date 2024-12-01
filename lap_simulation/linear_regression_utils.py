# linear_regression_utils.py

import pickle
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

def train_linear_regression_with_preprocessor(X_train, y_train, n_components, categorical_features):
    """
    Trains a Linear Regression model with preprocessing and PCA.

    Parameters:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.
        n_components (int): Number of PCA components.
        categorical_features (list): List of categorical feature names.

    Returns:
        model: Trained Linear Regression model.
        preprocessor (dict): Dictionary containing the preprocessing pipeline and PCA.
    """
    # Define the preprocessing for categorical features
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

    # Define the ColumnTransformer
    preprocessor_pipeline = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Keep other columns unchanged
    )

    # Create a full preprocessing and PCA pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_pipeline),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components))
    ])

    # Fit the pipeline on training data
    X_pca = full_pipeline.fit_transform(X_train)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_pca, y_train)

    # Fit a scaler for lap times
    lap_time_scaler = StandardScaler()
    lap_time_scaler.fit(y_train.values.reshape(-1, 1))

    # Save the preprocessor pipeline and other components
    preprocessor = {
        "preprocessing_pipeline": full_pipeline,
        "lap_time_scaler": lap_time_scaler,
        "feature_names": full_pipeline.named_steps['preprocessor'].get_feature_names_out()
    }

    return model, preprocessor

def save_linear_regression_with_preprocessor(model, preprocessor, path):
    """
    Saves the trained Linear Regression model and preprocessor to a pickle file.

    Parameters:
        model: Trained Linear Regression model.
        preprocessor (dict): Preprocessing components.
        path (str): File path to save the model and preprocessor.
    """
    with open(path, "wb") as f:
        pickle.dump({"model": model, "preprocessor": preprocessor}, f)
