from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np
from typing import Dict
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np

class F1PolyPredictor:
    def __init__(self, processed_data: Dict):
        self.processed_data = processed_data
        self.static_features = processed_data['feature_info']['static_features']
        self.dynamic_features = processed_data['feature_info']['dynamic_features']
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.calculate_baseline_metrics()
    
    def calculate_baseline_metrics(self):
        """Calculate baseline metrics using both global mean and practice session means."""
        target_scaler = self.processed_data['scalers']['target_scaler']
        test_targets = self.processed_data['test']['targets']
        
        # 1. Global mean baseline (current)
        train_mean = np.mean(self.processed_data['train']['targets'])
        scaled_predictions_global = np.full_like(test_targets, train_mean)
        
        # 2. Practice session mean baseline
        practice_cols = ['fp1_median_time', 'fp2_median_time', 'fp3_median_time']
        practice_means = np.mean([
            self.processed_data['test']['features'][:, 
                self.static_features.index(col)] for col in practice_cols
        ], axis=0)
        
        # Transform predictions and targets back to original scale
        predictions_global = target_scaler.inverse_transform(scaled_predictions_global.reshape(-1, 1)).ravel()
        predictions_practice = target_scaler.inverse_transform(practice_means.reshape(-1, 1)).ravel()
        original_targets = target_scaler.inverse_transform(test_targets.reshape(-1, 1)).ravel()
        
        self.baseline_metrics = {
            'global_mean_rmse': np.sqrt(mean_squared_error(original_targets, predictions_global)),
            'global_mean_mae': mean_absolute_error(original_targets, predictions_global),
            'practice_mean_rmse': np.sqrt(mean_squared_error(original_targets, predictions_practice)),
            'practice_mean_mae': mean_absolute_error(original_targets, predictions_practice)
        }
        
        self.logger.info(f"Global Mean Baseline RMSE: {self.baseline_metrics['global_mean_rmse']:.2f} ms")
        self.logger.info(f"Practice Mean Baseline RMSE: {self.baseline_metrics['practice_mean_rmse']:.2f} ms")
    
    def optimize(self, n_trials: int = 100) -> optuna.Study:
        """Optimize model hyperparameters using Optuna."""
        def objective(trial: optuna.Trial) -> float:
            try:
                # Let Optuna choose the model type
                model_type = trial.suggest_categorical('model_type', 
                    ['linear', 'ridge', 'lasso', 'elasticnet'])
                
                # Model-specific parameters
                params = {
                    'degree': trial.suggest_int('degree', 1, 2),
                    'include_bias': trial.suggest_categorical('include_bias', [True, False]),
                    'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True) if model_type != 'linear' else 0.0,
                    'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0) if model_type == 'elasticnet' else 0.0
                }
                
                # Choose the appropriate model
                if model_type == 'linear':
                    regressor = LinearRegression(fit_intercept=True)
                elif model_type == 'ridge':
                    regressor = Ridge(alpha=params['alpha'], fit_intercept=True)
                elif model_type == 'lasso':
                    regressor = Lasso(alpha=params['alpha'], fit_intercept=True)
                else:
                    regressor = ElasticNet(alpha=params['alpha'], l1_ratio=params['l1_ratio'], fit_intercept=True)
                
                # Create model pipeline
                model = make_pipeline(
                    StandardScaler(),
                    PolynomialFeatures(
                        degree=params['degree'],
                        include_bias=params['include_bias']
                    ),
                    regressor
                )
                
                # Train and evaluate with scaled targets
                model.fit(self.processed_data['train']['features'], 
                        self.processed_data['train']['targets'])
                scaled_predictions = model.predict(self.processed_data['test']['features'])
                
                # Transform predictions back to original scale
                target_scaler = self.processed_data['scalers']['target_scaler']
                predictions = target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).ravel()
                original_targets = target_scaler.inverse_transform(
                    self.processed_data['test']['targets'].reshape(-1, 1)).ravel()
                
                rmse = np.sqrt(mean_squared_error(original_targets, predictions))
                
                self.logger.info(f"Trial RMSE: {rmse:.2f} ms (Baseline: {self.baseline_metrics['practice_mean_rmse']:.2f} ms)")
                return rmse
                
            except Exception as e:
                self.logger.error(f"Trial failed: {str(e)}")
                return float('inf')
            
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        return study
    
    def train(self, study: optuna.Study):
        """Train the final model using the best hyperparameters."""
        best_params = study.best_params
        model_type = best_params['model_type']
        
        # Create regressor based on best model type
        if model_type == 'linear':
            regressor = LinearRegression(fit_intercept=True)
        elif model_type == 'ridge':
            regressor = Ridge(alpha=best_params['alpha'], fit_intercept=True)
        elif model_type == 'lasso':
            regressor = Lasso(alpha=best_params['alpha'], fit_intercept=True)
        else:
            regressor = ElasticNet(alpha=best_params['alpha'], 
                                l1_ratio=best_params['l1_ratio'], 
                                fit_intercept=True)
        
        # Create final model with best parameters
        self.model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(
                degree=best_params['degree'],
                include_bias=best_params['include_bias']
            ),
            regressor
        )
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model's performance on test data."""
        if self.model is None:
            raise ValueError("Model needs to be trained before evaluation")
        
        test_features = self.processed_data['test']['features']
        test_targets = self.processed_data['test']['targets']
        
        predictions = self.model.predict(test_features)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(test_targets, predictions)),
            'mae': mean_absolute_error(test_targets, predictions),
            'baseline_rmse': self.baseline_metrics['rmse'],
            'baseline_mae': self.baseline_metrics['mae']
        }
        
        # Compare with baseline
        self.logger.info(f"Model RMSE: {metrics['rmse']:.2f} (Baseline: {metrics['baseline_rmse']:.2f})")
        self.logger.info(f"Model MAE: {metrics['mae']:.2f} (Baseline: {metrics['baseline_mae']:.2f})")
        
        return metrics
    
    def predict_lap_time(self, features: np.ndarray) -> float:
        """Predict lap time for a given set of features."""
        if self.model is None:
            raise ValueError("Model needs to be trained before prediction")
        
        features = features.reshape(1, -1)
        return self.model.predict(features)[0]
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No trained model to save")
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)