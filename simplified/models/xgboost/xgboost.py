import xgboost as xgb
import optuna
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Tuple
import pandas as pd
import logging

class F1XGBoostPredictor:
    """XGBoost-based lap time prediction model with hyperparameter optimization."""
    
    def __init__(self, processed_data: Dict):
        self.processed_data = processed_data
        self.static_features = processed_data['feature_info']['static_features']
        self.dynamic_features = processed_data['feature_info']['dynamic_features']
        self.model = None
        self.logger = logging.getLogger(__name__)
        
    def optimize(self, n_trials: int = 100) -> optuna.Study:
        """Optimize XGBoost hyperparameters using Optuna."""
        def objective(trial: optuna.Trial) -> float:
            params = {
                'objective': 'reg:squarederror',
                'tree_method': 'hist',
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
            }
            
            # Create model with current parameters
            model = xgb.XGBRegressor(**params, random_state=42)
            
            # Train the model
            train_features = self.processed_data['train']['features']
            train_targets = self.processed_data['train']['targets']
            val_features = self.processed_data['test']['features']
            val_targets = self.processed_data['test']['targets']
            
            # Train with validation set
            eval_set = [(val_features, val_targets)]
            model.fit(
                train_features, 
                train_targets,
                eval_set=eval_set,
                verbose=False
            )
            
            # Evaluate
            predictions = model.predict(val_features)
            rmse = np.sqrt(mean_squared_error(val_targets, predictions))
            
            return rmse
        
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=3
            )
        )
        
        study.optimize(objective, n_trials=n_trials)
        return study
    
    def train(self, study: optuna.Study):
        """Train the final model using the best hyperparameters."""
        best_params = study.best_params
        best_params['objective'] = 'reg:squarederror'
        best_params['tree_method'] = 'hist'
        
        self.model = xgb.XGBRegressor(**best_params, random_state=42)
        
        # Train on all training data
        self.model.fit(
            self.processed_data['train']['features'],
            self.processed_data['train']['targets']
        )
        
        # Calculate and log feature importance
        importance = self.model.feature_importances_
        feature_names = self.static_features + self.dynamic_features
        
        importance_dict = dict(zip(feature_names, importance))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        self.logger.info("Top 10 most important features:")
        for feat, imp in sorted_importance[:10]:
            self.logger.info(f"{feat}: {imp:.4f}")
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model's performance on test data."""
        if self.model is None:
            raise ValueError("Model needs to be trained before evaluation")
        
        test_features = self.processed_data['test']['features']
        test_targets = self.processed_data['test']['targets']
        
        # Get predictions in scaled form
        scaled_predictions = self.model.predict(test_features)
        
        # Convert both predictions and targets back to original scale
        target_scaler = self.processed_data['scalers']['target_scaler']
        predictions = target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).ravel()
        original_targets = target_scaler.inverse_transform(test_targets.reshape(-1, 1)).ravel()
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(original_targets, predictions)),
            'mae': mean_absolute_error(original_targets, predictions)
        }
        
        return metrics
    
    def predict_lap_time(self, features: np.ndarray) -> float:
        """Predict lap time for a given set of features."""
        if self.model is None:
            raise ValueError("Model needs to be trained before prediction")
        
        features = features.reshape(1, -1)
        scaled_prediction = self.model.predict(features)[0]
        
        # Convert prediction back to original scale
        target_scaler = self.processed_data['scalers']['target_scaler']
        prediction = target_scaler.inverse_transform([[scaled_prediction]])[0][0]
        
        return prediction
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No trained model to save")
        self.model.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        self.model = xgb.XGBRegressor()
        self.model.load_model(filepath)

    @classmethod
    def try_feature_subsets(cls, processed_data: Dict, max_combinations: int = 100):
        """Test different feature combinations for XGBoost."""
        from itertools import combinations
        
        static_features = processed_data['feature_info']['static_features']
        dynamic_features = processed_data['feature_info']['dynamic_features']
        all_features = static_features + dynamic_features
        
        # Define feature groups for smarter combinations
        driver_features = [f for f in all_features if 'driver' in f]
        car_features = [f for f in all_features if 'constructor' in f or 'tire' in f or 'fuel' in f]
        track_features = [f for f in all_features if 'circuit' in f or 'Track' in f]
        position_features = [f for f in all_features if 'position' in f or 'Gap' in f or 'Interval' in f]
        
        results = []
        
        # Try different combinations of feature groups
        feature_groups = [driver_features, car_features, track_features, position_features]
        for r in range(1, len(feature_groups) + 1):
            for group_combo in combinations(feature_groups, r):
                selected_features = list(set([f for group in group_combo for f in group]))
                
                # Create subset of data
                subset_data = {
                    'train': {
                        'features': processed_data['train']['features'][:, [all_features.index(f) for f in selected_features]],
                        'targets': processed_data['train']['targets'],
                        'metadata': processed_data['train']['metadata']
                    },
                    'test': {
                        'features': processed_data['test']['features'][:, [all_features.index(f) for f in selected_features]],
                        'targets': processed_data['test']['targets'],
                        'metadata': processed_data['test']['metadata']
                    },
                    'feature_info': {
                        'static_features': [f for f in selected_features if f in static_features],
                        'dynamic_features': [f for f in selected_features if f in dynamic_features]
                    }
                }
                
                # Train and evaluate model
                predictor = cls(subset_data)
                study = predictor.optimize(n_trials=50)  # Fewer trials for exploration
                # Update model evaluation to use scaled metrics
                metrics = predictor.evaluate()  # The evaluate method now handles scaling internally
                
                results.append({
                    'features': selected_features,
                    'rmse': metrics['rmse'],  # This is now in original milliseconds scale
                    'n_features': len(selected_features),
                    'feature_groups': [str(g) for g in group_combo],
                    'best_params': study.best_params
                })
                
        results_df = pd.DataFrame(results)
        return results_df.sort_values('rmse')