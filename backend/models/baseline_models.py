"""Baseline models for comparison with ANN."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple
import joblib
import os


class BaselineModel:
    """Base class for baseline models."""
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize baseline model.
        
        Args:
            model_type: Type of model ('random_forest' or 'xgboost')
        """
        if model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "xgboost":
            self.model = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the model."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_with_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with standard deviation (for Random Forest only).
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, std)
        """
        if self.model_type == "random_forest":
            # Get predictions from all trees
            tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            predictions = tree_predictions.mean(axis=0)
            std = tree_predictions.std(axis=0)
            return predictions, std
        else:
            # For XGBoost, just return predictions with zero std
            predictions = self.predict(X)
            return predictions, np.zeros_like(predictions)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True targets
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)
        
        return {
            "mae": mean_absolute_error(y, predictions),
            "rmse": np.sqrt(mean_squared_error(y, predictions)),
            "r2": r2_score(y, predictions)
        }
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return None
    
    def save(self, filepath: str):
        """Save model to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
    
    def load(self, filepath: str):
        """Load model from file."""
        self.model = joblib.load(filepath)


def compare_models(
    models: Dict[str, BaselineModel],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models on test data.
    
    Args:
        models: Dictionary of model name to model instance
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of model names to performance metrics
    """
    results = {}
    
    for name, model in models.items():
        metrics = model.evaluate(X_test, y_test)
        results[name] = metrics
        
        print(f"\n{name} Performance:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
    
    return results


def train_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> Dict[str, BaselineModel]:
    """
    Train both baseline models.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        
    Returns:
        Dictionary of trained models
    """
    models = {
        "Random Forest": BaselineModel("random_forest"),
        "XGBoost": BaselineModel("xgboost")
    }
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_metrics = model.evaluate(X_val, y_val)
        print(f"  Validation MAE: {val_metrics['mae']:.4f}")
        print(f"  Validation RMSE: {val_metrics['rmse']:.4f}")
        print(f"  Validation R²: {val_metrics['r2']:.4f}")
    
    return models