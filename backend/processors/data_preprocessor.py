"""Data preprocessing and normalization utilities."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Optional
import joblib
import os


class DataPreprocessor:
    """Preprocessor for materials feature data."""
    
    def __init__(self, scaler_type: str = "standard"):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard' or 'robust')
        """
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.imputer = SimpleImputer(strategy="median")
        self.is_fitted = False
        self.feature_names = None
    
    def fit(self, X: np.ndarray, feature_names: Optional[list] = None):
        """
        Fit preprocessor to training data.
        
        Args:
            X: Training features array
            feature_names: Optional list of feature names
        """
        # Handle missing values
        X_imputed = self.imputer.fit_transform(X)
        
        # Fit scaler
        self.scaler.fit(X_imputed)
        
        self.is_fitted = True
        self.feature_names = feature_names
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted preprocessor.
        
        Args:
            X: Features array to transform
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        # Handle missing values
        X_imputed = self.imputer.transform(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled
    
    def fit_transform(self, X: np.ndarray, feature_names: Optional[list] = None) -> np.ndarray:
        """
        Fit preprocessor and transform data.
        
        Args:
            X: Features array
            feature_names: Optional list of feature names
            
        Returns:
            Transformed features
        """
        self.fit(X, feature_names)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled features back to original scale.
        
        Args:
            X: Scaled features
            
        Returns:
            Features in original scale
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform")
        
        return self.scaler.inverse_transform(X)
    
    def save(self, filepath: str):
        """Save preprocessor to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "imputer": self.imputer,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> "DataPreprocessor":
        """Load preprocessor from file."""
        data = joblib.load(filepath)
        
        preprocessor = cls()
        preprocessor.scaler = data["scaler"]
        preprocessor.imputer = data["imputer"]
        preprocessor.feature_names = data["feature_names"]
        preprocessor.is_fitted = data["is_fitted"]
        
        return preprocessor


def prepare_training_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Features array
        y: Target values
        test_size: Fraction for test set
        val_size: Fraction for validation set (from remaining after test)
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate validation from training
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def remove_low_variance_features(
    X: np.ndarray,
    threshold: float = 0.01,
    feature_names: Optional[list] = None
) -> Tuple[np.ndarray, list]:
    """
    Remove features with low variance.
    
    Args:
        X: Features array
        threshold: Variance threshold
        feature_names: Optional list of feature names
        
    Returns:
        Tuple of (filtered features, remaining feature names)
    """
    variances = np.var(X, axis=0)
    keep_indices = variances > threshold
    
    X_filtered = X[:, keep_indices]
    
    if feature_names is not None:
        filtered_names = [name for i, name in enumerate(feature_names) if keep_indices[i]]
    else:
        filtered_names = list(range(X_filtered.shape[1]))
    
    return X_filtered, filtered_names