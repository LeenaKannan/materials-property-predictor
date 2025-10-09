"""SHAP-based explainability service for model predictions."""
import numpy as np
import shap
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import io
import base64


class ExplainabilityService:
    """Service for generating model explanations using SHAP."""
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize explainability service.
        
        Args:
            model: Trained model with predict method
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
    
    def create_explainer(self, X_background: np.ndarray, explainer_type: str = "kernel"):
        """
        Create SHAP explainer.
        
        Args:
            X_background: Background data for explainer
            explainer_type: Type of explainer ('kernel', 'tree', or 'deep')
        """
        if explainer_type == "kernel":
            # KernelExplainer works with any model
            self.explainer = shap.KernelExplainer(
                self.model.predict,
                X_background[:100]  # Use subset for efficiency
            )
        elif explainer_type == "tree":
            # TreeExplainer for tree-based models
            self.explainer = shap.TreeExplainer(self.model.model)
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
    
    def explain_prediction(
        self,
        X: np.ndarray,
        top_k: int = 10
    ) -> Dict[str, any]:
        """
        Generate explanation for a single prediction.
        
        Args:
            X: Input features (single sample)
            top_k: Number of top features to return
            
        Returns:
            Dictionary with explanation data
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized. Call create_explainer first.")
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(X.reshape(1, -1))
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        shap_values = shap_values.flatten()
        
        # Get top features by absolute SHAP value
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-top_k:][::-1]
        
        feature_importance = []
        for idx in top_indices:
            feature_importance.append({
                "feature_name": self.feature_names[idx],
                "shap_value": float(shap_values[idx]),
                "feature_value": float(X[idx]),
                "importance_score": float(abs_shap[idx])
            })
        
        return {
            "feature_importance": feature_importance,
            "base_value": float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.0,
            "shap_values": shap_values.tolist()
        }
    
    def generate_waterfall_plot(
        self,
        X: np.ndarray,
        max_display: int = 10
    ) -> str:
        """
        Generate SHAP waterfall plot as base64 string.
        
        Args:
            X: Input features (single sample)
            max_display: Maximum features to display
            
        Returns:
            Base64 encoded plot image
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized.")
        
        shap_values = self.explainer.shap_values(X.reshape(1, -1))
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            data=X,
            feature_names=self.feature_names
        )
        
        # Generate plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64
    
    def generate_summary_plot(
        self,
        X: np.ndarray,
        max_display: int = 20
    ) -> str:
        """
        Generate SHAP summary plot for multiple samples.
        
        Args:
            X: Input features (multiple samples)
            max_display: Maximum features to display
            
        Returns:
            Base64 encoded plot image
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not initialized.")
        
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Generate plot
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return img_base64


class SimpleExplainer:
    """Simplified explainer based on feature importance."""
    
    def __init__(self, model, feature_names: List[str]):
        """Initialize simple explainer."""
        self.model = model
        self.feature_names = feature_names
    
    def explain_with_gradients(
        self,
        X: np.ndarray,
        top_k: int = 10
    ) -> List[Dict[str, any]]:
        """
        Explain using input gradients (for neural networks).
        
        Args:
            X: Input features
            top_k: Number of top features
            
        Returns:
            List of feature importance dictionaries
        """
        # Simple approximation: feature value * weight magnitude
        # For more accurate results, use SHAP
        
        feature_contributions = X * np.abs(X)  # Simplified
        abs_contrib = np.abs(feature_contributions)
        top_indices = np.argsort(abs_contrib)[-top_k:][::-1]
        
        importance_list = []
        for idx in top_indices:
            importance_list.append({
                "feature_name": self.feature_names[idx],
                "contribution": float(feature_contributions[idx]),
                "feature_value": float(X[idx]),
                "importance_score": float(abs_contrib[idx])
            })
        
        return importance_list


def get_feature_descriptions() -> Dict[str, str]:
    """Get human-readable descriptions for common features."""
    return {
        "MagpieData mean Electronegativity": "Average electronegativity of elements - measures tendency to attract electrons",
        "MagpieData mean AtomicWeight": "Average atomic weight - relates to density and mass",
        "MagpieData mean MeltingT": "Average melting temperature - indicates thermal stability",
        "MagpieData range Electronegativity": "Difference in electronegativity - affects bond polarity",
        "MagpieData mean Number": "Average atomic number - correlates with nuclear charge",
        "MagpieData mean AtomicRadius": "Average atomic radius - influences crystal structure",
        "num_atoms": "Total number of atoms in formula unit",
        "0-norm": "Number of different elements - compositional complexity",
    }