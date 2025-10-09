"""Main prediction service orchestrating the prediction pipeline."""
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import hashlib
import json

from backend.processors.composition_parser import CompositionParser
from backend.processors.feature_engineer import MaterialsFeatureEngineer
from backend.processors.data_preprocessor import DataPreprocessor
from backend.services.explainability_service import ExplainabilityService, get_feature_descriptions


class PredictionService:
    """Service for orchestrating material property predictions."""
    
    def __init__(
        self,
        model,
        preprocessor: DataPreprocessor,
        feature_engineer: MaterialsFeatureEngineer,
        property_name: str = "band_gap",
        use_explainability: bool = True
    ):
        """
        Initialize prediction service.
        
        Args:
            model: Trained prediction model
            preprocessor: Data preprocessor
            feature_engineer: Feature engineering pipeline
            property_name: Name of property being predicted
            use_explainability: Whether to enable explainability
        """
        self.model = model
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.property_name = property_name
        
        # Explainability service
        self.explainability = None
        if use_explainability:
            feature_names = feature_engineer.get_feature_names()
            self.explainability = ExplainabilityService(model, feature_names)
        
        # Cache for features
        self.feature_cache = {}
    
    def _get_formula_hash(self, formula: str) -> str:
        """Generate hash for formula caching."""
        return hashlib.md5(formula.encode()).hexdigest()
    
    def _extract_features_cached(self, formula: str) -> np.ndarray:
        """Extract features with caching."""
        formula_hash = self._get_formula_hash(formula)
        
        if formula_hash in self.feature_cache:
            return self.feature_cache[formula_hash]
        
        features = self.feature_engineer.extract_features(formula)
        self.feature_cache[formula_hash] = features
        
        return features
    
    def predict(
        self,
        formula: str,
        include_uncertainty: bool = True,
        include_explanation: bool = True
    ) -> Dict:
        """
        Make prediction for a chemical formula.
        
        Args:
            formula: Chemical formula string
            include_uncertainty: Whether to include uncertainty estimates
            include_explanation: Whether to include feature importance
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        # Parse and validate formula
        try:
            parsed_composition = CompositionParser.parse_formula(formula)
        except ValueError as e:
            return {
                "error": str(e),
                "formula": formula,
                "success": False
            }
        
        # Extract features
        try:
            features = self._extract_features_cached(formula)
        except Exception as e:
            return {
                "error": f"Feature extraction failed: {str(e)}",
                "formula": formula,
                "success": False
            }
        
        # Preprocess features
        features_scaled = self.preprocessor.transform(features.reshape(1, -1))
        
        # Make prediction
        try:
            if include_uncertainty and hasattr(self.model, 'predict_with_uncertainty'):
                predictions, uncertainties = self.model.predict_with_uncertainty(features_scaled)
                prediction_value = float(predictions[0])
                uncertainty_value = float(uncertainties[0])
                
                # Calculate confidence interval (95%)
                confidence_interval = (
                    prediction_value - 1.96 * uncertainty_value,
                    prediction_value + 1.96 * uncertainty_value
                )
            else:
                predictions = self.model.predict(features_scaled)
                prediction_value = float(predictions[0])
                uncertainty_value = None
                confidence_interval = None
        except Exception as e:
            return {
                "error": f"Prediction failed: {str(e)}",
                "formula": formula,
                "success": False
            }
        
        # Generate explanation
        feature_importance = None
        if include_explanation and self.explainability is not None:
            try:
                explanation = self.explainability.explain_prediction(features_scaled[0])
                feature_importance = self._enrich_feature_importance(
                    explanation["feature_importance"]
                )
            except Exception as e:
                print(f"Warning: Explainability failed: {e}")
        
        processing_time = time.time() - start_time
        
        # Build response
        result = {
            "success": True,
            "formula": formula,
            "normalized_formula": CompositionParser.normalize_formula(formula),
            "prediction": {
                "property_name": self.property_name,
                "value": prediction_value,
                "units": self._get_units(self.property_name),
                "uncertainty": uncertainty_value,
                "confidence_interval": confidence_interval
            },
            "composition": parsed_composition,
            "processing_time": processing_time
        }
        
        if feature_importance is not None:
            result["feature_importance"] = feature_importance
        
        return result
    
    def predict_batch(
        self,
        formulas: List[str],
        include_uncertainty: bool = False,
        include_explanation: bool = False
    ) -> List[Dict]:
        """
        Make predictions for multiple formulas.
        
        Args:
            formulas: List of chemical formula strings
            include_uncertainty: Whether to include uncertainty
            include_explanation: Whether to include explanations
            
        Returns:
            List of prediction results
        """
        results = []
        for formula in formulas:
            result = self.predict(formula, include_uncertainty, include_explanation)
            results.append(result)
        
        return results
    
    def _enrich_feature_importance(
        self,
        feature_importance: List[Dict]
    ) -> List[Dict]:
        """Add descriptions to feature importance."""
        descriptions = get_feature_descriptions()
        
        for item in feature_importance:
            feature_name = item["feature_name"]
            if feature_name in descriptions:
                item["description"] = descriptions[feature_name]
            else:
                item["description"] = "Materials property feature"
        
        return feature_importance
    
    @staticmethod
    def _get_units(property_name: str) -> str:
        """Get units for property."""
        units_map = {
            "band_gap": "eV",
            "formation_energy": "eV/atom",
            "density": "g/cm³",
            "e_above_hull": "eV/atom"
        }
        return units_map.get(property_name, "")
    
    def initialize_explainer(self, X_background: np.ndarray):
        """Initialize explainability with background data."""
        if self.explainability is not None:
            try:
                self.explainability.create_explainer(X_background, explainer_type="kernel")
            except Exception as e:
                print(f"Warning: Failed to initialize explainer: {e}")


class ModelRegistry:
    """Registry for managing multiple property prediction models."""
    
    def __init__(self):
        """Initialize model registry."""
        self.services = {}
    
    def register_service(self, property_name: str, service: PredictionService):
        """Register a prediction service for a property."""
        self.services[property_name] = service
    
    def predict(
        self,
        formula: str,
        properties: Optional[List[str]] = None,
        include_uncertainty: bool = True,
        include_explanation: bool = True
    ) -> Dict:
        """
        Predict multiple properties for a formula.
        
        Args:
            formula: Chemical formula
            properties: List of properties to predict (None for all)
            include_uncertainty: Include uncertainty estimates
            include_explanation: Include feature importance
            
        Returns:
            Dictionary with predictions for all requested properties
        """
        if properties is None:
            properties = list(self.services.keys())
        
        results = {
            "formula": formula,
            "predictions": {}
        }
        
        for prop in properties:
            if prop not in self.services:
                results["predictions"][prop] = {
                    "error": f"Property '{prop}' not available"
                }
                continue
            
            service = self.services[prop]
            result = service.predict(formula, include_uncertainty, include_explanation)
            
            if result.get("success"):
                results["predictions"][prop] = result["prediction"]
                if "feature_importance" in result:
                    results["predictions"][prop]["feature_importance"] = result["feature_importance"]
            else:
                results["predictions"][prop] = {
                    "error": result.get("error", "Prediction failed")
                }
        
        return results