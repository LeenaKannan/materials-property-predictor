"""Pydantic models for API requests and responses."""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Tuple, Any
from backend.processors.composition_parser import CompositionParser


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    formula: str = Field(..., description="Chemical formula (e.g., 'SiO2', 'Fe2O3')")
    properties: Optional[List[str]] = Field(
        default=["band_gap"],
        description="List of properties to predict"
    )
    include_uncertainty: bool = Field(
        default=True,
        description="Include uncertainty estimates"
    )
    include_explanation: bool = Field(
        default=True,
        description="Include feature importance explanations"
    )
    
    @validator('formula')
    def validate_formula(cls, v):
        """Validate chemical formula."""
        is_valid, error_msg = CompositionParser.validate_formula(v)
        if not is_valid:
            raise ValueError(error_msg)
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "formula": "SiO2",
                "properties": ["band_gap"],
                "include_uncertainty": True,
                "include_explanation": True
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    formulas: List[str] = Field(..., description="List of chemical formulas")
    properties: Optional[List[str]] = Field(
        default=["band_gap"],
        description="List of properties to predict"
    )
    include_uncertainty: bool = Field(default=False)
    include_explanation: bool = Field(default=False)
    
    @validator('formulas')
    def validate_formulas(cls, v):
        """Validate all formulas."""
        if len(v) == 0:
            raise ValueError("At least one formula must be provided")
        if len(v) > 100:
            raise ValueError("Maximum 100 formulas per batch request")
        return v


class PropertyPrediction(BaseModel):
    """Model for a single property prediction."""
    property_name: str
    value: float
    units: str
    uncertainty: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None


class FeatureImportance(BaseModel):
    """Model for feature importance."""
    feature_name: str
    importance_score: float
    shap_value: Optional[float] = None
    feature_value: Optional[float] = None
    description: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    success: bool
    formula: str
    normalized_formula: Optional[str] = None
    prediction: Optional[PropertyPrediction] = None
    composition: Optional[Dict[str, float]] = None
    feature_importance: Optional[List[FeatureImportance]] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "formula": "SiO2",
                "normalized_formula": "SiO2",
                "prediction": {
                    "property_name": "band_gap",
                    "value": 8.92,
                    "units": "eV",
                    "uncertainty": 0.35,
                    "confidence_interval": [8.23, 9.61]
                },
                "composition": {"Si": 0.333, "O": 0.667},
                "processing_time": 0.123
            }
        }


class MultiPropertyResponse(BaseModel):
    """Response model for multiple property predictions."""
    formula: str
    predictions: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: List[str]
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    available_properties: List[str]
    model_versions: Dict[str, str]
    feature_count: int
    description: str


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    formula: Optional[str] = None