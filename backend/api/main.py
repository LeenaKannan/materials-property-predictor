"""FastAPI application for Materials Property Predictor."""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import sys
import os
from pathlib import Path
import torch
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api.models import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    MultiPropertyResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse
)
from backend.config import settings

# Global registry - will be initialized on startup
model_registry = {}

app = FastAPI(
    title="Materials Property Predictor API",
    description="API for predicting material properties using machine learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_models():
    """Load all available models from the models directory."""
    models_dir = Path(settings.model_path)
    loaded_models = {}
    
    if not models_dir.exists():
        print(f"⚠️  Models directory not found: {models_dir}")
        return loaded_models
    
    # Find all .pt model files
    model_files = list(models_dir.glob("ann_*.pt"))
    
    if not model_files:
        print(f"⚠️  No model files found in {models_dir}")
        return loaded_models
    
    print(f"Found {len(model_files)} model files")
    
    for model_file in model_files:
        try:
            # Extract property name from filename
            # ann_band_gap.pt -> band_gap
            property_name = model_file.stem.replace("ann_", "")
            
            # Load the model
            model = torch.load(model_file, map_location=torch.device('cpu'))
            
            # Load the corresponding scaler
            scaler_file = models_dir / f"scaler_{property_name}.pkl"
            scaler = None
            if scaler_file.exists():
                with open(scaler_file, 'rb') as f:
                    scaler = pickle.load(f)
            
            loaded_models[property_name] = {
                'model': model,
                'scaler': scaler,
                'path': str(model_file)
            }
            
            print(f"✓ Loaded model for: {property_name}")
            
        except Exception as e:
            print(f"✗ Failed to load {model_file.name}: {e}")
    
    return loaded_models


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global model_registry
    print("Starting up Materials Property Predictor API...")
    print(f"Configuration loaded: {settings.model_path}")
    
    # Load models
    model_registry = load_models()
    
    if model_registry:
        print(f"✓ Successfully loaded {len(model_registry)} models")
        print(f"  Available properties: {', '.join(model_registry.keys())}")
    else:
        print("⚠️  No models loaded - predictions will not be available")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Materials Property Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    models_loaded = list(model_registry.keys()) if model_registry else []
    
    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        version="1.0.0",
        models_loaded=models_loaded,
        timestamp=datetime.now().isoformat()
    )


@app.get("/api/v1/models/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about available models."""
    available_properties = list(model_registry.keys()) if model_registry else []
    
    model_versions = {prop: "1.0.0" for prop in available_properties}
    
    return ModelInfoResponse(
        available_properties=available_properties,
        model_versions=model_versions,
        feature_count=132,  # Typical matminer feature count
        description="ANN-based material property predictor using composition features"
    )


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict material properties from chemical formula.
    
    Args:
        request: Prediction request with formula and options
        
    Returns:
        Prediction results with uncertainty and feature importance
    """
    if not model_registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please ensure models are in the models/ directory."
        )
    
    try:
        # For single property prediction
        if len(request.properties) == 1:
            property_name = request.properties[0]
            
            if property_name not in model_registry:
                available = ', '.join(model_registry.keys())
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Property '{property_name}' not available. Available: {available}"
                )
            
            # Get model and scaler
            model_data = model_registry[property_name]
            model = model_data['model']
            scaler = model_data['scaler']
            
            # TODO: Implement actual prediction logic here
            # This is a placeholder response
            return PredictionResponse(
                success=True,
                formula=request.formula,
                normalized_formula=request.formula,
                prediction=None,
                composition=None,
                processing_time=0.1,
                error=None
            )
        
        # For multiple properties
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Multiple properties not yet supported. Use single property prediction."
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )


@app.post("/api/v1/batch-predict")
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict properties for multiple formulas.
    
    Args:
        request: Batch prediction request
        
    Returns:
        List of prediction results
    """
    if not model_registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please ensure models are in the models/ directory."
        )
    
    try:
        property_name = request.properties[0] if request.properties else "band_gap"
        
        if property_name not in model_registry:
            available = ', '.join(model_registry.keys())
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Property '{property_name}' not available. Available: {available}"
            )
        
        # TODO: Implement batch prediction logic
        results = []
        for formula in request.formulas:
            results.append({
                "success": True,
                "formula": formula,
                "prediction": None
            })
        
        return {"results": results}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error: {str(e)}"
        )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )