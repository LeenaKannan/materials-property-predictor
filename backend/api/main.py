"""FastAPI application for Materials Property Predictor."""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import sys
import os

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
model_registry = None

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


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global model_registry
    print("Starting up Materials Property Predictor API...")
    print(f"Configuration loaded: {settings.model_path}")
    # Model registry will be initialized when models are trained
    # For now, we'll handle this gracefully


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
    models_loaded = []
    if model_registry is not None:
        models_loaded = list(model_registry.services.keys())
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=models_loaded,
        timestamp=datetime.now().isoformat()
    )


@app.get("/api/v1/models/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about available models."""
    available_properties = settings.supported_properties
    
    return ModelInfoResponse(
        available_properties=available_properties,
        model_versions={"band_gap": "1.0.0"},
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
    if model_registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please train models first."
        )
    
    try:
        # For single property prediction
        if len(request.properties) == 1:
            property_name = request.properties[0]
            
            if property_name not in model_registry.services:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Property '{property_name}' not available"
                )
            
            service = model_registry.services[property_name]
            result = service.predict(
                request.formula,
                request.include_uncertainty,
                request.include_explanation
            )
            
            if not result.get("success"):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=result.get("error", "Prediction failed")
                )
            
            return result
        
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
    if model_registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please train models first."
        )
    
    try:
        property_name = request.properties[0] if request.properties else "band_gap"
        
        if property_name not in model_registry.services:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Property '{property_name}' not available"
            )
        
        service = model_registry.services[property_name]
        results = service.predict_batch(
            request.formulas,
            request.include_uncertainty,
            request.include_explanation
        )
        
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