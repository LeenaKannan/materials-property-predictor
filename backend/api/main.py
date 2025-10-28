"""FastAPI application for Materials Property Predictor."""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import pickle
import joblib

from backend.services.prediction_service import PredictionService
from backend.processors.feature_engineer import MaterialsFeatureEngineer
from pymatgen.core.composition import Composition
import time

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


# Model architectures
class ResidualBlock(nn.Module):
    """Residual block with skip connections"""
    def __init__(self, hidden_size, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class ResNetPredictor(nn.Module):
    """ResNet for complex materials properties"""
    def __init__(self, input_size, hidden_size=256, num_blocks=4, dropout=0.3):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.bn_input = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout)
            for _ in range(num_blocks)
        ])
        self.fc_reduce = nn.Linear(hidden_size, hidden_size // 2)
        self.bn_reduce = nn.BatchNorm1d(hidden_size // 2)
        self.output_layer = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.bn_input(x)
        x = self.relu(x)
        for block in self.blocks:
            x = block(x)
        x = self.fc_reduce(x)
        x = self.bn_reduce(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


class VanillaANN(nn.Module):
    """Standard feedforward network"""
    def __init__(self, input_size, hidden_layers, dropout_rates, use_batchnorm=True):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size, dropout in zip(hidden_layers, dropout_rates):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size) if use_batchnorm else nn.Identity(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

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
            
            # Load the checkpoint (contains state_dict, config, metrics)
            checkpoint = torch.load(model_file, map_location=torch.device('cpu'), weights_only=False)
            
            # Reconstruct the model based on architecture type
            if isinstance(checkpoint, dict) and 'architecture_type' in checkpoint:
                input_size = checkpoint['input_size']
                config = checkpoint['config']
                arch_type = checkpoint['architecture_type']
                
                if arch_type == 'resnet':
                    # Get ResNet-specific parameters
                    hidden_size = config.get('hidden_size', 256)
                    num_blocks = config.get('num_blocks', 4)
                    dropout = config.get('dropout', 0.3)
                    
                    model = ResNetPredictor(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_blocks=num_blocks,
                        dropout=dropout
                    )
                else:  # vanilla
                    hidden_layers = config.get('hidden_layers', [256, 128, 64])
                    dropout_rates = config.get('dropout_rates', [0.3, 0.3, 0.3])
                    
                    # Ensure dropout_rates matches hidden_layers length
                    if len(dropout_rates) < len(hidden_layers):
                        dropout_rates = dropout_rates + [dropout_rates[-1]] * (len(hidden_layers) - len(dropout_rates))
                    
                    model = VanillaANN(
                        input_size=input_size,
                        hidden_layers=hidden_layers,
                        dropout_rates=dropout_rates,
                        use_batchnorm=True
                    )
                
                # Load the state dict
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()  # Set to evaluation mode
                
                metrics = checkpoint.get('metrics', {})
            else:
                # Old format or just state_dict
                print(f"⚠️  Warning: {model_file.name} uses old format, skipping")
                continue
            
            # Load the corresponding scaler
            scaler_file = models_dir / f"scaler_{property_name}.pkl"
            scaler = None
            if scaler_file.exists():
                try:
                    # Try joblib first
                    scaler = joblib.load(scaler_file)
                except Exception as joblib_error:
                    try:
                        # Fallback to pickle
                        with open(scaler_file, 'rb') as f:
                            scaler = pickle.load(f)
                    except Exception as pickle_error:
                        print(f"  ⚠️  Could not load scaler for {property_name}")
                        print(f"     Joblib error: {joblib_error}")
                        print(f"     Pickle error: {pickle_error}")
                        scaler = None            
            
            loaded_models[property_name] = {
                'model': model,
                'scaler': scaler,
                'path': str(model_file),
                'metrics': metrics,
                'architecture': arch_type,
                'input_size': input_size
            }
            
            r2 = metrics.get('r2', 0)
            mae = metrics.get('mae', 0)
            print(f"✓ Loaded {arch_type:8s} model for {property_name:25s} (R²={r2:.3f}, MAE={mae:.3f})")
            
        except Exception as e:
            print(f"✗ Failed to load {model_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
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
        
            # Create feature engineer and service
            feature_engineer = MaterialsFeatureEngineer()
            service = PredictionService(
                model=model,
                preprocessor=scaler,
                feature_engineer=feature_engineer,
                property_name=property_name
            )

            # Make prediction
            start_time = time.time()
            result = service.predict(
                request.formula,
                include_uncertainty=request.include_uncertainty,
                include_explanation=request.include_explanation
            )
            processing_time = time.time() - start_time

            # Check success
            if not result.get("success"):
                raise HTTPException(status_code=400, detail=result.get("error"))

            # Get composition
            comp = Composition(request.formula)
            composition_dict = {str(el): float(amt) for el, amt in comp.fractional_composition.items()}

            # Return proper response
            pred_data = result["prediction"]
            return PredictionResponse(
                success=True,
                formula=request.formula,
                normalized_formula=comp.reduced_formula,
                prediction=PropertyPrediction(
                    property_name=property_name,
                    value=pred_data["value"],
                    units=pred_data.get("units", ""),
                    uncertainty=pred_data.get("uncertainty"),
                    confidence_interval=pred_data.get("confidence_interval")
                ),
                composition=composition_dict,
                processing_time=processing_time,
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