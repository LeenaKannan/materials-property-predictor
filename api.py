"""Simple FastAPI server for predictions"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.processors.composition_parser import CompositionParser
from backend.processors.feature_engineer import MaterialsFeatureEngineer
from backend.models.ann_predictor import ANNPredictor
from model_loader import load_models

app = FastAPI(title="Materials Property Predictor")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
loaded_models = {}
feature_engineer = MaterialsFeatureEngineer()


class PredictRequest(BaseModel):
    formula: str
    property: str = "band_gap"


@app.on_event("startup")
def startup():
    """Load models on startup"""
    global loaded_models
    print("\n🚀 Loading models...")
    loaded_models = load_models("./models")
    if loaded_models:
        print(f"✓ Ready! Loaded: {list(loaded_models.keys())}\n")
    else:
        print("⚠️  No models loaded - train models first!\n")


@app.get("/")
def root():
    return {
        "message": "Materials Property Predictor API",
        "models": list(loaded_models.keys()),
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if loaded_models else "no_models",
        "models": list(loaded_models.keys())
    }


@app.post("/predict")
def predict(request: PredictRequest):
    """Make a prediction"""
    # Check if model is loaded
    if request.property not in loaded_models:
        raise HTTPException(
            status_code=400,
            detail=f"Property '{request.property}' not available. Available: {list(loaded_models.keys())}"
        )
    
    # Validate formula
    try:
        composition = CompositionParser.parse_formula(request.formula)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Extract features
    try:
        features = feature_engineer.extract_features(request.formula)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")
    
    # Load and use model
    try:
        model_info = loaded_models[request.property]
        preprocessor = model_info['preprocessor']
        
        # Preprocess features
        features_scaled = preprocessor.transform(features.reshape(1, -1))
        
        # Load model
        model = ANNPredictor(input_size=features_scaled.shape[1])
        model.load(model_info['model_path'])
        
        # Predict with uncertainty
        predictions, uncertainties = model.predict_with_uncertainty(features_scaled)
        
        value = float(predictions[0])
        uncertainty = float(uncertainties[0])
        
        return {
            "success": True,
            "formula": request.formula,
            "property": request.property,
            "value": value,
            "uncertainty": uncertainty,
            "confidence_interval": [value - 1.96*uncertainty, value + 1.96*uncertainty],
            "composition": composition
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    import uvicorn
    print("\n🔬 Starting Materials Property Predictor API")
    print("📍 API will be at: http://localhost:8000")
    print("📖 Docs at: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
