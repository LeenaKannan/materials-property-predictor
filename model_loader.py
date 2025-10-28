
import os
import torch
import joblib
from pathlib import Path


def load_models(models_dir="./models"):
    """Load all trained models from the models directory
    """
    models_dir = Path(models_dir)
    
    if not models_dir.exists():
        print(f"⚠️  Models directory not found: {models_dir}")
        print(f"   Create it and add your .pt and .pkl files there")
        return None
    
    # Find model files
    model_files = list(models_dir.glob("ann_*.pt"))
    
    if not model_files:
        print(f"⚠️  No .pt model files found in {models_dir}")
        return None
    
    loaded_models = {}
    
    for model_file in model_files:
        # Get property name (e.g., "ann_band_gap.pt" -> "band_gap")
        property_name = model_file.stem.replace("ann_", "")
        
        # Find preprocessor
        preprocessor_file = models_dir / f"scaler_{property_name}.pkl"
        if not preprocessor_file.exists():
            preprocessor_file = models_dir / f"preprocessor_{property_name}.pkl"
        
        if not preprocessor_file.exists():
            print(f"⚠️  No preprocessor found for {property_name}")
            continue
        
        try:
            # Load preprocessor
            preprocessor = joblib.load(preprocessor_file)
            
            # Store model info
            loaded_models[property_name] = {
                'model_path': str(model_file),
                'preprocessor': preprocessor
            }
            
            print(f"✓ Loaded {property_name}")
            
        except Exception as e:
            print(f"✗ Failed to load {property_name}: {e}")
    
    return loaded_models


if __name__ == "__main__":
    models = load_models()
    if models:
        print(f"\n✓ Loaded {len(models)} model(s): {list(models.keys())}")
    else:
        print("\n✗ No models loaded")
