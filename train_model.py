"""Training script for materials property prediction models."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys
import argparse
from mp_api.client import MPRester

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.processors.feature_engineer import MaterialsFeatureEngineer
from backend.processors.data_preprocessor import DataPreprocessor, prepare_training_data
from backend.models.ann_predictor import ANNPredictor
from backend.models.baseline_models import BaselineModel, train_baseline_models, compare_models
from backend.services.prediction_service import PredictionService
from backend.config import settings


def fetch_materials_data(property_name: str = "band_gap", max_samples: int = 5000):
    """
    Fetch materials data from Materials Project.
    
    Args:
        property_name: Property to fetch
        max_samples: Maximum number of samples
        
    Returns:
        DataFrame with formulas and property values
    """
    print(f"Fetching data from Materials Project for {property_name}...")
    
    if not settings.mp_api_key:
        print("Warning: No Materials Project API key found.")
        print("Generating synthetic data for demonstration...")
        return generate_synthetic_data(property_name, max_samples)
    
    try:
        with MPRester(settings.mp_api_key) as mpr:
            # Query materials with band gap data
            docs = mpr.materials.summary.search(
                band_gap=(0.1, None),  # Materials with band gap
                num_elements=(1, 5),    # 1-5 elements
                fields=["formula_pretty", "band_gap", "formation_energy_per_atom", "density"]
            )
            
            # Convert to dataframe
            data = []
            for doc in docs[:max_samples]:
                data.append({
                    "formula": doc.formula_pretty,
                    "band_gap": doc.band_gap,
                    "formation_energy": doc.formation_energy_per_atom,
                    "density": doc.density
                })
            
            df = pd.DataFrame(data)
            print(f"Fetched {len(df)} materials from Materials Project")
            return df
    
    except Exception as e:
        print(f"Error fetching from Materials Project: {e}")
        print("Generating synthetic data instead...")
        return generate_synthetic_data(property_name, max_samples)


def generate_synthetic_data(property_name: str, n_samples: int = 1000):
    """Generate synthetic data for demonstration."""
    print(f"Generating {n_samples} synthetic samples...")
    
    # Common formulas
    base_formulas = [
        "SiO2", "TiO2", "Fe2O3", "Al2O3", "ZnO", "MgO",
        "CaTiO3", "BaTiO3", "NaCl", "CaO", "Cu2O", "FeO",
        "SrTiO3", "LaAlO3", "GaN", "InN", "AlN", "ZnS"
    ]
    
    formulas = []
    values = []
    
    # Generate variations
    for _ in range(n_samples):
        formula = np.random.choice(base_formulas)
        formulas.append(formula)
        
        # Generate pseudo-realistic values based on formula
        if property_name == "band_gap":
            # Band gaps typically 0-10 eV
            base_value = hash(formula) % 100 / 10.0
            noise = np.random.normal(0, 0.5)
            values.append(max(0.1, min(10.0, base_value + noise)))
        elif property_name == "formation_energy":
            # Formation energy typically -5 to 0 eV/atom
            base_value = -(hash(formula) % 50) / 10.0
            noise = np.random.normal(0, 0.2)
            values.append(base_value + noise)
        else:  # density
            base_value = (hash(formula) % 100) / 10.0 + 1.0
            noise = np.random.normal(0, 0.5)
            values.append(max(1.0, base_value + noise))
    
    df = pd.DataFrame({
        "formula": formulas,
        property_name: values
    })
    
    return df


def train_models(property_name: str = "band_gap", use_real_data: bool = False):
    """
    Train all models for a property.
    
    Args:
        property_name: Property to predict
        use_real_data: Whether to use real Materials Project data
    """
    print("=" * 60)
    print(f"Training models for {property_name}")
    print("=" * 60)
    
    # Fetch data
    if use_real_data:
        df = fetch_materials_data(property_name, max_samples=5000)
    else:
        df = generate_synthetic_data(property_name, n_samples=1000)
    
    # Remove duplicates and missing values
    df = df.drop_duplicates(subset=['formula'])
    df = df.dropna(subset=[property_name])
    
    print(f"Dataset size: {len(df)} samples")
    
    # Feature engineering
    print("\nExtracting features...")
    feature_engineer = MaterialsFeatureEngineer()
    
    X_list = []
    y_list = []
    valid_formulas = []
    
    for idx, row in df.iterrows():
        try:
            features = feature_engineer.extract_features(row['formula'])
            X_list.append(features)
            y_list.append(row[property_name])
            valid_formulas.append(row['formula'])
        except Exception as e:
            print(f"Skipping {row['formula']}: {e}")
            continue
    
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    print(f"Features extracted: {X.shape}")
    
    # Prepare data splits
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_training_data(
        X, y, test_size=0.15, val_size=0.15
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Preprocessing
    print("\nPreprocessing features...")
    preprocessor = DataPreprocessor(scaler_type="standard")
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Train ANN
    print("\n" + "=" * 60)
    print("Training Artificial Neural Network")
    print("=" * 60)
    
    ann_model = ANNPredictor(
        input_size=X_train_scaled.shape[1],
        hidden_layers=settings.ann_hidden_layers,
        dropout_rate=settings.ann_dropout_rate,
        learning_rate=settings.ann_learning_rate
    )
    
    ann_model.fit(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
        epochs=100,
        batch_size=32,
        early_stopping_patience=15
    )
    
    # Evaluate ANN
    y_pred_ann = ann_model.predict(X_test_scaled)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    ann_metrics = {
        "mae": mean_absolute_error(y_test, y_pred_ann),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_ann)),
        "r2": r2_score(y_test, y_pred_ann)
    }
    
    print(f"\nANN Test Performance:")
    print(f"  MAE: {ann_metrics['mae']:.4f}")
    print(f"  RMSE: {ann_metrics['rmse']:.4f}")
    print(f"  R²: {ann_metrics['r2']:.4f}")
    
    # Train baseline models
    print("\n" + "=" * 60)
    print("Training Baseline Models")
    print("=" * 60)
    
    baseline_models = train_baseline_models(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Compare all models
    print("\n" + "=" * 60)
    print("Model Comparison on Test Set")
    print("=" * 60)
    
    all_models = {"ANN": ann_model}
    all_models.update(baseline_models)
    
    results = compare_models(all_models, X_test_scaled, y_test)
    
    # Save models
    os.makedirs(settings.model_path, exist_ok=True)
    
    ann_path = os.path.join(settings.model_path, f"ann_{property_name}.pt")
    ann_model.save(ann_path)
    print(f"\nSaved ANN model to {ann_path}")
    
    preprocessor_path = os.path.join(settings.model_path, f"preprocessor_{property_name}.pkl")
    preprocessor.save(preprocessor_path)
    print(f"Saved preprocessor to {preprocessor_path}")
    
    for name, model in baseline_models.items():
        model_name = name.lower().replace(" ", "_")
        model_path = os.path.join(settings.model_path, f"{model_name}_{property_name}.pkl")
        model.save(model_path)
        print(f"Saved {name} to {model_path}")
    
    # Create prediction service
    print("\n" + "=" * 60)
    print("Creating Prediction Service")
    print("=" * 60)
    
    prediction_service = PredictionService(
        model=ann_model,
        preprocessor=preprocessor,
        feature_engineer=feature_engineer,
        property_name=property_name,
        use_explainability=False  # Will be initialized when needed
    )
    
    # Test prediction
    test_formula = "SiO2"
    print(f"\nTest prediction for {test_formula}:")
    result = prediction_service.predict(test_formula, include_uncertainty=True, include_explanation=False)
    
    if result['success']:
        pred = result['prediction']
        print(f"  Predicted {pred['property_name']}: {pred['value']:.3f} {pred['units']}")
        if pred['uncertainty']:
            print(f"  Uncertainty: ±{pred['uncertainty']:.3f}")
    else:
        print(f"  Error: {result['error']}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return prediction_service, results


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train materials property prediction models")
    parser.add_argument(
        "--property",
        type=str,
        default="band_gap",
        choices=["band_gap", "formation_energy", "density"],
        help="Property to predict"
    )
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Use real Materials Project data (requires API key)"
    )
    
    args = parser.parse_args()
    
    try:
        service, results = train_models(args.property, args.real_data)
        print("\n✅ Training successful!")
        print("\nTo start the API server:")
        print("  python backend/api/main.py")
        print("\nTo start the web interface:")
        print("  streamlit run frontend/app.py")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()