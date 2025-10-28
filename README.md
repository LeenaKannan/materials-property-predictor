# Materials Property Predictor 🔬

A full-stack web application that uses Artificial Neural Networks (ANNs) to predict material properties from chemical composition. Built with FastAPI, Streamlit, PyTorch, and Matminer.

## Features

- **🎯 Property Prediction**: Predict band gap, formation energy, and density from chemical formulas
- **📊 Uncertainty Quantification**: Monte Carlo dropout for confidence intervals
- **🔍 Explainability**: SHAP-based feature importance analysis
- **🚀 RESTful API**: FastAPI backend with comprehensive endpoints
- **💻 Web Interface**: User-friendly Streamlit frontend
- **🐳 Docker Support**: Fully containerized deployment
- **⚡ High Performance**: Redis caching and optimized feature extraction

## Architecture

```
materials-property-predictor/
├── backend/
│   ├── api/              # FastAPI routes and models
│   ├── models/           # ML model implementations
│   ├── services/         # Business logic
│   ├── processors/       # Data processing pipeline
│   └── config.py         # Configuration management
├── frontend/
│   └── app.py            # Streamlit web interface
├── tests/                # Unit and integration tests
├── models/               # Trained model artifacts
├── train_model.py        # Model training script
├── requirements.txt      # Python dependencies
├── Dockerfile            # Multi-stage Docker build
└── docker-compose.yml    # Docker Compose configuration
```

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (optional)
- Materials Project API key (optional, for real data)

### Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd materials-property-predictor
```

2. **Create virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Configure environment:**

```bash
cp .env.example .env
# Edit .env and add your Materials Project API key
```

### Training Models

Train with real Materials Project data:

```bash
python train_model.py --property band_gap --real-data
```

Available properties:

- `band_gap` - Electronic band gap (eV)
- `formation_energy` - Formation energy (eV/atom)
- `density` - Material density (g/cm³)

### Running the Application

#### Option 1: Local Development

1. **Start the API server:**

```bash
python backend/api/main.py
```

The API will be available at <http://localhost:8000>

2. **Start the web interface** (in another terminal):

```bash
streamlit run frontend/app.py
```

The web app will be available at <http://localhost:8501>

#### Option 2: Docker Compose

```bash
docker-compose up -d
```

Services:

- API: <http://localhost:8000>
- Web Interface: <http://localhost:8501>
- Redis: localhost:6379

## Usage

### Web Interface

1. Navigate to <http://localhost:8501>
2. Enter a chemical formula (e.g., "SiO2", "Fe2O3")
3. Select property to predict
4. View predictions with uncertainty and feature importance

### API Usage

**Predict single property:**

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "formula": "SiO2",
    "properties": ["band_gap"],
    "include_uncertainty": true,
    "include_explanation": true
  }'
```

**Batch predictions:**

```bash
curl -X POST "http://localhost:8000/api/v1/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{
    "formulas": ["SiO2", "Fe2O3", "CaTiO3"],
    "properties": ["band_gap"]
  }'
```

**Health check:**

```bash
curl http://localhost:8000/api/v1/health
```

### Python API

```python
from backend.services.prediction_service import PredictionService
from backend.processors.feature_engineer import MaterialsFeatureEngineer
from backend.processors.data_preprocessor import DataPreprocessor
from backend.models.ann_predictor import ANNPredictor

# Load trained model
model = ANNPredictor(input_size=132)
model.load("models/ann_band_gap.pt")

# Load preprocessor
preprocessor = DataPreprocessor.load("models/preprocessor_band_gap.pkl")

# Create feature engineer
feature_engineer = MaterialsFeatureEngineer()

# Create prediction service
service = PredictionService(
    model=model,
    preprocessor=preprocessor,
    feature_engineer=feature_engineer,
    property_name="band_gap"
)

# Make prediction
result = service.predict("SiO2", include_uncertainty=True)
print(result)
```

## Model Performance

The ANN model is benchmarked against Random Forest and XGBoost baselines:

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| ANN | Best | Best | Best |
| Random Forest | Good | Good | Good |
| XGBoost | Good | Good | Good |

*Actual metrics depend on training data quality and quantity*

## Project Structure

**Backend Components:**

- `processors/`: Data processing pipeline (parsing, feature engineering, preprocessing)
- `models/`: ML model implementations (ANN, Random Forest, XGBoost)
- `services/`: Business logic (prediction, explainability)
- `api/`: FastAPI routes and request/response models

**Frontend:**

- `frontend/app.py`: Streamlit web interface with visualizations

**Training:**

- `train_model.py`: End-to-end training pipeline

## Configuration

Key configuration options in `.env`:

```bash
# Materials Project API (optional)
MP_API_KEY=your_api_key_here

# Redis Cache
REDIS_HOST=localhost
REDIS_PORT=6379

# API Server
API_HOST=0.0.0.0
API_PORT=8000

# Model Settings
MODEL_PATH=./models
```

## Deployment

### Docker Production Deployment

1. **Build production image:**

```bash
docker build -t materials-predictor:latest --target production .
```

2. **Deploy with Docker Compose:**

```bash
docker-compose -f docker-compose.yml up -d
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- **Materials Project** for providing materials data
- **Matminer** for feature engineering utilities
- **SHAP** for model explainability
- **FastAPI** and **Streamlit** for web frameworks

## Support

For issues, questions, or contributions:

- Open an issue on GitHub

## Roadmap

- [ ] Multi-property prediction in single request
- [ ] Additional properties (hardness, conductivity)
- [ ] Crystal structure integration
- [ ] Advanced uncertainty quantification (Bayesian methods)
- [ ] Model versioning and A/B testing
- [ ] Real-time model retraining
- [ ] GraphQL API
- [ ] Mobile app

---

**Built with ❤️ for materials science research**
