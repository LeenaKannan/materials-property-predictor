# Materials Property Predictor - Project Structure

## Directory Structure

```
materials-property-predictor/
├── backend/                    # Backend API and ML services
│   ├── api/                   # FastAPI routes and endpoints
│   ├── models/                # ML model classes and wrappers
│   ├── services/              # Business logic services
│   ├── processors/            # Data processing and feature engineering
│   ├── utils/                 # Utility functions and helpers
│   ├── config.py              # Configuration management
│   └── __init__.py
├── frontend/                   # Streamlit web interface
│   └── __init__.py
├── models/                     # Trained model artifacts (created at runtime)
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
└── PROJECT_STRUCTURE.md      # This file
```

## Key Components

### Backend Structure
- **api/**: FastAPI application, routes, and middleware
- **models/**: ML model implementations (ANN, baselines)
- **services/**: Core business logic (prediction, explainability)
- **processors/**: Data processing pipeline (parsing, feature engineering)
- **utils/**: Shared utilities (logging, caching, helpers)

### Configuration
- **config.py**: Centralized configuration using Pydantic settings
- **.env**: Environment-specific variables (not committed)
- **.env.example**: Template for environment variables

### Dependencies
Core dependencies include:
- FastAPI + Uvicorn for API backend
- Streamlit for web interface
- PyTorch for neural networks
- Scikit-learn for baseline models
- Matminer for materials feature engineering
- SHAP for model explainability
- Redis for caching
- Pydantic for data validation