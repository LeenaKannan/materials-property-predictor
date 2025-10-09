"""Integration tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api.main import app

client = TestClient(app)


class TestAPIEndpoints:
    """Test cases for API endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "models_loaded" in data
    
    def test_model_info(self):
        """Test model info endpoint."""
        response = client.get("/api/v1/models/info")
        assert response.status_code == 200
        data = response.json()
        assert "available_properties" in data
        assert "model_versions" in data
        assert "feature_count" in data
    
    def test_predict_endpoint_no_models(self):
        """Test prediction fails gracefully when no models loaded."""
        response = client.post(
            "/api/v1/predict",
            json={
                "formula": "SiO2",
                "properties": ["band_gap"],
                "include_uncertainty": True,
                "include_explanation": True
            }
        )
        assert response.status_code == 503  # Service unavailable
    
    def test_predict_invalid_formula(self):
        """Test prediction with invalid formula."""
        response = client.post(
            "/api/v1/predict",
            json={
                "formula": "InvalidFormula@#$",
                "properties": ["band_gap"]
            }
        )
        assert response.status_code == 422  # Validation error
    
    def test_predict_empty_formula(self):
        """Test prediction with empty formula."""
        response = client.post(
            "/api/v1/predict",
            json={
                "formula": "",
                "properties": ["band_gap"]
            }
        )
        assert response.status_code == 422
    
    def test_batch_predict_no_models(self):
        """Test batch prediction without models."""
        response = client.post(
            "/api/v1/batch-predict",
            json={
                "formulas": ["SiO2", "Fe2O3"],
                "properties": ["band_gap"]
            }
        )
        assert response.status_code == 503
    
    def test_batch_predict_empty_list(self):
        """Test batch prediction with empty formula list."""
        response = client.post(
            "/api/v1/batch-predict",
            json={
                "formulas": [],
                "properties": ["band_gap"]
            }
        )
        assert response.status_code == 422
    
    def test_batch_predict_too_many(self):
        """Test batch prediction with too many formulas."""
        formulas = ["SiO2"] * 101  # More than 100
        response = client.post(
            "/api/v1/batch-predict",
            json={
                "formulas": formulas,
                "properties": ["band_gap"]
            }
        )
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])