import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_home():
    """Test health check endpoint"""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_invalid_input():
    """Test API handles bad input"""

    bad_data = {"continent": "Asia"}   # missing fields

    response = client.post("/predict", json=bad_data)

    assert response.status_code in [400, 422]

def test_model_info():
    """Test model info endpoint"""
    response = client.get("/model-info")

    assert response.status_code == 200
    assert "model_type" in response.json()


def test_predict():
    """Test prediction endpoint"""

    sample_data = {
        "continent": "Asia",
        "education_of_employee": "Master's",
        "has_job_experience": "Y",
        "requires_job_training": "N",
        "no_of_employees": 500,
        "yr_of_estab": 2010,
        "region_of_employment": "West",
        "prevailing_wage": 85000.0,
        "unit_of_wage": "Yearly",
        "full_time_position": "Y"
    }

    response = client.post("/predict", json=sample_data)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in ["Certified", "Denied"]