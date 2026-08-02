"""
test_api.py
Pruebas automatizadas de la API FastAPI de predicción de diabetes.
Se ejecutan en el pipeline CI/CD y localmente con: pytest test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    """Verifica que el endpoint /health responde correctamente."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["modelo"] == "RandomForest"
    assert data["umbral"] == 0.2


def test_home():
    """Verifica que el frontend carga correctamente."""
    response = client.get("/")
    assert response.status_code == 200


def test_predict_alto_riesgo():
    """Caso de prueba: paciente con alto riesgo de diabetes."""
    payload = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediccion" in data
    assert "probabilidad" in data
    assert "riesgo" in data
    assert data["prediccion"] == 1
    assert data["riesgo"] == "Alto"
    assert 0 <= data["probabilidad"] <= 1


def test_predict_bajo_riesgo():
    """Caso de prueba: paciente con bajo riesgo de diabetes."""
    payload = {
        "Pregnancies": 1,
        "Glucose": 85,
        "BloodPressure": 66,
        "SkinThickness": 29,
        "Insulin": 0,
        "BMI": 26.6,
        "DiabetesPedigreeFunction": 0.351,
        "Age": 21
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediccion"] == 0
    assert data["riesgo"] == "Bajo"


def test_predict_campos_completos():
    """Verifica que la respuesta incluye todos los campos esperados."""
    payload = {
        "Pregnancies": 3,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 30,
        "Insulin": 100,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 35
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"prediccion", "probabilidad", "riesgo", "umbral"}
    assert data["umbral"] == 0.2


def test_predict_caso_extremo_ceros():
    """Caso extremo: todos los valores en mínimo."""
    payload = {
        "Pregnancies": 0,
        "Glucose": 0,
        "BloodPressure": 0,
        "SkinThickness": 0,
        "Insulin": 0,
        "BMI": 0,
        "DiabetesPedigreeFunction": 0,
        "Age": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediccion"] in [0, 1]


def test_predict_caso_extremo_valores_altos():
    """Caso extremo: valores clínicamente muy elevados."""
    payload = {
        "Pregnancies": 17,
        "Glucose": 199,
        "BloodPressure": 122,
        "SkinThickness": 99,
        "Insulin": 846,
        "BMI": 67.1,
        "DiabetesPedigreeFunction": 2.42,
        "Age": 81
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediccion"] == 1
    assert data["riesgo"] == "Alto"
