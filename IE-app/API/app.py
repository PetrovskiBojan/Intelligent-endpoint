from fastapi.testclient import TestClient
from unittest.mock import patch
import pytest
import numpy as np
from API.app import app, PredictionRequest # Adjusted import statement

client = TestClient(app)

def test_predict():
    # Mock the model loading and prediction logic
    with patch("API.app.load_model") as mock_load_model, \
         patch("API.app.load_scaler_parameters") as mock_load_scaler_parameters:
        # Mock the model to return a dummy prediction
        mock_load_model.return_value.predict.return_value = np.array([0.5])
        # Mock the scaler parameters to return dummy values
        mock_load_scaler_parameters.return_value = (np.array([0]), np.array([1]))
        
        # Prepare the request data
        request_data = {
            "station_name": "1_GOSPOSVETSKA_C.___TURNERJEVA_UL._combined",
            "current_time": "2024-03-11 13:35:00",
            "temperatures_2m": [22.3, 22.7, 13.0, 13.5, 24.0, 34.5, 35.0],
            "precipitation_probabilities": [0.1, 0.05, 0.0, 0.20, 0.05, 0.3, 0.25]
        }
        
        # Send a POST request to the /predict endpoint
        response = client.post("/predict", json=request_data)
        
        # Assert the response status code is 200 (OK)
        assert response.status_code == 200