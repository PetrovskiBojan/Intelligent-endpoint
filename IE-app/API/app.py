from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
from tensorflow.keras.models import load_model
from datetime import datetime
import json

app = FastAPI()

class PredictionRequest(BaseModel):
    station_name: str
    current_time: datetime
    temperatures_2m: list[float]
    precipitation_probabilities: list[float]

# Adjusted path to where the models and scaler parameters are stored
MODELS_DIR = os.path.join("..", "models")
SCALER_DIR = os.path.join("..", "data", "scaler_params")

def load_scaler_parameters(station_name: str):
    scaler_path = os.path.join(SCALER_DIR, f"{station_name}_scaler_params.json")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler parameters not found for station {station_name}")
    
    with open(scaler_path) as fp:
        scaler_parameters = json.load(fp)
    
    return np.array(scaler_parameters["min_"]), np.array(scaler_parameters["scale_"])

def normalize(data, min_, scale_):
    return (data - min_) / scale_

@app.post("/predict")
async def predict(request: PredictionRequest):
    model_path = os.path.join(MODELS_DIR, f"{request.station_name}.h5")
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found for the given station")

    min_, scale_ = load_scaler_parameters(request.station_name)
    
    model = load_model(model_path)

    predictions = []
    for i in range(7):
        temperature = np.array([request.temperatures_2m[i]])
        precipitation = np.array([request.precipitation_probabilities[i]])
        input_features = np.hstack((temperature, precipitation))
        normalized_features = normalize(input_features, min_, scale_)
        
        # Reshape for prediction
        normalized_features = normalized_features.reshape(1, 1, 2)
        
        pred = model.predict(normalized_features).flatten()[0]
        rounded_pred = int(round(pred))
        predictions.append(rounded_pred)

    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
