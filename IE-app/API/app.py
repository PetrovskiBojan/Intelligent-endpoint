from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import numpy as np
import os
from tensorflow.keras.models import load_model 

app = FastAPI()

class PredictionRequest(BaseModel):
    station_name: str
    current_time: datetime
    temperatures_2m: list  # List of temperatures for the next 7 hours
    precipitation_probabilities: list  # List of precipitation probabilities for the next 7 hours

MODELS_DIR = "../models"  # Update this path to correctly point to your models directory

@app.post("/predict")
async def predict(request: PredictionRequest):
    model_path = os.path.join(MODELS_DIR, f"{request.station_name}.h5")
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found for the given station")

    model = load_model(model_path)

    predictions = []

    # Iterate over the next 7 hours
    for i in range(7):
        # Create a sequence for this particular hour
        sequence = np.array([[request.temperatures_2m[i], request.precipitation_probabilities[i]]])
        sequence = sequence.reshape(1, 1, -1)  # Shape: (1, 1, features)

        pred = model.predict(sequence).flatten()[0]  # Predict and take the first value
        rounded_pred = int(round(pred))
        predictions.append(rounded_pred)

    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
