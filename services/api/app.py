from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

MODEL_SERVICE_URL = "http://localhost:8001/predict"

# ✅ Define request schema properly
class PredictionRequest(BaseModel):
    features: list

@app.get("/")
def health():
    return {"status": "api service running"}

@app.post("/predict")
def predict(request: PredictionRequest):
    response = requests.post(
        MODEL_SERVICE_URL,
        json={"features": request.features}
    )

    return {
        "model_response": response.json()
    }
