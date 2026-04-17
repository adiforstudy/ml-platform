from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from model import load_model

app = FastAPI()

model = load_model()

# ✅ Proper schema
class PredictionRequest(BaseModel):
    features: list

@app.get("/")
def health():
    return {"status": "model service running"}

@app.post("/predict")
def predict(request: PredictionRequest):
    data = np.array(request.features).reshape(1, -1)
    prediction = model.predict(data).tolist()

    return {"prediction": prediction}
