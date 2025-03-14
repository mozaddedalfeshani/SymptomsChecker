from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()
model = load_model('')

class SymptomInput(BaseModel):
    symptoms: list

@app.post("/predict_disease/")
async def predict_disease(data: SymptomInput):
    symptoms_array = np.array(data.symptoms).reshape(1, -1)  # Ensure correct shape
    prediction = model.predict(symptoms_array)
    predicted_class = prediction.argmax(axis=-1)[0]  # Get the class label
    return {"disease_prediction": str(predicted_class)}

