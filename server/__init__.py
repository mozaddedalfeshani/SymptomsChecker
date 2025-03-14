from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib  # For saving and loading the model
import os
print("Current working directory:", os.getcwd())

app = FastAPI()

# Load the dataset (replace with your file path)
df = pd.read_csv("D:/Code/titleDefense/Database/Dataset_1.csv")


# Preprocessing
X = df.drop('diseases', axis=1)
y = df['diseases']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train the model (or load a pre-trained model)
try:
    model = joblib.load("../model/symptoms_checker_model.joblib") #Change to joblib file extension.
    print("Model loaded successfully")
except FileNotFoundError:
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    joblib.dump(model, "../model/symptoms_checker_model.joblib") #Save the model with joblib extension.
    print("Model trained and saved.")

# Input data model (Pydantic)
class Symptoms(BaseModel):
    symptoms: list[str]

@app.post("/predict")
async def predict_disease(symptoms_data: Symptoms):
    try:
        input_data = [0] * len(X.columns)
        for symptom in symptoms_data.symptoms:
            if symptom in X.columns:
                input_data[X.columns.get_loc(symptom)] = 1

        input_data = np.array([input_data])
        prediction = model.predict(input_data)
        disease_name = label_encoder.inverse_transform(prediction)[0]
        return {"disease": disease_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Running the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True) #added if main block to run the app.