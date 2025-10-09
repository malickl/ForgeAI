from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import json

# ===== Initialisation =====
app = FastAPI(title="Satisfaction Prediction API")

# Charger le modèle une seule fois
MODEL_PATH = "models/model_best.joblib"
model = joblib.load(MODEL_PATH)

# ===== Schéma d’entrée =====
class CourseInput(BaseModel):
    course_title: str
    desc_len_words: float
    n_diplomas: int
    n_experiences: int
    teacher_mean_excl: float
    teacher_course_count: int
    has_master: int
    has_licence: int
    has_doctorat: int
    has_certif: int
    has_secondary: int
    has_teaching_exp: int
    has_industry_exp: int
    has_management_exp: int
    has_academic_exp: int

# ===== Endpoint principal =====
@app.post("/api/predict")
async def predict_rating(request: Request):
    try:
        data_json = await request.json()
        print("Received JSON:", json.dumps(data_json, indent=2))
        data = CourseInput(**data_json)
        # Convertir les données en DataFrame
        df = pd.DataFrame([data.dict()])

        # Prédiction avec le modèle
        prediction = model.predict(df)[0]

        # Retourner la note prédite
        return {"predicted_satisfaction": round(float(prediction), 2)}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# ===== Endpoint test =====
@app.get("/")
def root():
    return {"message": "Bienvenue sur l’API de prédiction de satisfaction 🎯"}