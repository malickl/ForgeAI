# api.py
import joblib
import json
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from preprocess_input import preprocess_input

# Charger le modèle et les colonnes
MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "feature_columns.json"

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    feature_columns = json.load(f)

# --- Définition du schéma d'entrée ---
class Diploma(BaseModel):
    level: str

class Experience(BaseModel):
    title: str

class PastCourse(BaseModel):
    title: str
    numberOfStars: float

class TeacherInput(BaseModel):
    fistname: str
    lastname: str
    city: str | None = None
    description: str | None = None
    diplomas: list[Diploma]
    experiences: list[Experience]
    pastCourses: list[PastCourse]
    current_course: str
    current_course_description: str | None = ""

# --- Création de l'app ---
app = FastAPI(
    title="FORGE_AI Predictor",
    description="API de prédiction de satisfaction — sortie : prédiction uniquement",
    version="1.0.3"
)

# --- Groupe de routes /api ---
from fastapi import APIRouter

router = APIRouter(prefix="/api")

@router.post("/predict")
def predict(input_data: TeacherInput):
    """Retourne uniquement la prédiction de note."""
    df = preprocess_input([input_data.dict()])

    # Encodage identique à l’entraînement
    df_cat = pd.get_dummies(df[["highest_degree", "course_theme"]], drop_first=False, dtype=int)
    df_num = df.drop(columns=["highest_degree", "course_theme", "prof_id", "course_title"], errors="ignore")
    X = pd.concat([df_num, df_cat], axis=1)

    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]

    pred = model.predict(X)[0]

    return round(float(pred), 3)

# Enregistrer le routeur
app.include_router(router)

@app.get("/")
def home():
    return {"message": "Bienvenue sur l’API FORGE_AI 🚀 — endpoint principal : /api/predict"}