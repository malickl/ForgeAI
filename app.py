from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ===== Pydantic Models =====
class ProfessorData(BaseModel):
    professor: dict
    course: dict


# ===== Initialisation =====
app = FastAPI(title="Satisfaction Prediction API")

# Charger le modèle une seule fois
MODEL_PATH = "models/model_best.joblib"
model = joblib.load(MODEL_PATH)


# ===== Fonction d’adaptation corrigée =====
def adapt_input(body):
    prof = body.get("professor", {})
    course = body.get("course", {})

    diplomas = prof.get("diplomas", [])
    exps = prof.get("experiences", [])
    past_courses = prof.get("pastCourses", [])

    # ---- Diplômes ----
    levels = [d.get("level", "").lower() for d in diplomas]
    has_master = int(any("master" in lvl for lvl in levels))
    has_licence = int(any("licence" in lvl or "bachelor" in lvl for lvl in levels))
    has_doctorat = int(any("doctor" in lvl or "phd" in lvl for lvl in levels))
    has_certif = int(any("certif" in lvl for lvl in levels))
    has_secondary = int(any("second" in lvl for lvl in levels))

    # ---- Expériences ----
    all_exp_text = " ".join(
        f"{e.get('title', '')} {e.get('description', '')}".lower()
        for e in exps
    )
    has_teaching_exp = int(any(k in all_exp_text for k in ["prof", "enseign", "formateur"]))
    has_industry_exp = int(any(k in all_exp_text for k in ["dev", "engineer", "ingénieur", "consult", "industrie"]))
    has_management_exp = int(any(k in all_exp_text for k in ["chef", "lead", "manager", "responsable"]))
    has_academic_exp = int(any(k in all_exp_text for k in ["recherche", "thèse", "doctorat", "universit"]))

    # ---- Moyenne et nombre de cours passés ----
    past_ratings = [c.get("numberOfStars") for c in past_courses if c.get("numberOfStars") is not None]
    teacher_mean_excl = float(sum(past_ratings) / len(past_ratings)) if past_ratings else 0
    teacher_course_count = len(past_courses)

    # ---- Description du prof ----
    desc_len_words = len(prof.get("description", "").split())
    n_diplomas = len(diplomas)
    n_experiences = len(exps)

    # ---- Nouveau cours ----
    course_title = (course.get("title", "") + " " + course.get("description", "")).strip().lower()

    # ✅ Sortie corrigée : mêmes colonnes, même ordre que le modèle
    return {
        "desc_len_words": desc_len_words,
        "n_diplomas": n_diplomas,
        "n_experiences": n_experiences,
        "has_master": has_master,
        "has_licence": has_licence,
        "has_doctorat": has_doctorat,
        "has_certif": has_certif,
        "has_secondary": has_secondary,
        "has_teaching_exp": has_teaching_exp,
        "has_industry_exp": has_industry_exp,
        "has_management_exp": has_management_exp,
        "has_academic_exp": has_academic_exp,
        "teacher_mean_excl": teacher_mean_excl,
        "teacher_course_count": teacher_course_count,
        "course_title": course_title
    }


# ===== Endpoint principal =====
@app.post("/api/predict")
async def predict_rating(data: ProfessorData):
    try:
        # Lire le JSON brut reçu
        body = data.dict()

        print("\n===== 📥 Données reçues =====")
        print(json.dumps(body, indent=2, ensure_ascii=False))
        print("=============================\n")

        # Adapter les données
        adapted = adapt_input(body)
        print("➡️ Features calculées (adapted input) :")
        print(json.dumps(adapted, indent=2, ensure_ascii=False))

        # Créer DataFrame dans le bon ordre
        df = pd.DataFrame([adapted])
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

        print("\n➡️ Colonnes envoyées au modèle :")
        print(list(df.columns))

        # Prédire
        prediction = model.predict(df)[0]
        print(f"✅ Prédiction brute : {prediction}")

        return {"gradeAverage": round(float(prediction), 2)}

    except Exception as e:
        print("❌ Erreur pendant la prédiction :", e)
        return {"error": str(e)}


# ===== Endpoint test =====
@app.get("/")
def root():
    return {
        "message": "Bienvenue sur l’API de prédiction de satisfaction 🎯",
        "usage": "POST /api/predict avec le JSON complet d’un professeur et d’un cours",
    }