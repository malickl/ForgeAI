from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# ===============================
# ⚙️ Configuration
# ===============================
MODEL_PATH = "models/model_best.joblib"  # ou model.pkl selon ton fichier
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Satisfaction Prediction API",
    description="API de prédiction de satisfaction apprenants",
    version="1.0.0"
)

# ===============================
# 📦 Schémas d’entrée
# ===============================
class Diploma(BaseModel):
    level: str = ""
    title: str = ""

class Experience(BaseModel):
    title: str = ""
    description: str = ""

class PastCourse(BaseModel):
    title: str = ""
    description: str = ""
    numberOfStars: float | None = None

class Professor(BaseModel):
    fistname: str = ""
    lastname: str = ""
    city: str = ""
    description: str = ""
    diplomas: list[Diploma] = []
    experiences: list[Experience] = []
    pastCourses: list[PastCourse] = []

class Course(BaseModel):
    title: str = ""
    description: str = ""

class InputData(BaseModel):
    professor: Professor
    course: Course

# ===============================
# 🧩 Prétraitement
# ===============================
def adapt_input(body: dict) -> dict:
    prof = body.get("professor", {})
    course = body.get("course", {})

    diplomas = prof.get("diplomas", [])
    exps = prof.get("experiences", [])
    past_courses = prof.get("pastCourses", [])

    # Diplômes
    levels = [d.get("level", "").lower() for d in diplomas]
    has_master = int(any("master" in lvl for lvl in levels))
    has_licence = int(any("licence" in lvl or "bachelor" in lvl for lvl in levels))
    has_doctorat = int(any("doctor" in lvl or "phd" in lvl for lvl in levels))
    has_certif = int(any("certif" in lvl for lvl in levels))
    has_secondary = int(any("second" in lvl for lvl in levels))

    # Expériences
    all_exp_text = " ".join(
        f"{e.get('title', '')} {e.get('description', '')}".lower()
        for e in exps
    )
    has_teaching_exp = int(any(k in all_exp_text for k in ["prof", "enseign", "formateur"]))
    has_industry_exp = int(any(k in all_exp_text for k in ["dev", "engineer", "ingénieur", "consult", "industrie"]))
    has_management_exp = int(any(k in all_exp_text for k in ["chef", "lead", "manager", "responsable"]))
    has_academic_exp = int(any(k in all_exp_text for k in ["recherche", "thèse", "doctorat", "universit"]))

    # Moyenne et nombre de cours passés
    past_ratings = [c.get("numberOfStars") for c in past_courses if c.get("numberOfStars") is not None]
    teacher_mean_excl = float(sum(past_ratings) / len(past_ratings)) if past_ratings else 0
    teacher_course_count = len(past_courses)

    # Infos générales
    desc_len_words = len(prof.get("description", "").split())
    n_diplomas = len(diplomas)
    n_experiences = len(exps)

    # Nouveau cours
    course_title = course.get("title", "").strip().lower()
    course_description = course.get("description", "").strip().lower()
    full_course_text = f"{course_title} {course_description}".strip()

    # ✅ Données prêtes pour le modèle
    return {
        "course_title": full_course_text,
        "desc_len_words": desc_len_words,
        "n_diplomas": n_diplomas,
        "n_experiences": n_experiences,
        "teacher_mean_excl": teacher_mean_excl,
        "teacher_course_count": teacher_course_count,
        "has_master": has_master,
        "has_licence": has_licence,
        "has_doctorat": has_doctorat,
        "has_certif": has_certif,
        "has_secondary": has_secondary,
        "has_teaching_exp": has_teaching_exp,
        "has_industry_exp": has_industry_exp,
        "has_management_exp": has_management_exp,
        "has_academic_exp": has_academic_exp
    }

# ===============================
# 🚀 Endpoint principal
# ===============================
@app.post("/api/predict")
def predict(data: InputData):
    body = data.dict()
    features = adapt_input(body)
    df = pd.DataFrame([features])

    prediction = model.predict(df)[0]
    prediction = round(float(prediction), 2)

    return {"gradeAverage": prediction}

# ===============================
# 🧪 Test rapide
# ===============================
@app.get("/")
def root():
    return {
        "message": "Bienvenue sur l’API de prédiction de satisfaction 🎯",
        "usage": "POST /api/predict avec le JSON d’un professeur et d’un cours"
    }