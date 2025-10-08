from fastapi import FastAPI
from typing import List, Dict, Any
import pandas as pd
import unicodedata
import joblib
from scipy.sparse import csr_matrix, hstack
import logging

# --- Configuration du logger ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialisation de l'app ---
app = FastAPI(title="API Prédiction IA - JSON brut", version="1.0")

# --- Chargement du modèle et du TF-IDF ---
model = joblib.load("models/model.pkl")
tfidf = joblib.load("models/tfidf.pkl")

# Colonnes numériques attendues
num_cols = [
    "nb_experiences", "nb_courses_taught",
    "highest_degree_certificat", "highest_degree_licence",
    "highest_degree_master", "highest_degree_doctorat"
]

# --- Fonctions utilitaires ---
def remove_accents(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFD', text)
    return text.encode('ascii', 'ignore').decode('utf-8').lower().strip()

def clean_degree(level: str) -> str:
    if not isinstance(level, str):
        return "inconnu"
    level = remove_accents(level)
    if any(word in level for word in ["doctorat", "phd", "doctor", "dr", "dr ing"]):
        return "doctorat"
    elif any(word in level for word in ["master", "maitre", "maitrise", "mastere", "bac +5"]):
        return "master"
    elif any(word in level for word in ["licence", "bachelor", "bac +3"]):
        return "licence"
    elif "certificat" in level or "certification" in level:
        return "certificat"
    else:
        return "inconnu"

def get_highest_degree(diplomas: List[Dict[str, Any]]) -> (str, str):
    hierarchy = ["certificat", "licence", "master", "doctorat"]
    best_rank = -1
    best_degree = "inconnu"
    best_title = ""
    for d in diplomas:
        level = clean_degree(d.get("level", ""))
        title = remove_accents(d.get("title", ""))
        if level in hierarchy and hierarchy.index(level) > best_rank:
            best_rank = hierarchy.index(level)
            best_degree = level
            best_title = title
    return best_degree, best_title

# --- Endpoint principal ---
@app.post("/api/predict_raw")
def predict_note_raw(data: Dict[str, Any]):
    """
    Endpoint pour prédire la note à partir du JSON brut (profil complet enseignant + cours)
    """
    try:
        diplomas = data.get("diplomas", [])
        experiences = data.get("experiences", [])
        past_courses = data.get("pastCourses", [])
        course = data.get("course", None)

        # Sélection du cours : si "course" présent, l'utiliser, sinon dernier de pastCourses
        selected_course = course if course is not None else (past_courses[-1] if past_courses else {})

        # Nettoyage et construction des textes
        highest_degree, diploma_title = get_highest_degree(diplomas)
        nb_experiences = len(experiences)
        nb_courses_taught = len(past_courses)

        # Description enseignant
        description = remove_accents(data.get("description", ""))

        # Diplômes titres concaténés
        diplomas_text = " | ".join([remove_accents(d.get("title", "")) for d in diplomas])

        # Expériences titres et entreprises concaténés
        exp_titles = " | ".join([remove_accents(e.get("title", "")) for e in experiences])
        exp_companies = " | ".join([remove_accents(e.get("company", "")) for e in experiences])

        # Titre du cours sélectionné
        course_title = remove_accents(selected_course.get("title", ""))

        # Reconstruction texte complet
        text_full = f"{description} {diplomas_text} {exp_titles} {exp_companies} {course_title}".strip()

        X_text = tfidf.transform([text_full])

        num_data = {col: 0 for col in num_cols}
        num_data["nb_experiences"] = nb_experiences
        num_data["nb_courses_taught"] = nb_courses_taught
        degree_col = f"highest_degree_{highest_degree.lower()}"
        if degree_col in num_cols:
            num_data[degree_col] = 1
        num_data = pd.DataFrame([num_data])

        X_num = csr_matrix(num_data[num_cols].values.astype("float32"))
        X_final = hstack([X_num, X_text])

        if X_final.shape[1] != model.n_features_in_:
            logger.warning(f"Ajustement des dimensions : {X_final.shape[1]} → {model.n_features_in_}")
            X_final = X_final[:, :model.n_features_in_]

        note = model.predict(X_final)[0]
        return {"note_predite": round(float(note), 2)}

    except Exception as e:
        logger.error(f"Erreur dans predict_raw: {str(e)}")
        return {"error": str(e)}