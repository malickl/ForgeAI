from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Charger les modèles ---
model = joblib.load("models/model.pkl")
tfidf = joblib.load("models/tfidf.pkl")

# Colonnes numériques (même ordre qu'à l'entraînement)
num_cols = ["nb_experiences", "nb_courses_taught",
            "highest_degree_certificat", "highest_degree_licence",
            "highest_degree_master", "highest_degree_doctorat"]

# --- Définir l'app FastAPI ---
app = FastAPI(title="IA - Prédiction de satisfaction des apprenants")

# --- Modèle d’entrée ---
class ProfInput(BaseModel):
    highest_degree: str
    nb_experiences: int
    nb_courses_taught: int
    course_title: str
    diploma_title: str
    exp_title: str
    exp_company: str

# --- Route principale ---
@app.post("/api/predict")
def predict_note(data: ProfInput):
    # Texte combiné
    text_full = f"{data.course_title} {data.diploma_title} {data.exp_title} {data.exp_company}"
    X_text = tfidf.transform([text_full])

    # Créer toutes les colonnes numériques dans le bon ordre
    num_data = {col: 0 for col in num_cols}
    num_data["nb_experiences"] = data.nb_experiences
    num_data["nb_courses_taught"] = data.nb_courses_taught

    # Activer la bonne colonne de diplôme si elle existe
    degree_col = f"highest_degree_{data.highest_degree.lower()}"
    if degree_col in num_cols:
        num_data[degree_col] = 1

    num_data = pd.DataFrame([num_data])

    X_num = csr_matrix(num_data[num_cols].values.astype("float32"))
    X_final = hstack([X_num, X_text])

    logger.info(f"num_cols dans l'API : {num_cols}")
    logger.info(f"Colonnes attendues : {model.get_booster().feature_names}")
    logger.info(f"Colonnes fournies : {X_final.shape[1]}")

    # S'assurer que la forme correspond à celle du modèle
    if X_final.shape[1] != model.n_features_in_:
        logger.warning(f"Ajustement de la forme des features : {X_final.shape[1]} → {model.n_features_in_}")
        X_final = X_final[:, :model.n_features_in_]
    # Prédiction
    note = model.predict(X_final)[0]
    return {"note_predite": round(float(note), 2)}