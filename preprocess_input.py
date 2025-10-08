import os
import re
import time
import json
import random
import unicodedata
import numpy as np
import pandas as pd
from mistralai import Mistral

# Liste de thèmes autorisés
THEMES = [
    "Initiation & bases de programmation",
    "Web front-end & frameworks (HTML/CSS/JS, React/Angular)",
    "Algorithmes & structures de données",
    "Programmation orientée objet (POO/OOP)",
    "Python (langage & écosystème, dont web Python)",
    "Data science, analyse & data mining",
    "Java/C++ & mobile (Android, iOS, Flutter)",
    "Intelligence artificielle & deep learning",
    "Bases de données & SQL",
    "Architecture logicielle, cloud & sécurité",
]

# ----------------- UTILITAIRES -----------------
def remove_accents(text):
    if not isinstance(text, str): return text
    t = unicodedata.normalize('NFD', text)
    return t.encode('ascii', 'ignore').decode('utf-8')

def clean_degree(level):
    if not isinstance(level, str): return "inconnu"
    level = remove_accents(level.lower().strip())
    if any(w in level for w in ["doctorat","phd","doctor","dr","dr ing"]): return "doctorat"
    if any(w in level for w in ["master","maitre","maitrise","mastère","bac +5"]): return "master"
    if any(w in level for w in ["licence","bachelor","bac +3"]): return "licence"
    if "certificat" in level or "certification" in level: return "certificat"
    return "inconnu"

# ----------------- EXTRACTION PROF -----------------
def analyze_prof_features(prof, idx=0):
    firstname = remove_accents(prof.get("fistname","")).strip().lower().replace(" ","_")
    lastname  = remove_accents(prof.get("lastname","")).strip().lower().replace(" ","_")
    prof_id = f"{firstname}_{lastname}_{idx}"

    diplomas = prof.get("diplomas", [])
    experiences = prof.get("experiences", [])
    courses = prof.get("pastCourses", [])

    hierarchy = ["certificat","licence","master","doctorat"]
    best_rank, best_degree = -1, "inconnu"
    for d in diplomas:
        level = clean_degree(d.get("level",""))
        if level in hierarchy and hierarchy.index(level) > best_rank:
            best_rank, best_degree = hierarchy.index(level), level

    stars = [c.get("numberOfStars") for c in courses if c.get("numberOfStars") is not None]
    if stars:
        avg = float(np.mean(stars))
        std = float(np.std(stars))
        best = float(np.max(stars))
        worst = float(np.min(stars))
        trend = float(stars[-1] - stars[0])
    else:
        avg = std = best = worst = trend = np.nan

    return {
        "prof_id": prof_id,
        "highest_degree": best_degree,
        "nb_diplomas": len(diplomas),
        "nb_experiences": len(experiences),
        "nb_courses_taught": len(courses),
        "avg_past_stars": avg,
        "std_past_stars": std,
        "best_past_star": best,
        "worst_past_star": worst,
        "performance_trend": trend,
    }

# ----------------- APPEL MISTRAL -----------------
def ask_mistral_theme(title, description="", retries=5, backoff_base=1.8):
    if "MISTRAL_API_KEY" not in os.environ or not os.environ["MISTRAL_API_KEY"]:
        raise ValueError("⚠️ MISTRAL_API_KEY manquante dans les variables d'environnement.")

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    prompt = f"""
Tu es un assistant d'analyse de cours.
À partir du titre (et description) ci-dessous, choisis UNIQUEMENT un thème principal **exactement** parmi la liste.
Réponds STRICTEMENT en JSON valide, une seule ligne:
{{"course_theme":"..."}}  (la valeur doit être exactement un des thèmes)

Thèmes autorisés:
{THEMES}

Titre : {title}
Description : {description}
    """.strip()

    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            txt = resp.choices[0].message.content.strip()
            txt = re.sub(r"```json|```", "", txt).strip()
            data = json.loads(txt)
            theme = data.get("course_theme", "autre")
            return theme if theme in THEMES else "autre"
        except Exception as e:
            last_err = e
            sleep_s = (backoff_base ** attempt) + random.uniform(0, 0.4)
            print(f"⚠️ tentative {attempt+1}/{retries} sur '{title}' → {e} | pause {sleep_s:.1f}s")
            time.sleep(sleep_s)
    print(f"❌ échec définitif sur '{title}' → {last_err}")
    return "autre"

# ----------------- PIPELINE -----------------
def preprocess_input(data_json):
    """
    Transforme un JSON brut (profil enseignant + cours à venir)
    en une ligne DataFrame conforme au dataset d'entraînement.
    """
    prof = data_json[0]

    # 1) Extraire features prof
    feats = analyze_prof_features(prof)

    # 2) Cours courant
    title = prof.get("current_course", "")
    desc = prof.get("current_course_description", "")

    # 3) Appel Mistral
    course_theme = ask_mistral_theme(title, desc)

    # 4) Fusion
    df = pd.DataFrame([{
        "prof_id": feats["prof_id"],
        "course_title": title,
        "course_theme": course_theme,
        **feats
    }])

    return df