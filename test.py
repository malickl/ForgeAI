from preprocess_input import preprocess_input

sample = [{
    "fistname": "Antoine",
    "lastname": "Dubois",
    "city": "Paris",
    "diplomas": [{"level": "Master"}, {"level": "Licence"}],
    "experiences": [{"title": "Formateur en développement"}, {"title": "Professeur d'informatique"}],
    "pastCourses": [
        {"title": "Programmation Orientée Objet", "numberOfStars": 4.5},
        {"title": "Développement Web avec JavaScript", "numberOfStars": 4.9},
    ],
    "current_course": "Python avancé pour l’analyse de données",
    "current_course_description": "Cours pratique sur Pandas, NumPy et visualisation."
}]

df = preprocess_input(sample)
print(df["course_theme"])  # Affiche le thème prédit