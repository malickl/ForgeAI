import joblib

model = joblib.load("models/model_best.joblib")

print("\n📊 Colonnes attendues par le modèle :")
print(model.feature_names_in_)