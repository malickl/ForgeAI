import joblib
model = joblib.load("models/model_best.joblib")
print(model.feature_names_in_)