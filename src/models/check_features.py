import joblib

model = joblib.load("creditpath_pipeline.pkl")

print("\nTraining Features:\n")
print(model.feature_names_in_)