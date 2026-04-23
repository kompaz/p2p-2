import joblib

MODEL_PATH = "./artifacts/model_pipeline.pkl"

model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")
print(type(model))