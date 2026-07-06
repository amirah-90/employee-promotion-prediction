"""
Prediction utilities. The dashboard imports these to LOAD the frozen saved model
and make predictions. Prediction logic is kept separate from the dashboard so it
is easy to test and maintain.
"""
from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "promotion_model.pkl"

def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Run src/train_model.py first.")
    return joblib.load(MODEL_PATH)

def predict_promotion(input_data, bundle=None):
    """input_data: dict or DataFrame with the model's features.
    Returns (prediction 0/1, probability)."""
    if bundle is None:
        bundle = load_model_bundle()
    model = bundle["model"]
    threshold = bundle["threshold"]
    features = bundle["features"]

    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    prob = float(model.predict_proba(input_data[features])[:, 1][0])
    pred = int(prob >= threshold)
    return pred, prob
