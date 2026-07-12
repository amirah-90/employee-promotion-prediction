"""
Train the Employee Promotion model(s) and SAVE as a frozen artefact.
Trains TWO models:
  1. The FULL model (all features) - used for real evaluation/monitoring metrics.
  2. The SLIDER model (5 features) - used for the dashboard's live prediction sliders.
Run: python src/train_model.py
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dashboard_data.csv"
_SAMPLE = BASE_DIR / "data" / "sample_employees.csv"
if not DATA_PATH.exists() and _SAMPLE.exists():
    DATA_PATH = _SAMPLE
MODEL_PATH = BASE_DIR / "model" / "promotion_model.pkl"

SLIDER_FEATURES = ["performance_score", "kpi_achievement_percent", "manager_rating",
                    "tasks_completed", "years_at_company"]
TARGET = "promoted"
THRESHOLD = 0.75

def _train_pipeline(X_train, y_train):
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X_train, y_train)
    return pipe

def _evaluate(pipe, X_test, y_test, threshold=THRESHOLD):
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)
    return {
        "recall": float(recall_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
    }

def train_and_save_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # ---------- 1. FULL model (all features, real evaluation) ----------
    X_full = df.drop(columns=[TARGET])
    X_full_encoded = pd.get_dummies(X_full, drop_first=True).astype(int)
    y = df[TARGET]

    Xf_train, Xf_test, yf_train, yf_test = train_test_split(
        X_full_encoded, y, test_size=0.2, random_state=42, stratify=y)
    full_pipe = _train_pipeline(Xf_train, yf_train)
    full_metrics = _evaluate(full_pipe, Xf_test, yf_test)

    # ---------- 2. SLIDER model (5 features, for live dashboard prediction) ----------
    X_slider = df[SLIDER_FEATURES]
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        X_slider, y, test_size=0.2, random_state=42, stratify=y)
    slider_pipe = _train_pipeline(Xs_train, ys_train)
    slider_metrics = _evaluate(slider_pipe, Xs_test, ys_test)

    bundle = {
        "model": slider_pipe,                 # used for the live sliders
        "threshold": THRESHOLD,
        "features": SLIDER_FEATURES,
        "recall": slider_metrics["recall"],           # slider model's own recall (for reference)
        "precision": slider_metrics["precision"],
        "f1": slider_metrics["f1"],
        "full_model_recall": full_metrics["recall"],  # <-- always freshly computed, no manual typing
        "full_model_precision": full_metrics["precision"],
        "full_model_f1": full_metrics["f1"],
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    print("Model training completed.")
    print(f"Slider model recall:  {slider_metrics['recall']:.3f}")
    print(f"Full model recall:    {full_metrics['recall']:.3f}")
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()
