"""
Train the Employee Promotion model and SAVE it as a frozen artefact.
Run: python src/train_model.py
This follows the Agile pattern: train once -> save model -> dashboard loads it.
When the model changes, the new saved file is pushed and auto-deployed to Streamlit.
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

# features the dashboard will collect / use
FEATURES = ["performance_score", "kpi_achievement_percent", "manager_rating",
            "tasks_completed", "years_at_company"]
TARGET = "promoted"
THRESHOLD = 0.75   # tuned decision threshold from Sprint 3

def train_and_save_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # scale -> SMOTE (train only) -> Logistic Regression, all in one saveable pipeline
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X_train, y_train)

    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= THRESHOLD).astype(int)
    rec = recall_score(y_test, pred)
    prec = precision_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    bundle = {
        "model": pipe,
        "threshold": THRESHOLD,
        "features": FEATURES,
        "recall": float(rec),
        "precision": float(prec),
        "f1": float(f1),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    print("Model training completed.")
    print(f"Validation recall: {rec:.3f}")
    print(f"Validation precision: {prec:.3f}")
    print(f"Validation F1: {f1:.3f}")
    print(f"Model saved to: {MODEL_PATH}")
    return bundle

if __name__ == "__main__":
    train_and_save_model()
