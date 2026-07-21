import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def load_feature_names():
    return joblib.load(FEATURE_PATH)