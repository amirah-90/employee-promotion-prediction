"""Tests for the saved model and prediction utilities, run by pytest in CI/CD."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict_utils import load_model_bundle, predict_promotion

def test_model_bundle_loads():
    bundle = load_model_bundle()
    assert "model" in bundle
    assert "threshold" in bundle
    assert "features" in bundle

def test_prediction_is_binary():
    bundle = load_model_bundle()
    sample = {f: 50 for f in bundle["features"]}
    pred, prob = predict_promotion(sample, bundle)
    assert pred in (0, 1)
    assert 0.0 <= prob <= 1.0

def test_threshold_is_valid():
    bundle = load_model_bundle()
    assert 0.0 < bundle["threshold"] < 1.0
