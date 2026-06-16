"""
utils/predict.py
Handles model loading and single-image prediction.
"""

import json
import numpy as np
import joblib
from pathlib import Path
from PIL import Image

IMG_SIZE     = (64, 64)
MODELS_DIR   = Path("models")
MODEL_PATH   = MODELS_DIR / "gesture_model.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
SCALER_PATH  = MODELS_DIR / "scaler.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

_model   = None
_encoder = None
_scaler  = None


def is_model_trained() -> bool:
    return MODEL_PATH.exists() and ENCODER_PATH.exists() and SCALER_PATH.exists()


def load_model():
    global _model, _encoder, _scaler
    if _model is None:
        _model   = joblib.load(MODEL_PATH)
        _encoder = joblib.load(ENCODER_PATH)
        _scaler  = joblib.load(SCALER_PATH)
    return _model, _encoder, _scaler


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Resize → greyscale → flatten → scale."""
    img = pil_image.convert("L").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32).flatten().reshape(1, -1)
    return arr


def predict(pil_image: Image.Image):
    """
    Returns:
        label (str): predicted gesture name
        confidence (float): probability of top class [0, 1]
        all_probs (dict): {class_name: probability}
    """
    model, encoder, scaler = load_model()
    arr = preprocess_image(pil_image)
    arr_scaled = scaler.transform(arr)
    proba = model.predict_proba(arr_scaled)[0]
    idx   = int(np.argmax(proba))
    label = encoder.classes_[idx]
    confidence = float(proba[idx])
    all_probs = {encoder.classes_[i]: float(p) for i, p in enumerate(proba)}
    return label, confidence, all_probs


def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None
