"""
train_model.py
==============
Downloads the LeapGestRecog dataset from Kaggle (or auto-generates synthetic data),
preprocesses images, trains a Random Forest classifier, and saves the model.

Usage:
    python train_model.py
"""

import os
import sys
import json
import time
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ── Windows safe print (ASCII only, no emoji) ──────────────────────────────
def log(msg):
    try:
        print(msg, flush=True)
    except Exception:
        print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)

# ── CONFIG ─────────────────────────────────────────────────────────────────
IMG_SIZE     = (64, 64)
TEST_SIZE    = 0.20
RANDOM_STATE = 42
MODELS_DIR   = Path("models")
DATASET_DIR  = Path("dataset")
METRICS_PATH = MODELS_DIR / "metrics.json"
MODEL_PATH   = MODELS_DIR / "gesture_model.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
SCALER_PATH  = MODELS_DIR / "scaler.pkl"
CM_PATH      = MODELS_DIR / "confusion_matrix.png"
SAMPLE_DIR   = Path("sample_images")

GESTURE_MAP = {
    "01_palm":       "Palm",
    "02_l":          "L Shape",
    "03_fist":       "Fist",
    "04_fist_moved": "Fist (Moved)",
    "05_thumb":      "Thumb Up",
    "06_index":      "Index Finger",
    "07_ok":         "OK Sign",
    "08_palm_moved": "Palm (Moved)",
    "09_c":          "C Shape",
    "10_down":       "Down",
}

# ── HELPERS ────────────────────────────────────────────────────────────────
def ensure_dirs():
    MODELS_DIR.mkdir(exist_ok=True)
    SAMPLE_DIR.mkdir(exist_ok=True)


def download_dataset():
    if DATASET_DIR.exists() and any(DATASET_DIR.rglob("*.png")):
        log("[OK] Dataset already present -- skipping download.")
        return True
    log("[...] Attempting Kaggle download ...")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "gti-upm/leapgestrecog",
            path=str(DATASET_DIR),
            unzip=True,
            quiet=False,
        )
        log("[OK] Kaggle download complete.")
        return True
    except Exception as e:
        log(f"[WARN] Kaggle download failed: {e}")
        return False


def generate_synthetic_dataset():
    log("[...] Generating synthetic dataset for demo ...")
    np.random.seed(RANDOM_STATE)
    classes = list(GESTURE_MAP.keys())
    n_per_class = 60
    for cls in classes:
        cls_dir = DATASET_DIR / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            arr = np.random.randint(0, 255, IMG_SIZE, dtype=np.uint8)
            offset = classes.index(cls) * 25
            arr = np.clip(arr.astype(int) + offset, 0, 255).astype(np.uint8)
            Image.fromarray(arr, mode="L").save(cls_dir / f"img_{i:04d}.png")
    log(f"[OK] Synthetic dataset created ({len(classes) * n_per_class} images).")


def load_images():
    X, y = [], []
    all_paths = (list(DATASET_DIR.rglob("*.png")) +
                 list(DATASET_DIR.rglob("*.jpg")) +
                 list(DATASET_DIR.rglob("*.jpeg")))
    if not all_paths:
        return np.array(X), np.array(y)

    log(f"[...] Loading {len(all_paths)} images ...")
    label_map = {k.lower(): v for k, v in GESTURE_MAP.items()}
    skipped = 0

    for img_path in tqdm(all_paths, ncols=80, ascii=True):
        label = None
        for part in [img_path.parent.name.lower(), img_path.parent.parent.name.lower()]:
            for key, val in label_map.items():
                if key in part:
                    label = val
                    break
            if label:
                break
        if label is None:
            skipped += 1
            continue
        try:
            img = Image.open(img_path).convert("L").resize(IMG_SIZE)
            X.append(np.array(img, dtype=np.float32).flatten())
            y.append(label)
        except Exception:
            skipped += 1

    if skipped:
        log(f"[WARN] Skipped {skipped} files.")
    return np.array(X), np.array(y)


def save_sample_images():
    for folder_name in GESTURE_MAP:
        for sub in DATASET_DIR.rglob(f"*{folder_name}*"):
            if sub.is_dir():
                imgs = list(sub.glob("*.png")) + list(sub.glob("*.jpg"))
                if imgs:
                    out = SAMPLE_DIR / f"{folder_name}.png"
                    if not out.exists():
                        shutil.copy(imgs[0], out)
                    break


def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix - Hand Gesture Recognition", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=150)
    plt.close()
    log(f"[OK] Confusion matrix saved -> {CM_PATH}")


# ── MAIN ───────────────────────────────────────────────────────────────────
def train():
    ensure_dirs()

    downloaded = download_dataset()
    if not downloaded or not any(DATASET_DIR.rglob("*.png")):
        generate_synthetic_dataset()

    X, y = load_images()
    if len(X) == 0:
        log("[ERROR] No images found. Exiting.")
        sys.exit(1)

    log(f"\n[INFO] Dataset: {len(X)} samples | {len(set(y))} classes")
    class_counts = pd.Series(y).value_counts()
    print(class_counts.to_string())

    le     = LabelEncoder()
    y_enc  = le.fit_transform(y)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y_enc, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y_enc)
    log(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    log("\n[...] Training Random Forest (200 trees) ...")
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE,
        n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)
    log(f"[INFO] Training time: {time.time()-t0:.1f}s")

    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred,        average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    log(f"\n[RESULTS]")
    log(f"  Accuracy  : {acc:.4f}")
    log(f"  Precision : {prec:.4f}")
    log(f"  Recall    : {rec:.4f}")
    log(f"  F1 Score  : {f1:.4f}")
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=le.classes_, zero_division=0))

    joblib.dump(model,  MODEL_PATH)
    joblib.dump(le,     ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metrics = {
        "accuracy":     round(float(acc),  4),
        "precision":    round(float(prec), 4),
        "recall":       round(float(rec),  4),
        "f1_score":     round(float(f1),   4),
        "n_classes":    int(len(le.classes_)),
        "n_train":      int(len(X_train)),
        "n_test":       int(len(X_test)),
        "classes":      list(le.classes_),
        "class_counts": class_counts.to_dict(),
        "cm":           cm.tolist(),
        "trained_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
        "synthetic":    not downloaded,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(cm, le.classes_)
    save_sample_images()

    log(f"\n[OK] Model saved       -> {MODEL_PATH}")
    log(f"[OK] Metrics saved     -> {METRICS_PATH}")
    log("\n[DONE] Training complete! Run:  streamlit run app.py\n")


if __name__ == "__main__":
    train()