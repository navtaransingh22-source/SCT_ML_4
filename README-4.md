# SCT_ML_4
# 🖐️ Hand Gesture Recognition System

> A production-ready Machine Learning project that classifies hand gestures from images using a **Random Forest classifier** trained on the **LeapGestRecog** dataset.  
> Built with Python · Scikit-learn · Streamlit · Matplotlib · Seaborn.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Image Preprocessing](#image-preprocessing)
- [Model Architecture](#model-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Installation & Running](#installation--running)
- [Screenshots](#screenshots)
- [Tech Stack](#tech-stack)

---

## Project Overview

Hand Gesture Recognition (HGR) is a computer-vision task that maps hand poses captured in images to semantic labels (e.g. "Thumbs Up", "Fist", "OK"). This project implements a full ML pipeline:

```
Raw Image → Greyscale → Resize → Flatten → Standardise → Random Forest → Predicted Gesture
```

Applications include sign-language interpretation, touchless UI control, gaming, and robotics.

---

## Dataset

| Property | Value |
|---|---|
| **Name** | LeapGestRecog |
| **Source** | [Kaggle — gti-upm/leapgestrecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) |
| **Sensor** | Leap Motion Controller (near-infrared) |
| **Subjects** | 10 participants |
| **Classes** | 10 hand gestures |
| **Format** | Greyscale PNG |
| **Licence** | CC0 (Public Domain) |

### Gesture Classes

| ID | Folder | Label |
|---|---|---|
| 01 | `01_palm` | Palm |
| 02 | `02_l` | L Shape |
| 03 | `03_fist` | Fist |
| 04 | `04_fist_moved` | Fist (Moved) |
| 05 | `05_thumb` | Thumb Up |
| 06 | `06_index` | Index Finger |
| 07 | `07_ok` | OK Sign |
| 08 | `08_palm_moved` | Palm (Moved) |
| 09 | `09_c` | C Shape |
| 10 | `10_down` | Down |

---

## Image Preprocessing

Each image goes through a deterministic preprocessing pipeline before entering the model:

1. **Load** — `PIL.Image.open()` reads any PNG/JPG.
2. **Greyscale** — `.convert("L")` collapses RGB → single luminance channel (eliminates colour bias).
3. **Resize** — `.resize((64, 64))` gives every image an identical spatial footprint (4 096 pixels).
4. **Flatten** — `numpy.flatten()` converts the 2-D matrix into a 1-D feature vector.
5. **Standardise** — `StandardScaler` (fit on training data only) transforms each feature to zero mean and unit variance, preventing large pixel ranges from dominating distance-based splits.

---

## Model Architecture

### Algorithm — Random Forest Classifier

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 200 trees |
| `max_depth` | None (full trees) |
| `min_samples_split` | 2 |
| `class_weight` | "balanced" |
| `random_state` | 42 |
| `n_jobs` | -1 (all CPUs) |

**Why Random Forest?**
- Ensemble method: 200 decision trees vote → reduces variance significantly over a single tree.
- Handles high-dimensional data (4 096 features) well with feature subsampling.
- Returns `predict_proba` for confidence scores.
- `class_weight="balanced"` corrects for any class imbalance in the data.

### Train / Test Split
- 80 % training / 20 % test (`stratify=y` maintains class proportions in both sets).

---

## Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| **Accuracy** | Correct / Total | Overall fraction of correct predictions |
| **Precision** | TP / (TP + FP) | Of all positive predictions, how many were truly positive |
| **Recall** | TP / (TP + FN) | Of all actual positives, how many were found |
| **F1 Score** | 2 · P · R / (P + R) | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | Grid (True vs Predicted) | Detailed per-class breakdown of errors |

All multi-class scores are computed with `average="weighted"` to account for class imbalance.

---

## Project Structure

```
hand_gesture_recognition/
├── app.py                   # Streamlit multi-page dashboard
├── train_model.py           # Data download → preprocess → train → evaluate → save
├── requirements.txt         # pip dependencies
├── README.md                # This file
│
├── models/                  # Created after training
│   ├── gesture_model.pkl    # Serialised Random Forest
│   ├── label_encoder.pkl    # LabelEncoder (class ↔ integer)
│   ├── scaler.pkl           # StandardScaler (fit on train set)
│   ├── metrics.json         # All evaluation scores
│   └── confusion_matrix.png # Heatmap
│
├── utils/
│   ├── __init__.py
│   └── predict.py           # load_model(), predict(), load_metrics()
│
├── sample_images/           # One representative image per class
└── dataset/                 # LeapGestRecog images (auto-populated)
    ├── 01_palm/
    ├── 02_l/
    └── …
```

---

## Installation & Running

### Prerequisites
- Python 3.9 or higher
- pip

### Step 1 — Clone / download the project

```bash
git clone https://github.com/your-username/hand-gesture-recognition.git
cd hand-gesture-recognition
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — (Recommended) Set up Kaggle API

To download the full LeapGestRecog dataset automatically:

1. Go to [kaggle.com → Account → API → Create New Token](https://www.kaggle.com/settings)
2. Download `kaggle.json`
3. Place it at `~/.kaggle/kaggle.json` (Linux/macOS) or `C:\Users\<user>\.kaggle\kaggle.json` (Windows)

> **Without Kaggle credentials:** `train_model.py` automatically generates a synthetic dataset so every feature of the app still works — the accuracy will be lower but the full pipeline runs.

### Step 4 — Train the model

```bash
python train_model.py
```

This will:
- Download the dataset (or generate synthetic data)
- Preprocess all images
- Train and evaluate the Random Forest
- Save the model, metrics and confusion matrix to `models/`

### Step 5 — Launch the dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Streamlit Dashboard Pages

| Page | Description |
|---|---|
| 🏠 Dashboard | KPI cards, pipeline overview, performance charts |
| 📊 Model Analytics | Confusion matrix, per-class F1/Precision/Recall |
| 🔮 Predict Gesture | Upload image → instant AI prediction with confidence |
| 🖼️ Dataset Preview | Sample images, preprocessing explanation, class distribution |
| ℹ️ About | Architecture, concepts, how-to guide |

---

## Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Interactive web dashboard |
| `scikit-learn` | ML algorithm, metrics, preprocessing |
| `numpy` | Array operations |
| `pandas` | Tabular data, reports |
| `Pillow` | Image loading & preprocessing |
| `opencv-python-headless` | Additional image utilities |
| `matplotlib` | Charts and confusion matrix |
| `seaborn` | Heatmaps |
| `joblib` | Model serialisation |
| `kaggle` | Dataset download |
| `tqdm` | Progress bars during training |

---

## Acknowledgements

- Dataset by **GTI — Universidad Politécnica de Madrid** published on Kaggle under CC0.
- Project inspired by real-world gesture-based HCI research.

---

*Built as an internship-level portfolio project demonstrating the full ML lifecycle: data acquisition, preprocessing, model training, evaluation, persistence, and deployment.*
