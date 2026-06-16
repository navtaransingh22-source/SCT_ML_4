"""
app.py  —  Hand Gesture Recognition System
Streamlit dashboard · Random Forest · LeapGestRecog dataset
"""

import os
import sys
import json
import warnings
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from PIL import Image

import streamlit as st

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.predict import (
    is_model_trained, predict, load_metrics, preprocess_image,
    IMG_SIZE, MODELS_DIR, ENCODER_PATH
)

SAMPLE_DIR   = ROOT / "sample_images"
CM_PATH      = MODELS_DIR / "confusion_matrix.png"
METRICS_PATH = MODELS_DIR / "metrics.json"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root palette ── */
:root {
  --bg-deep:    #0a0e1a;
  --bg-card:    #111827;
  --bg-hover:   #1c2537;
  --accent:     #6366f1;
  --accent2:    #06b6d4;
  --accent3:    #10b981;
  --warn:       #f59e0b;
  --danger:     #ef4444;
  --text-main:  #f1f5f9;
  --text-muted: #94a3b8;
  --border:     #1e293b;
  --radius:     14px;
  --shadow:     0 8px 32px rgba(0,0,0,.45);
}

/* ── Global ── */
html, body, [class*="css"] {
  font-family: 'Space Grotesk', sans-serif !important;
  background-color: var(--bg-deep) !important;
  color: var(--text-main) !important;
}

/* ── Remove default Streamlit padding ── */
.block-container { padding-top: 1.5rem !important; max-width: 1280px; }
.stApp { background-color: var(--bg-deep) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f172a 0%, #0a0e1a 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-main) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p { color: var(--text-muted) !important; }

/* ── Header banner ── */
.hero-banner {
  background: linear-gradient(135deg, #1a1f35 0%, #0f172a 50%, #1a1432 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 2rem 2.5rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
}
.hero-banner::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(ellipse 60% 80% at 80% 50%, rgba(99,102,241,.12) 0%, transparent 70%);
  pointer-events: none;
}
.hero-title {
  font-size: 2.2rem;
  font-weight: 700;
  background: linear-gradient(90deg, #a5b4fc, #67e8f9);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin: 0 0 .35rem;
  line-height: 1.2;
}
.hero-sub {
  color: var(--text-muted);
  font-size: 1rem;
  margin: 0;
}

/* ── Metric cards ── */
.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.2rem 1.5rem;
  text-align: center;
  transition: transform .2s, border-color .2s;
}
.metric-card:hover { transform: translateY(-3px); border-color: var(--accent); }
.metric-label {
  color: var(--text-muted);
  font-size: .78rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: .08em;
  margin-bottom: .4rem;
}
.metric-value {
  font-size: 2rem;
  font-weight: 700;
  color: var(--text-main);
  font-family: 'JetBrains Mono', monospace;
}
.metric-icon { font-size: 1.4rem; margin-bottom: .3rem; }

/* ── Section title ── */
.section-title {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text-main);
  border-left: 3px solid var(--accent);
  padding-left: .75rem;
  margin: 1.5rem 0 1rem;
}

/* ── Info / warning boxes ── */
.info-box {
  background: rgba(99,102,241,.1);
  border: 1px solid rgba(99,102,241,.35);
  border-radius: 10px;
  padding: 1rem 1.25rem;
  color: #c7d2fe;
  font-size: .9rem;
}
.warn-box {
  background: rgba(245,158,11,.08);
  border: 1px solid rgba(245,158,11,.35);
  border-radius: 10px;
  padding: 1rem 1.25rem;
  color: #fde68a;
  font-size: .9rem;
}
.success-box {
  background: rgba(16,185,129,.08);
  border: 1px solid rgba(16,185,129,.3);
  border-radius: 10px;
  padding: 1rem 1.25rem;
  color: #6ee7b7;
  font-size: .9rem;
}

/* ── Prediction result ── */
.pred-badge {
  background: linear-gradient(135deg, rgba(99,102,241,.2), rgba(6,182,212,.2));
  border: 1px solid rgba(99,102,241,.5);
  border-radius: 12px;
  padding: 1.5rem;
  text-align: center;
}
.pred-label {
  font-size: 2rem;
  font-weight: 700;
  color: #a5b4fc;
}
.pred-conf {
  color: var(--text-muted);
  font-size: .9rem;
  margin-top: .25rem;
}

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), #4f46e5) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  padding: .55rem 1.5rem !important;
  font-family: 'Space Grotesk', sans-serif !important;
  transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .85 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: var(--bg-card) !important;
  border: 1px dashed #334155 !important;
  border-radius: var(--radius) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  gap: .25rem;
  background: var(--bg-card) !important;
  border-radius: 10px;
  padding: .3rem;
  border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px !important;
  color: var(--text-muted) !important;
  font-weight: 500 !important;
  padding: .4rem .9rem !important;
}
.stTabs [aria-selected="true"] {
  background: var(--accent) !important;
  color: #fff !important;
}
.stTabs [data-baseweb="tab-panel"] { background: transparent !important; }

/* ── Progress bars ── */
.stProgress > div > div { background: var(--accent) !important; }

/* ── Generic text ── */
h1,h2,h3,h4,h5,h6,p,span,label,li { color: var(--text-main) !important; }
.stMarkdown p { color: var(--text-main) !important; }
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }
[data-testid="stMetricValue"] { color: var(--text-main) !important; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { background: var(--bg-card) !important; border-radius: var(--radius) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }

/* ── Sample image grid ── */
.sample-grid { display:flex; flex-wrap:wrap; gap:.75rem; }
.sample-item {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: .6rem;
  text-align: center;
  width: 110px;
  font-size: .75rem;
  color: var(--text-muted);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 .5rem;'>
      <div style='font-size:2.5rem;'>🖐️</div>
      <div style='font-size:1rem; font-weight:700; color:#a5b4fc;'>Gesture AI</div>
      <div style='font-size:.75rem; color:#64748b; margin-top:.2rem;'>Recognition System v1.0</div>
    </div>
    <hr style='border-color:#1e293b; margin:.75rem 0;'>
    """, unsafe_allow_html=True)

    nav = st.selectbox(
        "Navigate",
        ["🏠 Dashboard", "📊 Model Analytics", "🔮 Predict Gesture", "🖼️ Dataset Preview", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#1e293b; margin:.75rem 0;'>", unsafe_allow_html=True)

    model_trained = is_model_trained()
    if model_trained:
        st.markdown("""
        <div class='success-box'>
          <b>✅ Model Ready</b><br>
          <span style='font-size:.8rem;'>Trained model loaded.</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='warn-box'>
          <b>⚠️ Model Not Found</b><br>
          <span style='font-size:.8rem;'>Run train_model.py first.</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Train Model Now", use_container_width=True):
        with st.spinner("Training in progress — this may take a few minutes …"):
            result = subprocess.run(
                [sys.executable, str(ROOT / "train_model.py")],
                capture_output=True, text=True, cwd=str(ROOT)
            )
        if result.returncode == 0:
            st.success("✅ Training complete! Refresh the page.")
            st.rerun()
        else:
            st.error("Training failed. Check terminal output.")
            st.code(result.stderr[-2000:], language="bash")

    st.markdown("""
    <hr style='border-color:#1e293b; margin:.75rem 0;'>
    <div style='font-size:.7rem; color:#475569; text-align:center;'>
      LeapGestRecog Dataset<br>Random Forest Classifier<br>Scikit-learn · Streamlit
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def dark_fig():
    """Return a matplotlib Figure with dark theme matching the UI."""
    fig, ax = plt.subplots(facecolor="#111827")
    ax.set_facecolor("#111827")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")
    ax.tick_params(colors="#94a3b8")
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    ax.title.set_color("#f1f5f9")
    return fig, ax


def metric_card(icon, label, value, color="#a5b4fc"):
    st.markdown(f"""
    <div class='metric-card'>
      <div class='metric-icon'>{icon}</div>
      <div class='metric-label'>{label}</div>
      <div class='metric-value' style='color:{color};'>{value}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown("""
    <div class='hero-banner'>
      <div class='hero-title'>🖐️ Hand Gesture Recognition System</div>
      <div class='hero-sub'>
        Computer Vision · Random Forest · LeapGestRecog Dataset · Scikit-learn + Streamlit
      </div>
    </div>
    """, unsafe_allow_html=True)

    metrics = load_metrics()

    # ── Quick-stats row ──────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    if metrics:
        vals = [
            ("🎯", "Accuracy",  f"{metrics['accuracy']*100:.1f}%",  "#a5b4fc"),
            ("🔬", "Precision", f"{metrics['precision']*100:.1f}%", "#67e8f9"),
            ("📡", "Recall",    f"{metrics['recall']*100:.1f}%",    "#6ee7b7"),
            ("⚡", "F1 Score",  f"{metrics['f1_score']*100:.1f}%",  "#fbbf24"),
            ("🏷️", "Classes",   str(metrics['n_classes']),          "#f472b6"),
        ]
    else:
        vals = [
            ("🎯","Accuracy","—","#a5b4fc"),("🔬","Precision","—","#67e8f9"),
            ("📡","Recall","—","#6ee7b7"),("⚡","F1 Score","—","#fbbf24"),
            ("🏷️","Classes","10","#f472b6"),
        ]
    for col, (icon, lbl, val, color) in zip([c1,c2,c3,c4,c5], vals):
        with col:
            metric_card(icon, lbl, val, color)

    # ── Info / description ──────────────────────────────────
    st.markdown("<div class='section-title'>Project Overview</div>", unsafe_allow_html=True)
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown("""
        <div class='info-box'>
        <b>What is Hand Gesture Recognition?</b><br><br>
        Hand gesture recognition is a computer vision technique that enables machines to 
        interpret human hand movements as commands or inputs. It powers applications in 
        sign-language translation, human-computer interaction, gaming, robotics, and 
        accessibility technology.<br><br>
        This system uses the <b>LeapGestRecog</b> dataset — 10 distinct hand gestures 
        captured under near-infrared light by a Leap Motion sensor — and classifies them 
        with a <b>Random Forest</b> model trained on flattened, normalised grayscale images.
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div style='background:var(--bg-card); border:1px solid var(--border); 
                    border-radius:var(--radius); padding:1.2rem;'>
          <div style='color:#94a3b8; font-size:.78rem; text-transform:uppercase; 
                      letter-spacing:.08em; margin-bottom:.8rem;'>Tech Stack</div>
          <div style='display:flex; flex-direction:column; gap:.5rem; font-size:.88rem;'>
            <span>🐍 Python 3.10+</span>
            <span>🌊 Streamlit 1.32</span>
            <span>🌲 Random Forest (sklearn)</span>
            <span>🖼️ Pillow / OpenCV</span>
            <span>📊 Matplotlib / Seaborn</span>
            <span>💾 Joblib (model save)</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Pipeline diagram ────────────────────────────────────
    st.markdown("<div class='section-title'>ML Pipeline</div>", unsafe_allow_html=True)
    steps = [
        ("📥", "Data\nCollection"),
        ("🔄", "Image\nPreprocessing"),
        ("✂️", "Train/Test\nSplit"),
        ("🌲", "Model\nTraining"),
        ("📈", "Evaluation\nMetrics"),
        ("💾", "Save\nModel"),
        ("🔮", "Prediction"),
    ]
    cols = st.columns(len(steps))
    for i, (col, (icon, lbl)) in enumerate(zip(cols, steps)):
        with col:
            connector = "→" if i < len(steps)-1 else ""
            st.markdown(f"""
            <div style='text-align:center;'>
              <div style='background:var(--bg-card); border:1px solid {"var(--accent)" if i%2==0 else "#334155"};
                          border-radius:10px; padding:.8rem .4rem; font-size:.85rem; line-height:1.4;'>
                <div style='font-size:1.3rem;'>{icon}</div>
                <div style='color:var(--text-main); font-size:.72rem; font-weight:600;
                            white-space:pre-line; margin-top:.3rem;'>{lbl}</div>
              </div>
              <div style='color:#475569; font-size:1rem; margin-top:.3rem;'>{connector}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Accuracy gauge if trained ───────────────────────────
    if metrics:
        st.markdown("<div class='section-title'>Performance at a Glance</div>", unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        with g1:
            # Bar chart of metrics
            fig, ax = dark_fig()
            m_names  = ["Accuracy", "Precision", "Recall", "F1 Score"]
            m_values = [metrics["accuracy"], metrics["precision"],
                        metrics["recall"],   metrics["f1_score"]]
            colors = ["#6366f1","#06b6d4","#10b981","#f59e0b"]
            bars = ax.barh(m_names, [v*100 for v in m_values], color=colors, height=0.5)
            for bar, val in zip(bars, m_values):
                ax.text(bar.get_width()-1.5, bar.get_y()+bar.get_height()/2,
                        f"{val*100:.1f}%", va="center", ha="right",
                        color="white", fontsize=10, fontweight="bold")
            ax.set_xlim(0, 105)
            ax.set_xlabel("Score (%)")
            ax.set_title("Evaluation Metrics", fontsize=13)
            ax.grid(axis="x", color="#1e293b", linewidth=0.8)
            ax.set_facecolor("#111827")
            fig.patch.set_facecolor("#111827")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with g2:
            # Class distribution
            if "class_counts" in metrics:
                fig2, ax2 = dark_fig()
                cc = metrics["class_counts"]
                cnames = [k[:12] for k in cc.keys()]
                cvals  = list(cc.values())
                grad_colors = plt.cm.cool(np.linspace(0.3, 0.9, len(cnames)))
                wedges, texts, autotexts = ax2.pie(
                    cvals, labels=cnames, colors=grad_colors,
                    autopct="%1.0f%%", startangle=140,
                    textprops={"color":"#f1f5f9","fontsize":7},
                    pctdistance=0.78,
                )
                for at in autotexts:
                    at.set_fontsize(7)
                    at.set_color("white")
                ax2.set_title("Class Distribution", fontsize=13)
                fig2.patch.set_facecolor("#111827")
                st.pyplot(fig2, use_container_width=True)
                plt.close()

    if not metrics:
        st.markdown("""
        <div class='warn-box'>
          <b>⚠️  Model not trained yet.</b><br>
          Click <b>🚀 Train Model Now</b> in the sidebar, or run <code>python train_model.py</code>
          in your terminal to train and see live metrics here.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
def page_analytics():
    st.markdown("<div class='hero-banner'><div class='hero-title'>📊 Model Analytics</div><div class='hero-sub'>Detailed evaluation metrics, confusion matrix & class-level report</div></div>", unsafe_allow_html=True)

    metrics = load_metrics()
    if not metrics:
        st.markdown("<div class='warn-box'>⚠️ No metrics found. Train the model first.</div>", unsafe_allow_html=True)
        return

    # ── Metric scorecards ──────────────────────────────────
    st.markdown("<div class='section-title'>Evaluation Scores</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1,"🎯","Accuracy",  metrics["accuracy"],  "#a5b4fc"),
        (c2,"🔬","Precision", metrics["precision"], "#67e8f9"),
        (c3,"📡","Recall",    metrics["recall"],    "#6ee7b7"),
        (c4,"⚡","F1 Score",  metrics["f1_score"],  "#fbbf24"),
    ]
    for col, icon, lbl, val, color in cards:
        with col:
            metric_card(icon, lbl, f"{val*100:.2f}%", color)

    # ── Dataset split info ─────────────────────────────────
    st.markdown("<div class='section-title'>Dataset Split</div>", unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)
    total = metrics["n_train"] + metrics["n_test"]
    with sc1: metric_card("📦","Total Samples", str(total), "#a5b4fc")
    with sc2: metric_card("🏋️","Train Samples", str(metrics["n_train"]), "#6ee7b7")
    with sc3: metric_card("🧪","Test Samples",  str(metrics["n_test"]),  "#67e8f9")

    st.markdown(f"""
    <div style='color:#64748b; font-size:.82rem; margin: .5rem 0 1.5rem;'>
      Split ratio: 80 % train / 20 % test &nbsp;·&nbsp;
      Trained at: {metrics.get('trained_at','—')} &nbsp;·&nbsp;
      Synthetic data: {'Yes' if metrics.get('synthetic') else 'No'}
    </div>
    """, unsafe_allow_html=True)

    # ── Confusion Matrix ───────────────────────────────────
    st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
    if CM_PATH.exists():
        img = Image.open(CM_PATH)
        st.image(img, use_column_width=True)
    elif "cm" in metrics:
        classes = metrics["classes"]
        cm = np.array(metrics["cm"])
        fig, ax = plt.subplots(figsize=(11, 8), facecolor="#111827")
        ax.set_facecolor("#111827")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes,
                    linewidths=.5, linecolor="#1e293b", ax=ax,
                    annot_kws={"size":9, "color":"white"})
        ax.set_xlabel("Predicted", color="#94a3b8")
        ax.set_ylabel("True Label", color="#94a3b8")
        ax.set_title("Confusion Matrix", color="#f1f5f9", fontsize=14)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        plt.xticks(rotation=40, ha="right")
        plt.tight_layout()
        fig.patch.set_facecolor("#111827")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Per-class bar chart ────────────────────────────────
    if "cm" in metrics:
        st.markdown("<div class='section-title'>Per-Class Metrics</div>", unsafe_allow_html=True)
        classes = metrics["classes"]
        cm_arr  = np.array(metrics["cm"])

        per_class = []
        for i, cls in enumerate(classes):
            tp = cm_arr[i, i]
            fp = cm_arr[:, i].sum() - tp
            fn = cm_arr[i, :].sum() - tp
            prec = tp/(tp+fp) if (tp+fp) else 0
            rec  = tp/(tp+fn) if (tp+fn) else 0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
            per_class.append({"Gesture": cls, "Precision": round(prec,3),
                               "Recall": round(rec,3), "F1": round(f1,3)})

        df = pd.DataFrame(per_class).set_index("Gesture")

        fig3, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="#111827")
        palette = {"Precision":"#6366f1","Recall":"#06b6d4","F1":"#10b981"}
        for ax_i, (col, color) in zip(axes, palette.items()):
            vals = df[col]
            ax_i.barh(df.index, vals, color=color, height=0.6)
            ax_i.set_xlim(0, 1.1)
            ax_i.set_title(col, color="#f1f5f9", fontsize=11)
            ax_i.set_facecolor("#111827")
            ax_i.tick_params(colors="#94a3b8", labelsize=8)
            for spine in ax_i.spines.values():
                spine.set_edgecolor("#1e293b")
            for j, (val, name) in enumerate(zip(vals, df.index)):
                ax_i.text(val+.01, j, f"{val:.2f}", va="center",
                          color="#f1f5f9", fontsize=8)
            ax_i.grid(axis="x", color="#1e293b", linewidth=0.7)

        plt.suptitle("Per-Class Performance", color="#f1f5f9", fontsize=13, y=1.02)
        plt.tight_layout()
        fig3.patch.set_facecolor("#111827")
        st.pyplot(fig3, use_container_width=True)
        plt.close()

        st.markdown("<div class='section-title'>Classification Report Table</div>", unsafe_allow_html=True)
        st.dataframe(df.style.background_gradient(cmap="Blues", vmin=0, vmax=1)
                           .format("{:.3f}"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
def page_predict():
    st.markdown("<div class='hero-banner'><div class='hero-title'>🔮 Predict Gesture</div><div class='hero-sub'>Upload a hand gesture image and get an instant AI prediction</div></div>", unsafe_allow_html=True)

    if not is_model_trained():
        st.markdown("<div class='warn-box'>⚠️ Model not trained yet. Use the sidebar to train first.</div>", unsafe_allow_html=True)
        return

    col_up, col_res = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown("<div class='section-title'>Upload Image</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box' style='margin-bottom:1rem; font-size:.85rem;'>
          📌  Supported formats: <b>PNG, JPG, JPEG, BMP</b><br>
          For best results, use a grayscale or near-infrared hand image 
          similar to the LeapGestRecog dataset.
        </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader("Choose a gesture image", type=["png","jpg","jpeg","bmp"])

        # Also allow using sample images
        st.markdown("<div class='section-title'>… or Try a Sample</div>", unsafe_allow_html=True)
        samples = list(SAMPLE_DIR.glob("*.png")) if SAMPLE_DIR.exists() else []
        if samples:
            sample_names = {s.stem: s for s in samples}
            chosen = st.selectbox("Select sample image", ["—"] + list(sample_names.keys()))
            if chosen != "—":
                uploaded = sample_names[chosen]
        else:
            st.markdown("<div style='color:#64748b; font-size:.82rem;'>No sample images yet — train the model first.</div>", unsafe_allow_html=True)

    with col_res:
        st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)

        if uploaded is not None:
            try:
                if isinstance(uploaded, Path):
                    pil_img = Image.open(uploaded)
                else:
                    pil_img = Image.open(uploaded)

                # Show uploaded image
                display_img = pil_img.convert("RGB").resize((220, 220))
                st.image(display_img, caption="Input Image", width=220)

                with st.spinner("Analysing gesture …"):
                    label, conf, all_probs = predict(pil_img)

                st.markdown(f"""
                <div class='pred-badge'>
                  <div style='color:#94a3b8; font-size:.8rem; text-transform:uppercase;
                               letter-spacing:.1em; margin-bottom:.5rem;'>Detected Gesture</div>
                  <div class='pred-label'>✋ {label}</div>
                  <div class='pred-conf'>Confidence: <b style='color:#6ee7b7;'>{conf*100:.1f}%</b></div>
                </div>
                """, unsafe_allow_html=True)

                # Top-5 probabilities bar chart
                st.markdown("<div class='section-title'>Top Predictions</div>", unsafe_allow_html=True)
                sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:5]
                fig, ax = dark_fig()
                names  = [p[0][:14] for p in sorted_probs]
                values = [p[1]*100 for p in sorted_probs]
                colors = ["#6366f1" if i==0 else "#334155" for i in range(len(names))]
                bars   = ax.barh(names[::-1], values[::-1], color=colors[::-1], height=0.55)
                for bar, val in zip(bars, values[::-1]):
                    ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                            f"{val:.1f}%", va="center", color="#94a3b8", fontsize=9)
                ax.set_xlim(0, 115)
                ax.set_xlabel("Probability (%)")
                ax.set_title("Confidence per Class", fontsize=11)
                ax.grid(axis="x", color="#1e293b", linewidth=0.7)
                fig.patch.set_facecolor("#111827")
                ax.set_facecolor("#111827")
                st.pyplot(fig, use_container_width=True)
                plt.close()

            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.markdown("""
            <div style='text-align:center; padding:3rem 1rem;
                        color:#334155; border: 1px dashed #1e293b;
                        border-radius:var(--radius); font-size:.95rem;'>
              📂  Upload an image or select a sample to see the prediction
            </div>""", unsafe_allow_html=True)

    # ── Preprocessing demo ────────────────────────────────
    if uploaded is not None:
        st.markdown("<div class='section-title'>Preprocessing Visualisation</div>", unsafe_allow_html=True)
        try:
            if isinstance(uploaded, Path):
                raw_img = Image.open(uploaded)
            else:
                uploaded.seek(0)
                raw_img = Image.open(uploaded)

            grey  = raw_img.convert("L")
            resized = grey.resize(IMG_SIZE)
            arr = np.array(resized, dtype=np.float32) / 255.0

            fig2, axes = plt.subplots(1, 3, figsize=(10, 3), facecolor="#111827")
            panels = [
                (np.array(raw_img.convert("RGB")), "1. Original",      None),
                (np.array(grey),                   "2. Greyscale",     "gray"),
                (arr,                               f"3. Resized {IMG_SIZE}", "gray"),
            ]
            for ax_i, (data, title, cmap) in zip(axes, panels):
                if cmap:
                    ax_i.imshow(data, cmap=cmap)
                else:
                    ax_i.imshow(data)
                ax_i.set_title(title, color="#f1f5f9", fontsize=10)
                ax_i.axis("off")
                ax_i.set_facecolor("#111827")

            plt.tight_layout(pad=0.5)
            fig2.patch.set_facecolor("#111827")
            st.pyplot(fig2, use_container_width=True)
            plt.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET PREVIEW
# ══════════════════════════════════════════════════════════════════════════════
def page_dataset():
    st.markdown("<div class='hero-banner'><div class='hero-title'>🖼️ Dataset Preview</div><div class='hero-sub'>LeapGestRecog — 10 hand gesture classes · near-infrared imaging</div></div>", unsafe_allow_html=True)

    GESTURE_INFO = {
        "01_palm":       ("Palm",         "Open flat hand facing camera."),
        "02_l":          ("L Shape",      "Index and thumb extended at 90°."),
        "03_fist":       ("Fist",         "All fingers curled into a fist."),
        "04_fist_moved": ("Fist (Moved)", "Fist captured mid-movement."),
        "05_thumb":      ("Thumb Up",     "Thumb extended upward."),
        "06_index":      ("Index Finger", "Only index finger extended."),
        "07_ok":         ("OK Sign",      "Thumb and index touching."),
        "08_palm_moved": ("Palm (Moved)", "Open palm in motion."),
        "09_c":          ("C Shape",      "Hand curved like the letter C."),
        "10_down":       ("Down",         "Hand pointing/pressing down."),
    }

    # ── Dataset stats ──────────────────────────────────────
    st.markdown("<div class='section-title'>Dataset Statistics</div>", unsafe_allow_html=True)
    metrics = load_metrics()
    ds1, ds2, ds3, ds4 = st.columns(4)
    total = metrics["n_train"]+metrics["n_test"] if metrics else "—"
    with ds1: metric_card("🗂️","Total Images",  str(total),  "#a5b4fc")
    with ds2: metric_card("🏷️","Gesture Classes","10",       "#6ee7b7")
    with ds3: metric_card("📐","Image Size",    "64×64 px",  "#67e8f9")
    with ds4: metric_card("🎨","Colour Mode",   "Greyscale", "#fbbf24")

    # ── Sample images grid ────────────────────────────────
    st.markdown("<div class='section-title'>Sample Images per Class</div>", unsafe_allow_html=True)
    samples = {p.stem: p for p in SAMPLE_DIR.glob("*.png")} if SAMPLE_DIR.exists() else {}

    cols_per_row = 5
    items = list(GESTURE_INFO.items())
    for row_start in range(0, len(items), cols_per_row):
        row_items = items[row_start:row_start+cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (key, (name, desc)) in zip(cols, row_items):
            with col:
                if key in samples:
                    img = Image.open(samples[key]).convert("RGB")
                    st.image(img, use_column_width=True)
                else:
                    st.markdown("""
                    <div style='background:#111827; border:1px dashed #334155;
                                border-radius:8px; height:110px; display:flex;
                                align-items:center; justify-content:center;
                                color:#475569; font-size:1.5rem;'>✋</div>
                    """, unsafe_allow_html=True)
                st.markdown(f"""
                <div style='text-align:center; font-size:.78rem; font-weight:600;
                            color:#a5b4fc; margin:.3rem 0 .1rem;'>{name}</div>
                <div style='text-align:center; font-size:.7rem; color:#64748b;
                            line-height:1.3;'>{desc}</div>
                """, unsafe_allow_html=True)

    # ── Preprocessing pipeline explanation ────────────────
    st.markdown("<div class='section-title'>Preprocessing Steps</div>", unsafe_allow_html=True)
    steps = [
        ("1️⃣","Load Image","PIL.Image.open() reads any standard image format (PNG/JPG)."),
        ("2️⃣","Greyscale Conversion","convert('L') removes colour channels → single channel."),
        ("3️⃣","Resize","Resize to 64×64 pixels for a fixed 4 096-feature vector."),
        ("4️⃣","Flatten","numpy.flatten() converts 2-D array → 1-D feature row."),
        ("5️⃣","Standardise","StandardScaler: zero mean, unit variance per feature."),
        ("6️⃣","Classify","Random Forest predicts class; predict_proba gives confidence."),
    ]
    pc1, pc2 = st.columns(2)
    for i, (num, title, desc) in enumerate(steps):
        col = pc1 if i % 2 == 0 else pc2
        with col:
            st.markdown(f"""
            <div style='background:var(--bg-card); border:1px solid var(--border);
                        border-radius:10px; padding:.85rem 1rem; margin-bottom:.6rem;
                        display:flex; align-items:flex-start; gap:.75rem;'>
              <span style='font-size:1.3rem;'>{num}</span>
              <div>
                <div style='font-weight:600; color:#a5b4fc; font-size:.88rem;'>{title}</div>
                <div style='color:#94a3b8; font-size:.8rem; margin-top:.2rem;'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Class distribution bar ─────────────────────────────
    if metrics and "class_counts" in metrics:
        st.markdown("<div class='section-title'>Class Distribution</div>", unsafe_allow_html=True)
        cc = metrics["class_counts"]
        fig, ax = dark_fig()
        colors = plt.cm.cool(np.linspace(0.3, 0.9, len(cc)))
        bars = ax.bar(list(cc.keys()), list(cc.values()), color=colors, width=0.6)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    str(int(bar.get_height())), ha="center", va="bottom",
                    color="#f1f5f9", fontsize=9)
        ax.set_xlabel("Gesture Class")
        ax.set_ylabel("Count")
        ax.set_title("Number of Samples per Class", fontsize=13)
        ax.grid(axis="y", color="#1e293b", linewidth=0.7)
        plt.xticks(rotation=35, ha="right", fontsize=8)
        ax.set_facecolor("#111827")
        fig.patch.set_facecolor("#111827")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
def page_about():
    st.markdown("<div class='hero-banner'><div class='hero-title'>ℹ️ About This Project</div><div class='hero-sub'>Architecture · Dataset · Credits · How to run</div></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='section-title'>Project Architecture</div>

    <div class='info-box'>
    <pre style='background:transparent; color:#c7d2fe; font-family:JetBrains Mono,monospace;
                font-size:.8rem; line-height:1.6; margin:0; white-space:pre;'>
hand_gesture_recognition/
├── app.py                  # Streamlit dashboard (this file)
├── train_model.py          # Data download, training & evaluation
├── requirements.txt        # Python dependencies
├── README.md               # Full documentation
├── models/
│   ├── gesture_model.pkl   # Trained Random Forest
│   ├── label_encoder.pkl   # LabelEncoder
│   ├── scaler.pkl          # StandardScaler
│   ├── metrics.json        # Accuracy / Precision / Recall / F1
│   └── confusion_matrix.png
├── utils/
│   ├── __init__.py
│   └── predict.py          # Inference helpers
├── sample_images/          # One image per gesture class
└── dataset/                # LeapGestRecog images (auto-downloaded)
    </pre>
    </div>

    <div class='section-title'>Dataset — LeapGestRecog</div>
    <div style='background:var(--bg-card); border:1px solid var(--border);
                border-radius:var(--radius); padding:1.2rem 1.5rem; font-size:.88rem;
                color:#94a3b8; line-height:1.7;'>
      <b style='color:#f1f5f9;'>Source:</b> Kaggle — gti-upm/leapgestrecog<br>
      <b style='color:#f1f5f9;'>Sensor:</b> Leap Motion Controller (near-infrared)<br>
      <b style='color:#f1f5f9;'>Subjects:</b> 10 participants<br>
      <b style='color:#f1f5f9;'>Classes:</b> 10 distinct hand gestures<br>
      <b style='color:#f1f5f9;'>Format:</b> Greyscale PNG images<br>
      <b style='color:#f1f5f9;'>Licence:</b> CC0 — Public Domain<br>
    </div>

    <div class='section-title'>How to Run</div>
    <div style='background:#0a0e1a; border:1px solid #1e293b; border-radius:10px;
                padding:1rem 1.2rem; font-family:JetBrains Mono,monospace;
                font-size:.82rem; color:#6ee7b7; line-height:2;'>
      # 1. Install dependencies<br>
      pip install -r requirements.txt<br><br>
      # 2. (Optional) set up Kaggle API credentials<br>
      #    https://www.kaggle.com/docs/api<br><br>
      # 3. Train the model (downloads dataset automatically)<br>
      python train_model.py<br><br>
      # 4. Launch the Streamlit app<br>
      streamlit run app.py
    </div>

    <div class='section-title'>Key Concepts</div>
    """, unsafe_allow_html=True)

    concepts = [
        ("🌲","Random Forest","An ensemble of decision trees; each tree votes on the class. Robust to noise, fast to train, and provides class probabilities."),
        ("📐","Image Preprocessing","Greyscale conversion removes colour noise; resizing to 64×64 gives a fixed 4 096-feature vector; StandardScaler normalises them."),
        ("✂️","Train/Test Split","80 % of data trains the model; the remaining 20 % (unseen) evaluates real-world performance."),
        ("📊","Evaluation Metrics","Accuracy = correct predictions / total. Precision = TP/(TP+FP). Recall = TP/(TP+FN). F1 = harmonic mean of P & R."),
        ("🗂️","Confusion Matrix","Grid where row = true class, column = predicted class; diagonal shows correct predictions."),
        ("💾","Model Persistence","joblib.dump/load serialises the trained model, encoder and scaler so the app doesn't retrain on every run."),
    ]
    for i in range(0, len(concepts), 2):
        cc1, cc2 = st.columns(2)
        for col, (icon, title, desc) in zip([cc1,cc2], concepts[i:i+2]):
            with col:
                st.markdown(f"""
                <div style='background:var(--bg-card); border:1px solid var(--border);
                            border-radius:10px; padding:1rem; margin-bottom:.6rem;'>
                  <div style='font-size:1.4rem; margin-bottom:.3rem;'>{icon}</div>
                  <div style='font-weight:600; color:#a5b4fc; margin-bottom:.3rem;'>{title}</div>
                  <div style='color:#94a3b8; font-size:.83rem; line-height:1.5;'>{desc}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; color:#334155; font-size:.78rem; margin-top:2rem;'>
      Built with ❤️ using Python · Scikit-learn · Streamlit · Matplotlib · Seaborn
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if   nav == "🏠 Dashboard":       page_dashboard()
elif nav == "📊 Model Analytics":  page_analytics()
elif nav == "🔮 Predict Gesture":  page_predict()
elif nav == "🖼️ Dataset Preview":  page_dataset()
elif nav == "ℹ️ About":            page_about()