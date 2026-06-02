# =============================================================================
# DecodeLabs | Batch 2026 | Project 2: Data Classification Using AI
# Built independently by Egwuatu Chibuike Dominion
# File: app.py — Streamlit Frontend
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from classifier import run_pipeline, predict_single

st.set_page_config(
    page_title="IrisClassifier | DecodeLabs Batch 2026",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# STYLING
# =============================================================================

st.markdown("""
<style>
    * { box-sizing: border-box; }
    .stApp {
        background: linear-gradient(135deg, #05090f 0%, #07111e 50%, #05090f 100%);
        font-family: 'JetBrains Mono', 'Courier New', monospace;
    }
    .block-container {
        max-width: 860px !important;
        padding: 0.8rem 1.5rem 2rem !important;
    }

    /* Header */
    .page-header { text-align:center; padding: 1rem 0 0.6rem 0; }
    .page-header h1 {
        color: #f97316;
        font-size: clamp(1.4rem, 4vw, 2rem);
        font-weight: 700; letter-spacing: 2px;
        text-shadow: 0 0 24px rgba(249,115,22,0.4);
        margin: 0;
    }
    .page-header .sub {
        color: #3b82f6; font-size: clamp(0.55rem, 1.4vw, 0.7rem);
        letter-spacing: 2px; margin: 4px 0 2px 0;
    }
    .page-header .credit {
        color: rgba(249,115,22,0.4);
        font-size: clamp(0.5rem, 1.2vw, 0.6rem);
        letter-spacing: 1px; margin: 0;
    }

    /* Metric cards */
    .metric-row {
        display: flex; gap: 0.6rem; margin: 0.8rem 0; flex-wrap: wrap;
    }
    .metric-card {
        flex: 1; min-width: 100px;
        background: rgba(59,130,246,0.07);
        border: 1px solid rgba(59,130,246,0.22);
        border-radius: 10px; padding: 10px 12px; text-align: center;
    }
    .metric-card .val {
        color: #f97316; font-size: 1.4rem; font-weight: 700;
        display: block; line-height: 1.1;
    }
    .metric-card .lbl {
        color: rgba(180,210,240,0.5);
        font-size: 0.58rem; letter-spacing: 1px;
        display: block; margin-top: 3px;
    }
    .metric-card.green .val { color: #22c55e; }
    .metric-card.blue  .val { color: #60a5fa; }

    /* Section headers */
    .section-title {
        color: #f97316; font-size: 0.78rem; font-weight: 700;
        letter-spacing: 2px; margin: 1.2rem 0 0.6rem 0;
        border-bottom: 1px solid rgba(249,115,22,0.2);
        padding-bottom: 5px;
    }

    /* Prediction result box */
    .pred-box {
        background: rgba(34,197,94,0.08);
        border: 1.5px solid rgba(34,197,94,0.3);
        border-radius: 12px; padding: 1.2rem 1.5rem;
        text-align: center; margin: 0.8rem 0;
    }
    .pred-box .species {
        color: #22c55e; font-size: 1.6rem; font-weight: 700;
        letter-spacing: 2px; display: block;
    }
    .pred-box .confidence {
        color: rgba(180,240,200,0.6); font-size: 0.68rem;
        letter-spacing: 1px; margin-top: 4px; display: block;
    }

    /* Pipeline steps */
    .pipeline {
        display: flex; align-items: center;
        justify-content: center; gap: 0.4rem;
        flex-wrap: wrap; margin: 0.6rem 0 1rem 0;
    }
    .pipe-step {
        background: rgba(59,130,246,0.08);
        border: 1px solid rgba(59,130,246,0.22);
        border-radius: 8px; padding: 6px 14px;
        font-size: 0.66rem; color: #60a5fa; letter-spacing: 1px;
        white-space: nowrap;
    }
    .pipe-arrow { color: rgba(249,115,22,0.6); font-size: 1rem; }
    .pipe-step.highlight {
        background: rgba(249,115,22,0.1);
        border-color: rgba(249,115,22,0.35); color: #f97316;
    }

    /* Feature input labels */
    .stSlider label { color: #94a3b8 !important; font-size: 0.72rem !important; }

    /* Expanders */
    [data-testid="stExpander"] {
        background: rgba(59,130,246,0.04) !important;
        border: 1px solid rgba(59,130,246,0.18) !important;
        border-radius: 10px !important; margin-bottom: 8px !important;
    }
    [data-testid="stExpander"] summary {
        color: #60a5fa !important; font-size: 0.74rem !important;
        font-weight: 700 !important; letter-spacing: 0.5px !important;
    }
    [data-testid="stExpander"] summary:hover { color: #93c5fd !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(59,130,246,0.06) !important;
        border-radius: 8px !important; gap: 4px !important;
        padding: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: rgba(148,163,184,0.7) !important;
        font-size: 0.72rem !important; letter-spacing: 0.5px !important;
        border-radius: 6px !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(249,115,22,0.15) !important;
        color: #f97316 !important;
    }

    /* Buttons */
    .stButton > button {
        background: rgba(249,115,22,0.1) !important;
        border: 1.5px solid #f97316 !important;
        color: #f97316 !important;
        border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important; letter-spacing: 1px !important;
        height: 44px !important; transition: all 0.18s !important;
    }
    .stButton > button:hover {
        background: rgba(249,115,22,0.22) !important;
        box-shadow: 0 0 14px rgba(249,115,22,0.3) !important;
    }

    /* Tables */
    .stDataFrame { border-radius: 8px !important; }

    /* Mobile */
    @media (max-width: 640px) {
        .block-container { padding: 0.5rem 0.7rem 1.5rem !important; }
        .metric-card .val { font-size: 1.15rem !important; }
        .pipeline { gap: 0.25rem !important; }
        .pipe-step { font-size: 0.6rem !important; padding: 5px 9px !important; }
    }

    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-thumb { background: rgba(249,115,22,0.2); border-radius: 4px; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# LOAD PIPELINE — cached so it doesn't retrain on every interaction
# =============================================================================

@st.cache_resource
def get_pipeline(k=5):
    return run_pipeline(k=k)


# =============================================================================
# SESSION STATE
# =============================================================================

if "k_value" not in st.session_state:
    st.session_state.k_value = 5
if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False


# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div class="page-header">
    <h1>🌸 IRIS CLASSIFIER</h1>
    <p class="sub">DATA CLASSIFICATION USING AI &nbsp;|&nbsp; PROJECT 2 &nbsp;|&nbsp; DECODELABS BATCH 2026</p>
    <p class="credit">Built independently by Egwuatu Chibuike Dominion &nbsp;·&nbsp; KNN + Scikit-Learn</p>
</div>
""", unsafe_allow_html=True)

# Pipeline diagram
st.markdown("""
<div class="pipeline">
    <div class="pipe-step highlight">📊 IRIS DATASET</div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step">⚖️ STANDARDSCALER</div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step">✂️ 80/20 SPLIT</div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step highlight">🧠 KNN (K=5)</div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-step">📋 F1 SCORE</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# LOAD MODEL
# =============================================================================

pipeline = get_pipeline(st.session_state.k_value)
r        = pipeline["results"]
tn       = pipeline["target_names"]
fn       = pipeline["feature_names"]

# Clean feature names
fn_clean = [f.replace(" (cm)", "").title() for f in fn]


# =============================================================================
# TOP METRICS
# =============================================================================

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <span class="val">150</span>
        <span class="lbl">TOTAL SAMPLES</span>
    </div>
    <div class="metric-card blue">
        <span class="val">3</span>
        <span class="lbl">CLASSES</span>
    </div>
    <div class="metric-card blue">
        <span class="val">4</span>
        <span class="lbl">FEATURES</span>
    </div>
    <div class="metric-card">
        <span class="val">{pipeline['train_size']}</span>
        <span class="lbl">TRAIN (80%)</span>
    </div>
    <div class="metric-card">
        <span class="val">{pipeline['test_size']}</span>
        <span class="lbl">TEST (20%)</span>
    </div>
    <div class="metric-card green">
        <span class="val">{r['accuracy']*100:.1f}%</span>
        <span class="lbl">ACCURACY</span>
    </div>
    <div class="metric-card green">
        <span class="val">{r['f1_macro']*100:.1f}%</span>
        <span class="lbl">F1 SCORE</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# MAIN TABS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮  PREDICT",
    "📊  RESULTS",
    "📚  DATASET",
    "⚙️  TUNE K",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — LIVE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown('<p class="section-title">🌸 PREDICT IRIS SPECIES</p>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:rgba(148,163,184,0.55); font-size:0.7rem; margin-bottom:12px'>"
        "Enter the flower measurements below and click Classify to see the prediction.</p>",
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    with c1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
        sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
    with c2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.4, 0.1)
        petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 1.3, 0.1)

    features = [sepal_length, sepal_width, petal_length, petal_width]

    if st.button("🔮  CLASSIFY FLOWER", use_container_width=True):
        species, proba = predict_single(
            pipeline["model"], pipeline["scaler"], features, tn
        )
        st.session_state.prediction_made = True
        st.session_state.last_species    = species
        st.session_state.last_proba      = proba
        st.session_state.last_features   = features

    if st.session_state.prediction_made:
        sp    = st.session_state.last_species
        proba = st.session_state.last_proba
        conf  = max(proba) * 100

        emoji_map = {"setosa": "🌷", "versicolor": "🌼", "virginica": "🌺"}
        emoji = emoji_map.get(sp.lower(), "🌸")

        st.markdown(f"""
        <div class="pred-box">
            <span class="species">{emoji} Iris {sp.capitalize()}</span>
            <span class="confidence">Confidence: {conf:.1f}%  &nbsp;·&nbsp;  K={st.session_state.k_value} nearest neighbours</span>
        </div>
        """, unsafe_allow_html=True)

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(6, 2.2))
        fig.patch.set_facecolor("#07111e")
        ax.set_facecolor("#07111e")
        colors = ["#f97316" if t == sp.lower() else "#1e3a5f" for t in tn]
        bars = ax.barh(tn, proba * 100, color=colors, height=0.5)
        ax.set_xlabel("Probability (%)", color="#94a3b8", fontsize=8)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e3a5f")
        for bar, p in zip(bars, proba):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{p*100:.1f}%", va="center", color="#94a3b8", fontsize=7)
        ax.set_xlim(0, 110)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — RESULTS: Confusion Matrix + F1 + Report
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.markdown('<p class="section-title">📊 MODEL EVALUATION RESULTS</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            "<p style='color:#60a5fa; font-size:0.7rem; font-weight:700; margin-bottom:6px'>"
            "CONFUSION MATRIX</p>",
            unsafe_allow_html=True
        )
        cm = r["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(4, 3.2))
        fig.patch.set_facecolor("#07111e")
        ax.set_facecolor("#07111e")
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="YlOrBr",
            xticklabels=tn, yticklabels=tn,
            ax=ax, linewidths=0.5, linecolor="#1e3a5f",
            annot_kws={"size": 11, "weight": "bold", "color": "#0a0f1a"}
        )
        ax.set_xlabel("Predicted",  color="#94a3b8", fontsize=8)
        ax.set_ylabel("Actual",     color="#94a3b8", fontsize=8)
        ax.tick_params(colors="#94a3b8", labelsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # TP / FP / FN / TN legend
        st.markdown("""
        <div style='font-size:0.62rem; color:rgba(148,163,184,0.55); margin-top:4px'>
            ✅ Diagonal = Correct predictions (TP / TN)<br>
            ⚠️ Off-diagonal = Errors (FP = False Alarm | FN = Missed)
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(
            "<p style='color:#60a5fa; font-size:0.7rem; font-weight:700; margin-bottom:6px'>"
            "F1 SCORE PER CLASS</p>",
            unsafe_allow_html=True
        )
        f1_vals = r["f1_per_class"]
        fig, ax = plt.subplots(figsize=(4, 3.2))
        fig.patch.set_facecolor("#07111e")
        ax.set_facecolor("#07111e")
        bar_colors = ["#f97316", "#3b82f6", "#22c55e"]
        bars = ax.bar(tn, f1_vals * 100, color=bar_colors, width=0.5, zorder=3)
        ax.set_ylim(0, 115)
        ax.set_ylabel("F1 Score (%)", color="#94a3b8", fontsize=8)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.yaxis.grid(True, color="#1e3a5f", linewidth=0.5, zorder=0)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e3a5f")
        for bar, val in zip(bars, f1_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{val*100:.1f}%", ha="center", color="#e2e8f0", fontsize=8, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Macro F1 callout
        st.markdown(f"""
        <div style='background:rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.25);
        border-radius:8px; padding:10px 14px; margin-top:8px'>
            <p style='color:#22c55e; font-size:0.78rem; font-weight:700; margin:0'>
            Macro F1: {r["f1_macro"]*100:.2f}%</p>
            <p style='color:rgba(148,220,180,0.55); font-size:0.62rem; margin:3px 0 0 0'>
            Treats all 3 classes equally — no Accuracy Mirage</p>
        </div>
        """, unsafe_allow_html=True)

    # Classification report
    st.markdown('<p class="section-title">📋 FULL CLASSIFICATION REPORT</p>', unsafe_allow_html=True)
    lines = r["classification_report"].strip().split("\n")
    rows  = []
    for line in lines[2:-3]:
        parts = line.split()
        if len(parts) >= 5:
            rows.append({
                "Class":     parts[0],
                "Precision": float(parts[1]),
                "Recall":    float(parts[2]),
                "F1-Score":  float(parts[3]),
                "Support":   int(parts[4]),
            })
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.format({
                "Precision": "{:.2f}",
                "Recall":    "{:.2f}",
                "F1-Score":  "{:.2f}",
            }).background_gradient(subset=["F1-Score"], cmap="YlOrRd"),
            use_container_width=True, hide_index=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — DATASET EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown('<p class="section-title">📚 IRIS DATASET EXPLORER</p>', unsafe_allow_html=True)

    from sklearn.datasets import load_iris
    iris  = load_iris()
    df_full = pd.DataFrame(iris.data, columns=[f.replace(" (cm)", "") for f in iris.feature_names])
    df_full["species"] = [tn[t] for t in iris.target]

    # Class distribution
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            "<p style='color:#60a5fa; font-size:0.7rem; font-weight:700; margin-bottom:6px'>"
            "CLASS DISTRIBUTION</p>",
            unsafe_allow_html=True
        )
        fig, ax = plt.subplots(figsize=(3.5, 3))
        fig.patch.set_facecolor("#07111e")
        ax.set_facecolor("#07111e")
        counts  = [50, 50, 50]
        colors  = ["#f97316", "#3b82f6", "#22c55e"]
        wedges, texts, autotexts = ax.pie(
            counts, labels=tn, colors=colors,
            autopct="%1.0f%%", startangle=90,
            textprops={"color": "#e2e8f0", "fontsize": 8},
            wedgeprops={"linewidth": 1.5, "edgecolor": "#07111e"}
        )
        for at in autotexts:
            at.set_color("#0a0f1a")
            at.set_fontweight("bold")
        ax.set_title("Balanced — 50 per class", color="#94a3b8", fontsize=7, pad=6)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown(
            "<p style='color:#60a5fa; font-size:0.7rem; font-weight:700; margin-bottom:6px'>"
            "FEATURE STATISTICS</p>",
            unsafe_allow_html=True
        )
        stats = df_full.drop("species", axis=1).describe().round(2)
        st.dataframe(stats, use_container_width=True)

    # Scatter plot
    st.markdown(
        "<p style='color:#60a5fa; font-size:0.7rem; font-weight:700; margin:12px 0 6px 0'>"
        "PETAL LENGTH vs PETAL WIDTH — Decision Boundary Visualisation</p>",
        unsafe_allow_html=True
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#07111e")
    ax.set_facecolor("#07111e")
    colors_map = {"setosa": "#f97316", "versicolor": "#3b82f6", "virginica": "#22c55e"}
    for sp in tn:
        subset = df_full[df_full["species"] == sp]
        ax.scatter(
            subset["petal length"], subset["petal width"],
            c=colors_map[sp], label=sp.capitalize(),
            alpha=0.8, s=40, edgecolors="#07111e", linewidth=0.5
        )
    ax.set_xlabel("Petal Length (cm)", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Petal Width (cm)",  color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")
    ax.legend(facecolor="#0d1a2e", edgecolor="#1e3a5f",
              labelcolor="#e2e8f0", fontsize=8)
    ax.grid(True, color="#1e3a5f", linewidth=0.4, alpha=0.6)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Raw data preview
    with st.expander("🔍  VIEW RAW DATA (first 15 rows)", expanded=False):
        st.dataframe(df_full.head(15), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — K TUNING (Elbow Method)
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.markdown('<p class="section-title">⚙️ TUNE THE ENGINE: CHOOSING K</p>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:rgba(148,163,184,0.55); font-size:0.7rem; margin-bottom:12px'>"
        "K=1 overfits (noise). K=100 underfits (generic). The Elbow is the sweet spot.</p>",
        unsafe_allow_html=True
    )

    # Compute accuracy for K=1 to 20
    from sklearn.datasets import load_iris as _li
    from sklearn.model_selection import train_test_split as _tts
    from sklearn.preprocessing import StandardScaler as _ss
    from sklearn.neighbors import KNeighborsClassifier as _knn
    from sklearn.metrics import accuracy_score, f1_score

    @st.cache_data
    def compute_k_curve():
        iris   = _li()
        X, y   = iris.data, iris.target
        Xtr, Xte, ytr, yte = _tts(X, y, test_size=0.2, random_state=42, stratify=y)
        sc     = _ss(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        ks, accs, f1s = [], [], []
        for k in range(1, 21):
            m = _knn(n_neighbors=k).fit(Xtr, ytr)
            p = m.predict(Xte)
            ks.append(k)
            accs.append(accuracy_score(yte, p))
            f1s.append(f1_score(yte, p, average="macro"))
        return ks, accs, f1s

    ks, accs, f1s = compute_k_curve()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#07111e")
    ax.set_facecolor("#07111e")
    ax.plot(ks, [a*100 for a in accs], color="#3b82f6", linewidth=2,
            marker="o", markersize=5, label="Accuracy")
    ax.plot(ks, [f*100 for f in f1s],  color="#f97316", linewidth=2,
            marker="s", markersize=5, label="F1 Score (macro)")
    ax.axvline(x=st.session_state.k_value, color="#22c55e",
               linestyle="--", linewidth=1.5, alpha=0.8, label=f"Current K={st.session_state.k_value}")
    ax.set_xlabel("K Value", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Score (%)", color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")
    ax.legend(facecolor="#0d1a2e", edgecolor="#1e3a5f", labelcolor="#e2e8f0", fontsize=8)
    ax.grid(True, color="#1e3a5f", linewidth=0.4, alpha=0.6)
    ax.set_xticks(ks)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # K selector
    new_k = st.slider(
        "Select K value to retrain the model:",
        min_value=1, max_value=20,
        value=st.session_state.k_value, step=1
    )
    if st.button(f"🔄  RETRAIN WITH K = {new_k}", use_container_width=True):
        st.session_state.k_value = new_k
        st.session_state.prediction_made = False
        st.cache_resource.clear()
        st.rerun()

    k_acc = accs[new_k - 1] * 100
    k_f1  = f1s[new_k - 1]  * 100
    st.markdown(f"""
    <div style='background:rgba(59,130,246,0.07); border:1px solid rgba(59,130,246,0.2);
    border-radius:8px; padding:10px 14px; margin-top:8px; display:flex; gap:1.5rem; flex-wrap:wrap'>
        <div>
            <span style='color:#60a5fa; font-size:0.65rem; letter-spacing:1px'>K={new_k} ACCURACY</span><br>
            <span style='color:#f97316; font-size:1.2rem; font-weight:700'>{k_acc:.1f}%</span>
        </div>
        <div>
            <span style='color:#60a5fa; font-size:0.65rem; letter-spacing:1px'>K={new_k} F1 SCORE</span><br>
            <span style='color:#22c55e; font-size:1.2rem; font-weight:700'>{k_f1:.1f}%</span>
        </div>
        <div style='flex:1; min-width:160px'>
            <span style='color:#60a5fa; font-size:0.65rem; letter-spacing:1px'>DIAGNOSIS</span><br>
            <span style='color:#94a3b8; font-size:0.7rem'>
            {"⚠️ Overfitting risk — too sensitive to noise" if new_k <= 2
             else "⚠️ Underfitting risk — too generic" if new_k >= 15
             else "✅ Good balance — reliable predictions"}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("""
<div style='text-align:center; margin-top:1.5rem; padding:0.7rem;
border-top:1px solid rgba(249,115,22,0.08)'>
    <p style='color:rgba(255,255,255,0.1); font-size:0.57rem; letter-spacing:1.5px; margin:0'>
    BUILT BY EGWUATU CHIBUIKE DOMINION &nbsp;·&nbsp; DECODELABS BATCH 2026 &nbsp;·&nbsp; PROJECT 2
    </p>
    <p style='color:rgba(249,115,22,0.2); font-size:0.54rem; margin:3px 0 0 0'>
    KNN &nbsp;|&nbsp; StandardScaler &nbsp;|&nbsp; Confusion Matrix &nbsp;|&nbsp; F1 Score &nbsp;|&nbsp; Iris Dataset
    </p>
</div>
""", unsafe_allow_html=True)
