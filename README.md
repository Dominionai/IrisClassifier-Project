# 🌸 Iris Classifier — Data Classification Using AI

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

> **Live Demo:** [irisclassifier-project.streamlit.app](https://irisclassifier-project.streamlit.app/)

A production‑ready machine learning application that classifies Iris flower species using the **K‑Nearest Neighbors (KNN)** algorithm. Built with a clean IPO (Input → Process → Output) architecture and deployed on Streamlit Cloud.

**Built independently by Egwuatu Chibuike Dominion | DecodeLabs Batch 2026 | Project 2**

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Model Performance](#-model-performance)
- [Architecture](#-architecture-ipo-framework)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation-local)
- [Project Structure](#-project-structure)
- [Key ML Concepts](#-key-ml-concepts-demonstrated)
- [Deployment](#-deployment)
- [Author](#-author)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 📌 Overview

This project is **Project 2** of the DecodeLabs Batch 2026 curriculum. It demonstrates a complete ML workflow:

- **Data** — The classic Iris dataset (150 samples, 3 classes, 4 features)
- **Algorithm** — KNN with Euclidean distance & majority voting
- **Scaling** — StandardScaler (mean=0, variance=1) — the *Gatekeeper Rule*
- **Evaluation** — F1 Score (macro) + Confusion Matrix + Classification Report
- **Interface** — Interactive web app built with Streamlit

| Species | Setosa | Versicolor | Virginica |
|---------|--------|------------|-----------|
| Emoji   | 🌷     | 🌼         | 🌺        |

---

## 🌐 Live Demo

**Try it yourself:** [irisclassifier-project.streamlit.app](https://irisclassifier-project.streamlit.app/)

No installation required — works on any device with a browser.

---

## 🚀 Features

| Tab | Functionality |
|-----|---------------|
| 🔮 **Predict** | Live classification with probability distribution |
| 📊 **Results** | Confusion matrix + F1 score per class + classification report |
| 📚 **Dataset** | Raw data explorer + feature statistics + scatter plot |
| ⚙️ **Tune K** | Elbow curve visualisation + interactive K tuning |

### 🔮 Interactive Prediction
Slide the 4 feature sliders → click **Classify** → get species prediction + confidence score + probability bar chart.

### ⚙️ Model Tuning
Adjust K from 1–20 — see accuracy/F1 curves update in real time. Choose between:
- **K ≤ 2** → Overfitting (sensitive to noise)
- **K ≥ 15** → Underfitting (too generic)
- **K = 5** → Sweet spot (balanced performance)

---

## 📊 Model Performance (Test Set — 20% Holdout)

| Metric | Score |
|--------|-------|
| **Accuracy** | ~96% |
| **F1 Score (macro)** | ~96% |
| **Confusion Matrix** | All off‑diagonals = 0–1 misclassifications |

### Per-Class Metrics

| Class | Precision | Recall | F1‑Score | Support |
|-------|-----------|--------|----------|---------|
| setosa | 1.00 | 1.00 | 1.00 | 10 |
| versicolor | 0.93 | 1.00 | 0.96 | 10 |
| virginica | 1.00 | 0.90 | 0.95 | 10 |

### Confusion Matrix

[[10 0 0]
[ 0 10 0]
[ 0 1 9]]


---

## 🧠 Architecture: IPO Framework

┌─────────────────────────────────────────────────────────────────┐
│ 📥 INPUT │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │ Load │ │ Split │ │ Scale │ │
│ │ Iris │ → │ 80/20 │ → │ Standard │ │
│ │ Dataset │ │ Stratify │ │ Scaler │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ ⚙️ PROCESS │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ K-Nearest Neighbors (K=5) │ │
│ │ "Similar things exist in close proximity" │ │
│ └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ 📤 OUTPUT │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │ Accuracy │ │ F1 Score │ │ Confusion │ │
│ │ ~96% │ │ ~96% │ │ Matrix │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘


---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Streamlit |
| ML Pipeline | Scikit‑learn |
| Visualisation | Matplotlib, Seaborn |
| Data Handling | NumPy, Pandas |
| Deployment | Streamlit Cloud |

---

## 📦 Installation (Local)

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/iris-classifier.git
cd iris-classifier

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py

```
Dependencies (requirements.txt)

streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

Terminal Mode (CLI)
```bash
python classifier.py

Outputs full pipeline results — accuracy, F1, confusion matrix, classification report directly in the terminal.
```
📂 Project Structure
```text
iris-classifier/
│
├── app.py                 # Streamlit application (UI + visualisations)
├── classifier.py          # ML pipeline core (KNN + StandardScaler + metrics)
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
└── .streamlit/
    └── config.toml        # (optional) Streamlit theming
```
File Descriptions

File	Purpose
app.py	Streamlit frontend — all UI components, tabs, charts, and user interaction
classifier.py	ML backend — data loading, scaling, KNN training, evaluation metrics
requirements.txt	Project dependencies for pip installation

🧪 Key ML Concepts Demonstrated
Concept	Implementation
IPO Framework	Clean separation of Input → Process → Output
Data Leakage Prevention	fit_transform() on train, transform() on test
Stratified Split	Maintains class balance (50/50/50 across train/test)
KNN Proximity Principle	Euclidean distance + majority vote
Beyond Accuracy	F1 Score (macro) — treats all classes equally
Confusion Matrix	TP/TN/FP/FN visualisation
Elbow Method	K‑value selection via accuracy/F1 curves
Probability Calibration	predict_proba() outputs class confidence

🌐 Deployment
The app is deployed on Streamlit Cloud at irisclassifier-project.streamlit.app

Deployment Steps (for your own fork)
Push code to GitHub repository

Go to share.streamlit.io

Click "New app" → select your repository

Set app.py as the main file

Click "Deploy"

The app will automatically redeploy on every git push to the main branch.

👨‍💻 Author
Egwuatu Chibuike Dominion
DecodeLabs Batch 2026 — Project 2

Portfolio: your-portfolio.com

GitHub: @yourusername

LinkedIn: yourprofile

📝 License
MIT License — free for educational and commercial use. Attribution appreciated but not required.

🙏 Acknowledgements
DecodeLabs — Curriculum, mentorship, and project structure

Scikit‑learn — Machine learning toolkit

Streamlit — Rapid application framework

Ronald Fisher — Creator of the Iris dataset (1936)

Open source community — For all the tools that made this possible

📧 Contact
For questions, feedback, or collaboration opportunities:

Email: your.email@example.com

GitHub Issues: Open an issue

<div align="center">
Built independently as part of DecodeLabs Batch 2026 — Data Classification Using AI

"Similar things exist in close proximity" — The KNN Principle

⭐ Star this repo if you found it useful! ⭐

</div> ```
