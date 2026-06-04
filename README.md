# 🌸 Iris Classifier — Production-Ready Machine Learning App

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=Streamlit\&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)

> 🚀 **Live App:** https://irisclassifier-project.streamlit.app/
> 💡 **Core Idea:** *“Similar things exist in close proximity” — KNN Principle*

---

## 📌 Quick Navigation

* [🔍 Overview](#-overview)
* [🌐 Live Demo](#-live-demo)
* [🚀 Features](#-features)
* [📊 Model Performance](#-model-performance)
* [🧠 Architecture (IPO)](#-architecture-ipo-framework)
* [🛠 Tech Stack](#-tech-stack)
* [📦 Installation](#-installation-local)
* [📂 Project Structure](#-project-structure)
* [🧪 ML Concepts](#-key-ml-concepts-demonstrated)
* [🌍 Deployment](#-deployment)
* [👨‍💻 Author](#-author)
* [📧 Contact](#-contact)

---

## 🔍 Overview

This project delivers a **complete end-to-end machine learning pipeline**, from data preprocessing to deployment.

### ✅ What makes this project stand out:

* Clean **IPO Architecture (Input → Process → Output)**
* Proper **data scaling (StandardScaler)** — *prevents bias*
* Strong evaluation using **F1 Score (macro)** — *not just accuracy*
* Interactive **Streamlit UI for real-time predictions**
* Fully deployed and accessible online

### 📊 Dataset Summary

| Feature Count | Samples | Classes |
| ------------- | ------- | ------- |
| 4             | 150     | 3       |

| Species | Setosa | Versicolor | Virginica |
| ------- | ------ | ---------- | --------- |
| Emoji   | 🌷     | 🌼         | 🌺        |

---

## 🌐 Live Demo

👉 **Try it instantly (no installation required):**
https://irisclassifier-project.streamlit.app/

---

## 🚀 Features

### 🔮 Predict (Core Feature)

* Real-time classification using slider inputs
* Displays:

  * ✅ Predicted species
  * 📊 Probability distribution
  * 🎯 Confidence score

---

### 📊 Results Dashboard

* Confusion Matrix visualization
* Per-class F1 scores
* Full classification report

---

### 📚 Dataset Explorer

* Raw dataset view
* Feature statistics
* Scatter plot visualization

---

### ⚙️ Model Tuning (Advanced)

* Interactive **K selection (1–20)**
* Visual elbow curve

| K Value | Behavior        |
| ------- | --------------- |
| ≤ 2     | ⚠️ Overfitting  |
| 5       | ✅ Optimal       |
| ≥ 15    | ⚠️ Underfitting |

---

## 📊 Model Performance

### 🔥 Test Results (20% Holdout)

| Metric               | Score |
| -------------------- | ----- |
| **Accuracy**         | ~96%  |
| **F1 Score (Macro)** | ~96%  |

---

### 📌 Per-Class Metrics

| Class      | Precision | Recall | F1   |
| ---------- | --------- | ------ | ---- |
| Setosa     | 1.00      | 1.00   | 1.00 |
| Versicolor | 0.93      | 1.00   | 0.96 |
| Virginica  | 1.00      | 0.90   | 0.95 |

---

### 📉 Confusion Matrix

```
[[10 0 0]
 [ 0 10 0]
 [ 0  1 9]]
```

---

## 🧠 Architecture (IPO Framework)

```
INPUT → PROCESS → OUTPUT

📥 INPUT
- Load Dataset
- Train/Test Split (80/20, stratified)
- Feature Scaling (StandardScaler)

⚙️ PROCESS
- KNN Algorithm (K=5)
- Euclidean Distance
- Majority Voting

📤 OUTPUT
- Predictions
- Accuracy & F1 Score
- Confusion Matrix
```

---

## 🛠 Tech Stack

| Layer            | Technology          |
| ---------------- | ------------------- |
| Frontend         | Streamlit           |
| Machine Learning | Scikit-learn        |
| Data Handling    | Pandas, NumPy       |
| Visualization    | Matplotlib, Seaborn |
| Deployment       | Streamlit Cloud     |

---

## 📦 Installation (Local)

### 🔧 Prerequisites

* Python 3.8+
* pip

---

### ⚡ Setup

```bash
# Clone repository
git clone https://github.com/yourusername/iris-classifier.git
cd iris-classifier

# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

---

### 🖥 CLI Mode

```bash
python classifier.py
```

Outputs:

* Accuracy
* F1 Score
* Confusion Matrix
* Classification Report

---

## 📂 Project Structure

```
iris-classifier/
│
├── app.py              # UI (Streamlit frontend)
├── classifier.py       # ML pipeline
├── requirements.txt    # Dependencies
├── README.md
│
└── .streamlit/
    └── config.toml
```

---

## 🧪 Key ML Concepts Demonstrated

| Concept                 | Implementation                |
| ----------------------- | ----------------------------- |
| IPO Framework           | Structured pipeline design    |
| Data Leakage Prevention | Train/Test separation         |
| Feature Scaling         | StandardScaler                |
| KNN Principle           | Distance-based classification |
| Evaluation              | F1 Score (macro)              |
| Model Tuning            | Elbow method                  |
| Probabilities           | predict_proba()               |

---

## 🌍 Deployment

### 🚀 Streamlit Cloud

1. Push project to GitHub
2. Go to https://share.streamlit.io
3. Click **New App**
4. Select repo & set `app.py`
5. Deploy

✅ Auto-updates on every push

---

## 👨‍💻 Author

**Egwuatu Chibuike Dominion**
DecodeLabs Batch 2026 — Project 2

* 💻 AI Engineer
* 🎯 Focus: AI-powered applications

---

## 📧 Contact

* Email: **[chibuikedominion7@gmail.com](mailto:chibuikedominion7@gmail.com)**
* GitHub: [yourusername](https://github.com/Dominionai/IrisClassifier-Project/edit/main/README.md)
* LinkedIn: [yourprofile](https://www.linkedin.com/in/chibuikedominion/)

---

## 📝 License

MIT License — free for personal and commercial use.

---

## 🙏 Acknowledgements

* DecodeLabs — mentorship & curriculum
* Scikit-learn — ML framework
* Streamlit — rapid UI development
* Ronald Fisher — Iris dataset

---

<div align="center">

⭐ **If you found this project valuable, consider starring the repo!** ⭐

</div>
