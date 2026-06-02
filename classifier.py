# =============================================================================
# DecodeLabs | Batch 2026 | Project 2: Data Classification Using AI
# Egwuatu Chibuike Dominion | Batch 2026
# File: classifier.py — The ML Pipeline (Core Brain)
# Architecture: IPO Framework | KNN | StandardScaler | Train-Test Split
# =============================================================================

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    classification_report,
)


# =============================================================================
# PHASE 1: INPUT — Load & understand the dataset
# The Iris Benchmark: 150 samples | 3 classes | 4 features
# =============================================================================

def load_data():
    """
    Load the Iris dataset.
    Returns:
        X      : feature matrix (150 x 4)
        y      : target labels  (150,)
        target_names : ['setosa', 'versicolor', 'virginica']
        feature_names: ['sepal length', 'sepal width', 'petal length', 'petal width']
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y, iris.target_names, iris.feature_names


# =============================================================================
# PHASE 2: PROCESS — The Gatekeeper Rule: Feature Scaling
# Raw data is biased; StandardScaler normalises to mean=0, variance=1
# =============================================================================

def scale_features(X_train, X_test):
    """
    Apply StandardScaler — fit on training data ONLY,
    then transform both sets.  Never fit on test data (data leakage).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # learn mean/var from train
    X_test_scaled  = scaler.transform(X_test)         # apply same scale to test
    return X_train_scaled, X_test_scaled, scaler


# =============================================================================
# PHASE 2: PROCESS — Train-Test Split (80 / 20 | shuffle=True)
# Randomise before splitting to remove order bias
# =============================================================================

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into 80% training (pattern recognition)
    and 20% test (validation — locked away until evaluation).
    shuffle=True removes order bias.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y          # keep class proportions balanced in both splits
    )


# =============================================================================
# PHASE 2: PROCESS — The Algorithm: K-Nearest Neighbors
# Proximity Principle: similar things exist in close proximity
# K=5  →  majority vote among 5 nearest neighbours
# =============================================================================

def train_model(X_train_scaled, y_train, k=5):
    """
    Instantiate, fit, and return a KNN classifier.
    Steps:
        INSTANTIATE  — build the frame
        FIT          — memorise the map (store training points)
        (PREDICT is called separately)
    """
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    return model


# =============================================================================
# PHASE 3: OUTPUT — Validation Metrics
# Accuracy alone is the "Accuracy Mirage" — we use F1 Score instead
# =============================================================================

def evaluate_model(model, X_test_scaled, y_test, target_names):
    """
    Evaluate model and return a results dictionary containing:
      - predictions
      - accuracy
      - f1_score  (macro — treats all classes equally)
      - confusion_matrix
      - classification_report
    """
    predictions = model.predict(X_test_scaled)

    results = {
        "predictions":          predictions,
        "accuracy":             accuracy_score(y_test, predictions),
        "f1_macro":             f1_score(y_test, predictions, average="macro"),
        "f1_per_class":         f1_score(y_test, predictions, average=None),
        "confusion_matrix":     confusion_matrix(y_test, predictions),
        "classification_report": classification_report(
                                    y_test, predictions,
                                    target_names=target_names
                                ),
    }
    return results


# =============================================================================
# FULL PIPELINE — called by app.py and terminal mode
# =============================================================================

def run_pipeline(k=5):
    """
    Run the complete IPO pipeline and return all artefacts needed
    for the Streamlit frontend and terminal display.
    """
    # INPUT
    X, y, target_names, feature_names = load_data()

    # PROCESS — split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # PROCESS — scale (Gatekeeper Rule)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # PROCESS — train KNN
    model = train_model(X_train_scaled, y_train, k=k)

    # OUTPUT — evaluate
    results = evaluate_model(model, X_test_scaled, y_test, target_names)

    return {
        "model":          model,
        "scaler":         scaler,
        "X_train":        X_train,
        "X_test":         X_test,
        "y_train":        y_train,
        "y_test":         y_test,
        "target_names":   target_names,
        "feature_names":  feature_names,
        "results":        results,
        "k":              k,
        "train_size":     len(X_train),
        "test_size":      len(X_test),
    }


def predict_single(model, scaler, features: list, target_names):
    """
    Predict the species of a single flower from raw measurements.
    features: [sepal_length, sepal_width, petal_length, petal_width]
    """
    arr    = np.array(features).reshape(1, -1)
    scaled = scaler.transform(arr)
    pred   = model.predict(scaled)[0]
    proba  = model.predict_proba(scaled)[0]
    return target_names[pred], proba


# =============================================================================
# TERMINAL MODE — run directly: python classifier.py
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  DecodeLabs | Project 2 | Data Classification")
    print("  Built by Egwuatu Chibuike Dominion | Batch 2026")
    print("=" * 60)

    pipeline = run_pipeline(k=5)
    r        = pipeline["results"]

    print(f"\n📊 DATASET")
    print(f"   Samples      : 150  (Balanced)")
    print(f"   Classes      : 3   (Setosa, Versicolor, Virginica)")
    print(f"   Features     : 4   (Sepal L/W, Petal L/W)")
    print(f"   Train split  : {pipeline['train_size']} samples (80%)")
    print(f"   Test split   : {pipeline['test_size']} samples (20%)")

    print(f"\n⚙️  MODEL")
    print(f"   Algorithm    : K-Nearest Neighbors")
    print(f"   K value      : {pipeline['k']} (majority vote)")
    print(f"   Scaling      : StandardScaler (mean=0, variance=1)")

    print(f"\n✅ RESULTS")
    print(f"   Accuracy     : {r['accuracy']*100:.2f}%")
    print(f"   F1 Score     : {r['f1_macro']*100:.2f}%  (macro)")

    print(f"\n📋 CLASSIFICATION REPORT")
    print(r["classification_report"])

    print(f"\n🔢 CONFUSION MATRIX")
    print(r["confusion_matrix"])

    print("\n" + "=" * 60)
    print("  Pipeline complete. Run 'streamlit run app.py' for UI.")
    print("=" * 60)
