# 📝 Project Reflection & Portfolio Notes
## Iris Classifier — Data Classification Using AI | DecodeLabs Project 2

*Complete this file in your own words before submission. This is your evidence of understanding.*

---

## 1. What I Built

In this project, I built an **Iris Species Classifier** — a supervised machine learning model with two modes:
- A **terminal (CLI) version** in `classifier.py`
- A **web app frontend** using Streamlit in `app.py` with 4 interactive tabs

The model classifies iris flowers into 3 species (Setosa, Versicolor, Virginica) using 4 measurements, trained on 80% of the dataset and validated on 20%.

---

## 2. Key Concepts I Learned

### ✅ Supervised Learning vs Rule-Based

> The difference between Project 1 and Project 2 is fundamental. In the first project, every response the system gave was something I personally wrote. I defined every condition, every branch, every outcome. In this one, I stopped writing logic entirely. I gave the machine 120 labelled examples and it figured out the patterns by itself. That is what supervision means here — you provide the labelled history, and the algorithm derives the logic. The rules still exist, they just live inside the data now instead of inside the code.

### ✅ Why Feature Scaling (The Gatekeeper Rule)

> StandardScaler is applied before training because KNN measures distance between data points to make its decisions. If one feature spans a large range and another spans a small one, the larger-ranged feature will dominate every distance calculation regardless of how informative the smaller one actually is. Scaling brings all features to mean=0 and variance=1 so each gets a fair contribution. The critical rule is to fit the scaler on training data only, then transform both sets using those same learned parameters. Fitting on the full dataset before splitting would be data leakage — the model would have indirectly seen test information during training, making validation scores dishonestly optimistic.

### ✅ Train-Test Split & Why We Shuffle

> Splitting the data matters because without a held-out set, there is no honest way to measure how the model performs on data it has never seen. Training and testing on the same data produces a score that means nothing in the real world. The 80/20 split locks 20% away as a simulation of future, unseen data. Shuffling is necessary because the Iris dataset is ordered by species — without it, the test set would be heavily skewed toward one class. Stratify=y ensures class proportions are preserved in both splits so the evaluation is representative.

### ✅ How KNN Works

> KNN was the hardest concept in this project for me, not because the maths is complicated but because of how different it feels from everything else. Most algorithms produce a formula or equation during training — something you can inspect. KNN produces nothing. It simply memorises every training point and defers all reasoning to prediction time. When a new flower arrives, it calculates the distance from that point to every stored training example, finds the 5 closest, and takes a majority vote. That is the entire algorithm. The moment that clicked — when I accepted that proximity is the model — everything else in this project made sense.

### ✅ Why F1 Score Over Accuracy (The Accuracy Mirage)

> This was genuinely the biggest surprise in this project. I came in thinking accuracy was the standard metric — what percentage did the model get right? The DecodeLabs briefing called 99% accuracy an "Accuracy Mirage" and I didn't fully believe it until I thought through the example. If 99% of your dataset belongs to one class, a model that predicts that class every single time scores 99% accuracy while being completely useless. F1 Score is the harmonic mean of Precision and Recall and it cannot be fooled that way. It penalises false alarms and missed detections equally. Iris is a balanced dataset so accuracy and F1 both look good here, but using F1 is correct practice regardless — the habit is more important than whether this particular dataset needs it.

### ✅ What the Confusion Matrix Shows

> The confusion matrix tells me not just how many errors the model made, but exactly which classes it confused and in which direction. The diagonal is correct predictions. Off-diagonal entries reveal the mistakes — a misclassification in the row means the model missed a real positive (False Negative), in the column it means it raised a false alarm (False Positive). For Iris, Setosa is almost always perfectly classified because its petal dimensions are distinctly small and separate from the other two species. Versicolor and Virginica are closer together in feature space, which is where the matrix shows the occasional error. Seeing that visualised made the confusion matrix feel like an actual diagnostic tool, not just a table of numbers.

---

## 3. Challenges I Faced

**Challenge 1: Understanding KNN as a lazy learner.**
Every other algorithm I had encountered produces something tangible during training — coefficients, decision boundaries, learned weights. KNN produces nothing at training time. It stores the data and does all its work during prediction. This felt wrong to me for a while. I kept looking for what it had "learned." The concept of a lazy learner — one that defers all computation until it needs to make a prediction — took genuine effort to accept. Running the code and seeing correct predictions come out of something that appeared to have learned nothing was what finally made it real.

**Challenge 2: Data leakage — a subtle but serious mistake.**
My instinct was to fit the StandardScaler on the entire dataset before splitting. It only reads the feature values, not the labels, so I assumed it was harmless. The problem is that it exposes the test set's distribution to the training process. The test set must simulate completely unseen real-world data — if the scaler already knows its statistics, the validation score is optimistic in a way that does not reflect real performance. The fix is simple: fit on training data only, transform both. It is a small discipline but one that separates careful ML practice from careless ML practice.

**Challenge 3: Trusting the K=5 default and understanding why.**
The elbow curve in the app made the K tradeoff concrete. K=1 memorises noise. K=20 over-generalises. K=5 sits at the elbow where performance is reliable without either extreme. What surprised me was how sensitive the results were to a single integer. Changing K by 2 in either direction produced measurably different F1 scores. That gave me real appreciation for why hyperparameter tuning is treated as its own discipline in machine learning, even for something as simple as choosing a number.

---

## 4. What I Would Improve

1. **Compare multiple algorithms** — KNN was the specification for this project, but adding a Decision Tree and a Logistic Regression model alongside it would show where each one struggles and help build intuition about algorithm selection.
2. **Add cross-validation** — a single 80/20 split can be fortunate or unlucky depending on which samples land where. K-Fold cross-validation averages performance across multiple splits and gives a more reliable, stable accuracy estimate.
3. **Deploy publicly** — the app works locally but I would push it to Streamlit Cloud so it is accessible from a link without running any code. That turns it from a local script into a real portfolio project anyone can open.

---

## 5. Connection to Real-World AI

Based on the DecodeLabs briefing, this project is the bridge from rule-based systems to full machine learning.

> What supervised learning unlocks is the ability to handle complexity that no human could manually encode. A four-feature Iris dataset is manageable with rules if you try hard enough. But a medical imaging dataset with millions of pixels, or a fraud detection system with hundreds of transaction variables — no one could write those rules by hand. Supervised learning finds the patterns automatically from labelled examples, at a scale and speed that explicit logic never could. That is the real significance of Project 2. It is not just a different technique. It is a different philosophy about how machines should be taught.

---

## 6. Skills Demonstrated

| Skill | Evidence |
|---|---|
| Data Loading | `sklearn.datasets.load_iris()` |
| Feature Scaling | `StandardScaler.fit_transform()` — training data only |
| Train-Test Split | `train_test_split(stratify=y, shuffle=True)` |
| Model Training | `KNeighborsClassifier.fit()` |
| Prediction | `model.predict(X_test_scaled)` |
| Validation | Confusion Matrix + Macro F1 Score |
| Data Visualisation | Matplotlib + Seaborn charts |
| Frontend Development | Streamlit 4-tab interactive app |
| Hyperparameter Tuning | K-value elbow curve with live retrain |
| Anti-pattern awareness | Data leakage prevention + Accuracy Mirage avoided |

---

## 7. Self-Assessment

Rate yourself honestly (1–5):

| Area | Rating | Notes |
|---|---|---|
| Understanding of KNN | 5/5 | Lazy learner concept fully clicked |
| Feature Scaling concept | 5/5 | Data leakage rule is clear and permanent |
| Train-Test Split rationale | 5/5 | Stratify, shuffle, and why — all understood |
| F1 vs Accuracy | 5/5 | Accuracy Mirage was the defining lesson of this project |
| Confusion Matrix reading | 5/5 | Can read and interpret every cell |
| Scikit-Learn workflow | 5/5 | Instantiate, fit, predict — natural now |
| Frontend (Streamlit) | 4/5 | 4-tab app solid, always room to grow on design |

---

## 8. What Comes Next (Project 3 Preview)

According to the DecodeLabs briefing, Project 3 moves from tabular data to visual data — from KNN to Deep Learning and CNNs:

```
Project 2: Feature Vector → KNN → Class Label     (tabular)
Project 3: Image Pixels   → CNN → Class Label     (visual)
```

> Project 2 taught me that a machine can learn patterns from numbers in a table. Project 3 is about teaching it to learn patterns from images — where the input is not four clean measurements but potentially millions of raw pixel values. I want to understand how convolutional filters find edges and shapes the way the human eye does, and how that connects to the classification pipeline built here. The jump from a spreadsheet row to a photograph feels like the real entry point into modern AI, and this project was the foundation that makes that jump possible.

---

*Submitted by: Egwuatu Chibuike Dominion*
*Date: 29/05/2026*
*DecodeLabs Intern | Batch 2026 | AI Engineering Track*
