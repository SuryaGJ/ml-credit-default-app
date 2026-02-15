import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -----------------------------
# CREATE MODEL DIRECTORY
# -----------------------------
os.makedirs("model", exist_ok=True)

print("Loading dataset...")

df = pd.read_csv("UCI_Credit_Card.csv")

# -----------------------------
# BASIC CLEANING
# -----------------------------
TARGET = "default.payment.next.month"

y = df[TARGET]
X = df.drop(columns=[TARGET])

if "ID" in X.columns:
    X = X.drop(columns=["ID"])

# convert all to numeric (safety)
X = X.apply(pd.to_numeric, errors="coerce")

# fill missing values
X = X.fillna(X.mean())

# -----------------------------
# SAVE FEATURE ORDER ⭐ IMPORTANT
# -----------------------------
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "model/feature_columns.pkl")

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")

# -----------------------------
# MODELS
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=8),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=20,
        max_depth=6,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=30,
        max_depth=4,
        eval_metric="logloss",
        use_label_encoder=False
    ),
}

results = []

# -----------------------------
# TRAINING LOOP
# -----------------------------
for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "AUC": roc_auc_score(y_test, probs),
        "MCC": matthews_corrcoef(y_test, preds),
    }

    print(metrics)
    results.append(metrics)

    # save model
    filename = f"model/{name.replace(' ','_')}.pkl"
    joblib.dump(model, filename)

# -----------------------------
# SAVE RESULTS
# -----------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("model_results.csv", index=False)

print("\n✅ Training Complete!")
print(results_df)
