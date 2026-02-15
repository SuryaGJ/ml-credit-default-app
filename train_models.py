import pandas as pd
import numpy as np
import joblib

import os
os.makedirs("model", exist_ok=True)

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

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


print("Loading dataset...")
df = pd.read_csv("UCI_Credit_Card.csv")

# -----------------------------
# DATA PREPROCESSING
# -----------------------------

# Drop ID column (not useful)
df.drop("ID", axis=1, inplace=True)

# Target
y = df["default.payment.next.month"]

# Features
X = df.drop("default.payment.next.month", axis=1)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (important for LR & KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate(name, model, preds):
    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "AUC": roc_auc_score(y_test, preds),
        "MCC": matthews_corrcoef(y_test, preds)
    }
    print(results)
    return results


results_list = []

# -----------------------------
# TRAIN MODELS
# -----------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

import os
os.makedirs("model", exist_ok=True)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    results = evaluate(name, model, preds)
    results_list.append(results)

    joblib.dump(model, f"model/{name.replace(' ','_')}.pkl")


# -----------------------------
# SAVE RESULTS TABLE
# -----------------------------
results_df = pd.DataFrame(results_list)
results_df.to_csv("model_results.csv", index=False)

print("\n✅ Training Complete!")
print(results_df)
