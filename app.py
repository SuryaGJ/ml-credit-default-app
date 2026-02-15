import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Credit Card Default Prediction App")

# -----------------------------
# SAFE MODEL LOADER
# -----------------------------
def load_model_safe(path):
    model = joblib.load(path)

    # ---- sklearn forward compatibility patch ----
    # New sklearn expects this attribute for tree models
    if hasattr(model, "__dict__"):
        if "monotonic_cst" not in model.__dict__:
            model.monotonic_cst = None

    # RandomForest contains internal trees
    if hasattr(model, "estimators_"):
        for tree in model.estimators_:
            if "monotonic_cst" not in tree.__dict__:
                tree.monotonic_cst = None

    return model


# -----------------------------
# LOAD SCALER
# -----------------------------
scaler = joblib.load("model/scaler.pkl")

# -----------------------------
# MODEL SELECTION
# -----------------------------
model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

model_path = f"model/{model_name.replace(' ','_')}.pkl"
model = load_model_safe(model_path)

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Test CSV (Optional)", type=["csv"]
)

# Default dataset if none uploaded
if uploaded_file is None:
    st.info("No file uploaded. Using default sample test dataset.")
    data = pd.read_csv("UCI_Credit_Card.csv").sample(500, random_state=42)
else:
    data = pd.read_csv(uploaded_file)

# -----------------------------
# PREPROCESSING
# -----------------------------
if "default.payment.next.month" not in data.columns:
    st.error("Target column missing!")
    st.stop()

y_true = data["default.payment.next.month"]
X = data.drop("default.payment.next.month", axis=1)

if "ID" in X.columns:
    X = X.drop("ID", axis=1)

# Handle ANY sklearn version strictness
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))

X_scaled = scaler.transform(X)

# -----------------------------
# PREDICTION
# -----------------------------
preds = model.predict(X_scaled)

st.subheader("Predictions")
st.write(preds)

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
st.subheader("Classification Report")

report = classification_report(y_true, preds, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# Download button
csv_report = report_df.to_csv().encode("utf-8")
st.download_button(
    "Download Classification Report",
    csv_report,
    "classification_report.csv",
    "text/csv"
)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_true, preds)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# Download confusion matrix image
fig.savefig("confusion_matrix.png")
with open("confusion_matrix.png", "rb") as f:
    st.download_button(
        "Download Confusion Matrix",
        f,
        "confusion_matrix.png",
        "image/png"
    )
