import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------
st.title("Credit Card Default Prediction App")
st.write(
    "Interactive ML application to predict credit card default risk "
    "using multiple classification models."
)

# --------------------------------------------------
# SAFE MODEL LOADER (sklearn version compatible)
# --------------------------------------------------
def load_model_safe(path):
    model = joblib.load(path)

    # Patch for newer sklearn versions
    if hasattr(model, "__dict__") and "monotonic_cst" not in model.__dict__:
        model.monotonic_cst = None

    # Random Forest internal trees
    if hasattr(model, "estimators_"):
        for tree in model.estimators_:
            if hasattr(tree, "__dict__") and "monotonic_cst" not in tree.__dict__:
                tree.monotonic_cst = None

    return model


# --------------------------------------------------
# LOAD ARTIFACTS
# --------------------------------------------------
scaler = joblib.load("model/scaler.pkl")
feature_cols = joblib.load("model/feature_columns.pkl")

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------
model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost",
    ],
)

model_path = f"model/{model_name.replace(' ','_')}.pkl"
model = load_model_safe(model_path)

# --------------------------------------------------
# DATA INPUT
# --------------------------------------------------
st.subheader("Upload Test CSV (Optional)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Using uploaded dataset.")
else:
    data = pd.read_csv("sample_test.csv")
    st.info("No file uploaded. Using default sample dataset.")

TARGET = "default.payment.next.month"

if TARGET not in data.columns:
    st.error(f"Target column '{TARGET}' not found.")
    st.stop()

# --------------------------------------------------
# PREPROCESSING (MATCHES TRAINING EXACTLY)
# --------------------------------------------------
y_true = data[TARGET]
X = data.drop(columns=[TARGET])

# Drop ID if present
if "ID" in X.columns:
    X = X.drop(columns=["ID"])

# Convert to numeric safely
X = X.apply(pd.to_numeric, errors="coerce")

# Align feature order with training
X = X.reindex(columns=feature_cols)

# SAME imputation used during training
X = X.fillna(X.mean())

# Final safety cleanup
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# Scale
X_scaled = scaler.transform(X)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
preds = model.predict(X_scaled)

st.subheader("Predictions")
st.write(preds)

# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------
st.subheader("Evaluation Metrics (Classification Report)")

report = classification_report(y_true, preds, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# Download classification report
csv = report_df.to_csv().encode("utf-8")
st.download_button(
    "Download Classification Report (CSV)",
    csv,
    "classification_report.csv",
    "text/csv",
)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_true, preds)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

st.pyplot(fig)

# Download confusion matrix
buf = io.BytesIO()
fig.savefig(buf, format="png")

st.download_button(
    "Download Confusion Matrix (PNG)",
    buf.getvalue(),
    "confusion_matrix.png",
    "image/png",
)
