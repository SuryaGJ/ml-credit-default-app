import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------------
# PAGE
# --------------------------------------------------
st.title("Credit Card Default Prediction App")

# --------------------------------------------------
# SAFE MODEL LOADER
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_safe(path):
    model = joblib.load(path)

    # sklearn compatibility patch
    if hasattr(model, "__dict__") and "monotonic_cst" not in model.__dict__:
        model.monotonic_cst = None

    if hasattr(model, "estimators_"):
        for tree in model.estimators_:
            if "monotonic_cst" not in tree.__dict__:
                tree.monotonic_cst = None

    return model


@st.cache_resource(show_spinner=False)
def load_scaler():
    return joblib.load("model/scaler.pkl")


@st.cache_resource(show_spinner=False)
def load_features():
    return joblib.load("model/feature_columns.pkl")


scaler = load_scaler()
feature_cols = load_features()

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

st.success(f"✅ Active Model: {model_name}")

# --------------------------------------------------
# DATA INPUT
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("Using uploaded dataset")
else:
    data = pd.read_csv("sample_test.csv")
    st.info("Using default sample dataset")

TARGET = "default.payment.next.month"

if TARGET not in data.columns:
    st.error("Target column missing")
    st.stop()

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------
y_true = data[TARGET]
X = data.drop(columns=[TARGET])

if "ID" in X.columns:
    X = X.drop(columns=["ID"])

X = X.apply(pd.to_numeric, errors="coerce")

# FORCE SAME FEATURE ORDER
X = X.reindex(columns=feature_cols)

# SAME TRAINING IMPUTATION
X = X.fillna(X.mean())
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

X_scaled = scaler.transform(X)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
preds = model.predict(X_scaled)

st.subheader("Predictions")
st.write(preds[:20])

# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------
st.subheader("Evaluation Metrics")

report = classification_report(y_true, preds, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

st.download_button(
    "Download Classification Report",
    report_df.to_csv().encode(),
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

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

buf = io.BytesIO()
fig.savefig(buf, format="png")

st.download_button(
    "Download Confusion Matrix",
    buf.getvalue(),
    "confusion_matrix.png",
    "image/png",
)
