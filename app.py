import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("Credit Card Default Prediction App")
st.write(
    "Interactive Machine Learning application to predict credit card default risk "
    "using multiple classification models."
)

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
model = joblib.load(model_path)

# -----------------------------
# DATA INPUT
# -----------------------------
st.subheader("Upload Test CSV (Optional)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Use uploaded file OR default sample dataset
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Using uploaded dataset.")
else:
    data = pd.read_csv("sample_test.csv")
    st.info("No file uploaded. Using default sample test dataset.")

# -----------------------------
# PREDICTION SECTION
# -----------------------------
if "default.payment.next.month" in data.columns:

    y_true = data["default.payment.next.month"]

    X = data.drop("default.payment.next.month", axis=1)

    # Drop ID column safely
    if "ID" in X.columns:
        X = X.drop("ID", axis=1)

    # Scale features
    X_scaled = scaler.transform(X)

    # Predictions
    preds = model.predict(X_scaled)

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    st.subheader("Predictions")
    st.write(preds)

    # -----------------------------
    # CLASSIFICATION REPORT
    # -----------------------------
    st.subheader("Evaluation Metrics (Classification Report)")
    report = classification_report(y_true, preds, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, preds)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    st.pyplot(fig)

else:
    st.error("Target column 'default.payment.next.month' not found in dataset.")
