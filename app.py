import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# TITLE
# -----------------------------
st.title("Credit Card Default Prediction App")
st.write(
    "Interactive ML application to predict credit card default risk "
    "using multiple classification models."
)

# -----------------------------
# LOAD MODEL + SCALER
# -----------------------------
scaler = joblib.load("model/scaler.pkl")

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

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Using uploaded dataset.")
else:
    data = pd.read_csv("sample_test.csv")
    st.info("No file uploaded. Using default sample test dataset.")

# -----------------------------
# VALIDATION + PREPROCESSING
# -----------------------------
TARGET = "default.payment.next.month"

if TARGET not in data.columns:
    st.error(f"Target column '{TARGET}' not found.")
    st.stop()

y_true = data[TARGET]
X = data.drop(TARGET, axis=1)

# Drop ID safely
if "ID" in X.columns:
    X = X.drop("ID", axis=1)

# Convert everything to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Fill missing values safely
X = X.fillna(X.mean())

# Ensure same column order used during training
expected_features = scaler.feature_names_in_
X = X.reindex(columns=expected_features, fill_value=0)

# -----------------------------
# SCALING + PREDICTION
# -----------------------------
X_scaled = scaler.transform(X)
preds = model.predict(X_scaled)

# -----------------------------
# OUTPUT
# -----------------------------
st.subheader("Predictions")
st.write(preds)

# Evaluation metrics
st.subheader("Evaluation Metrics (Classification Report)")
report = classification_report(y_true, preds, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_true, preds)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

st.pyplot(fig)
