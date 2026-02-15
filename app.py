import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import io

from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------------
# TITLE
# -----------------------------------
st.title("Credit Card Default Prediction App")
st.write(
    "Interactive Machine Learning application to predict credit card "
    "default risk using multiple classification models."
)

# -----------------------------------
# LOAD ARTIFACTS
# -----------------------------------
scaler = joblib.load("model/scaler.pkl")
feature_cols = joblib.load("model/feature_columns.pkl")

# -----------------------------------
# MODEL SELECTION
# -----------------------------------
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

model = joblib.load(f"model/{model_name.replace(' ','_')}.pkl")

# -----------------------------------
# DATA INPUT
# -----------------------------------
st.subheader("Upload Test CSV (Optional)")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Using uploaded dataset.")
else:
    df = pd.read_csv("sample_test.csv")
    st.info("No file uploaded. Using default sample test dataset.")

TARGET = "default.payment.next.month"

if TARGET not in df.columns:
    st.error(f"Target column '{TARGET}' not found.")
    st.stop()

# -----------------------------------
# PREPROCESSING (SAFE PIPELINE)
# -----------------------------------
y_true = df[TARGET]
X = df.drop(columns=[TARGET])

# drop ID safely
if "ID" in X.columns:
    X = X.drop(columns=["ID"])

# convert to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# align with training features
X = X.reindex(columns=feature_cols)

# fill missing values
X = X.fillna(X.mean())
X = X.fillna(0)

# scale
X_scaled = scaler.transform(X)

# -----------------------------------
# PREDICTION
# -----------------------------------
preds = model.predict(X_scaled)

st.subheader("Predictions")
st.write(preds)

# -----------------------------------
# CLASSIFICATION REPORT
# -----------------------------------
st.subheader("Evaluation Metrics (Classification Report)")

report = classification_report(y_true, preds, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

# Download metrics
csv = report_df.to_csv().encode("utf-8")
st.download_button(
    label="Download Classification Report (CSV)",
    data=csv,
    file_name="classification_report.csv",
    mime="text/csv",
)

# -----------------------------------
# CONFUSION MATRIX
# -----------------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_true, preds)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

st.pyplot(fig)

# Download confusion matrix image
buf = io.BytesIO()
fig.savefig(buf, format="png")

st.download_button(
    label="Download Confusion Matrix (PNG)",
    data=buf.getvalue(),
    file_name="confusion_matrix.png",
    mime="image/png",
)
