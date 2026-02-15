import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Credit Card Default Prediction App")

# Load scaler
scaler = joblib.load("model/scaler.pkl")

# Model selection
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

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    if "default.payment.next.month" in data.columns:
        y_true = data["default.payment.next.month"]
        X = data.drop("default.payment.next.month", axis=1)
        X = X.drop("ID", axis=1)

        X_scaled = scaler.transform(X)

        preds = model.predict(X_scaled)

        st.subheader("Predictions")
        st.write(preds)

        st.subheader("Classification Report")
        report = classification_report(y_true, preds, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, preds)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

    else:
        st.error("Target column missing!")
