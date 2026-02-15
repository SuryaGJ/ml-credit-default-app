# Credit Card Default Prediction using Machine Learning

## Problem Statement
The objective of this project is to develop and compare multiple machine learning classification models to predict whether a credit card customer will default on payment in the following month. The project demonstrates a complete end-to-end machine learning workflow including data preprocessing, model training, evaluation using multiple metrics, and deployment through an interactive Streamlit web application.

---

## Dataset Description

This project uses the **UCI Credit Card Default Dataset**, which contains financial and demographic information of credit card clients.

### Dataset Information
- Total Records: 30,000
- Number of Features: 24
- Problem Type: Binary Classification
- Target Variable: `default.payment.next.month`

Target Meaning:
- **1** → Customer will default
- **0** → Customer will not default

### Feature Categories
The dataset includes:
- Credit limit information
- Demographic attributes (age, gender, education, marriage)
- Historical repayment behavior
- Bill statement amounts
- Previous payment amounts

---

## Machine Learning Models Implemented

The following six classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

---

## Evaluation Metrics

Each model was evaluated using:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## Model Performance Comparison

| Model               | Accuracy   | Precision  | Recall     | F1 Score   | AUC        | MCC        |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | 0.8077     | 0.6868     | 0.2396     | 0.3553     | 0.7076     | 0.3244     |
| Decision Tree       | 0.8153     | 0.6454     | 0.3662     | 0.4673     | 0.7406     | 0.3872     |
| KNN                 | 0.8003     | 0.5820     | 0.3451     | 0.4333     | 0.7098     | 0.3378     |
| Naive Bayes         | 0.7525     | 0.4515     | 0.5539     | 0.4975     | 0.7249     | 0.3386     |
| Random Forest       | 0.8138     | 0.6744     | 0.3060     | 0.4209     | 0.7702     | 0.3647     |
| **XGBoost** ⭐       | **0.8173** | **0.6571** | **0.3640** | **0.4685** | **0.7765** | **0.3925** |


---

## Observations on Model Performance

| Model               | Observation                                                             |
| ------------------- | ----------------------------------------------------------------------- |
| Logistic Regression | Good baseline model but struggles to detect defaulters (low recall).    |
| Decision Tree       | Captures nonlinear relationships effectively with balanced performance. |
| KNN                 | Performs reasonably well but sensitive to feature scaling.              |
| Naive Bayes         | High recall but lower precision due to probabilistic assumptions.       |
| Random Forest       | More stable than a single tree and improves overall robustness.         |
| XGBoost             | Achieved best overall performance with highest AUC and MCC scores.      |

---

## Streamlit Web Application Features

The deployed Streamlit application includes:

- CSV dataset upload functionality
- Default sample test dataset
- Model selection dropdown
- Prediction generation
- Classification report display (evaluation metrics)
- Confusion matrix visualization

---

## Project Structure

ml-credit-default-app/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── sample_test.csv
├── model/
│ ├── Logistic_Regression.pkl
│ ├── Decision_Tree.pkl
│ ├── KNN.pkl
│ ├── Naive_Bayes.pkl
│ ├── Random_Forest.pkl
│ ├── XGBoost.pkl
│ └── scaler.pkl

## Running Locally

Install dependencies:
pip install -r requirements.txt

Run application:
python -m streamlit run app.py

## Deployment

The application is deployed using **Streamlit Community Cloud** and provides an interactive interface for testing multiple machine learning models.

---


## Execution Environment

Model training and testing were performed locally and executed on BITS Virtual Lab as required in the assignment instructions.

## Project Links


- 🚀 **Live Streamlit App:** https://ml-credit-default-app-nwcceprhtwmyposcd8qcme.streamlit.app

