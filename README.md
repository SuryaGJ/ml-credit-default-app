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

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8098 | 0.6033 | 0.6920 | 0.2361 | 0.3521 | 0.3251 |
| Decision Tree | 0.7255 | 0.6105 | 0.3807 | 0.4059 | 0.3929 | 0.2160 |
| KNN | 0.7952 | 0.6373 | 0.5493 | 0.3564 | 0.4323 | 0.3252 |
| Naive Bayes | 0.7070 | 0.6866 | 0.3967 | 0.6504 | 0.4928 | 0.3218 |
| Random Forest | 0.8143 | 0.6213 | 0.6874 | 0.2780 | 0.3959 | 0.3531 |
| XGBoost | 0.8148 | 0.6520 | 0.6347 | 0.3625 | 0.4615 | 0.3801 |

---

## Observations on Model Performance

### Logistic Regression
Logistic Regression achieved high accuracy but low recall, indicating difficulty detecting default cases due to linear decision boundaries.

### Decision Tree
Decision Tree produced balanced precision and recall but lower overall accuracy, showing sensitivity to overfitting.

### K-Nearest Neighbors
KNN showed stable performance with moderate accuracy and balanced metrics but requires feature scaling and higher computation.

### Naive Bayes
Naive Bayes achieved the highest recall, identifying more default customers, though overall accuracy decreased due to independence assumptions.

### Random Forest (Ensemble)
Random Forest achieved strong accuracy and improved MCC score by combining multiple decision trees, reducing variance.

### XGBoost (Ensemble)
XGBoost achieved the best balanced performance among models, demonstrating strong capability for structured tabular datasets.

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


- 🚀 **Live Streamlit App:** https://ml-credit-default-app-brdwzpq5mbz6ssgrbeungf.streamlit.app
