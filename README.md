# Credit Card Default Prediction using Machine Learning

## Problem Statement
The goal of this project is to develop and compare multiple machine learning classification models to predict whether a credit card customer will default on their payment in the following month. The project demonstrates an end-to-end machine learning workflow including data preprocessing, model training, evaluation using multiple performance metrics, and deployment through an interactive Streamlit web application.

---

## Dataset Description

This project uses the **UCI Credit Card Default Dataset**, which contains demographic, financial, and payment history information of credit card clients.

### Dataset Information
- Total Instances: 30,000
- Number of Features: 24 input features
- Problem Type: Binary Classification
- Target Variable: `default.payment.next.month`

Target Meaning:
- **1** → Customer will default
- **0** → Customer will not default

### Feature Categories
The dataset includes:
- Credit limit information
- Demographic attributes (age, gender, education, marriage)
- Historical payment records
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

Each model was evaluated using the following metrics:

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
| Logistic Regression | 0.8098 | 0.6033 | 0.6919 | 0.2361 | 0.3521 | 0.3251 |
| Decision Tree | 0.7287 | 0.6180 | 0.3892 | 0.4212 | 0.4045 | 0.2295 |
| KNN | 0.7952 | 0.6373 | 0.5493 | 0.3564 | 0.4323 | 0.3252 |
| Naive Bayes | 0.7070 | 0.6866 | 0.3967 | 0.6504 | 0.4928 | 0.3218 |
| Random Forest | 0.8150 | 0.6521 | 0.6355 | 0.3625 | 0.4617 | 0.3806 |
| XGBoost | 0.8148 | 0.6520 | 0.6347 | 0.3625 | 0.4615 | 0.3801 |

---

## Observations on Model Performance

### Logistic Regression
The Logistic Regression model achieved strong overall accuracy but low recall, indicating difficulty identifying minority default cases. This suggests linear decision boundaries are insufficient for capturing complex relationships.

### Decision Tree
Decision Tree showed balanced recall and precision but lower overall accuracy. Single-tree models are prone to overfitting and reduced generalization.

### K-Nearest Neighbors (KNN)
KNN achieved moderate performance across metrics. The model benefits from feature scaling but becomes computationally expensive for large datasets.

### Naive Bayes
Naive Bayes achieved the highest recall, detecting more default customers. However, accuracy decreased due to its assumption of feature independence.

### Random Forest (Ensemble)
Random Forest achieved the best overall performance with highest accuracy and MCC score. Ensemble learning reduced variance and improved robustness.

### XGBoost (Ensemble)
XGBoost produced performance comparable to Random Forest and demonstrated strong predictive capability for structured tabular data.

---

## Streamlit Web Application

An interactive Streamlit application was developed with the following features:

- CSV dataset upload option
- Model selection dropdown
- Prediction generation
- Classification report display
- Confusion matrix visualization

The application allows users to test different trained models dynamically.

---
