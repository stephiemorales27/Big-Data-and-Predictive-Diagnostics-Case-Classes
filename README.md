# Big Data & Diagnostics and Predictability

A collection of data analysis and machine learning case studies developed during two university courses focused on **Big Data** and **Diagnostics and Predictability**. All projects are implemented in Python and cover the full analytical workflow — from data preprocessing to statistical inference, predictive modeling, and time series forecasting.

## Overview

This repository brings together case studies and applied exercises completed as part of coursework in two courses. It explores diagnostic analysis, predictive modeling, and Big Data techniques across a variety of real-world scenarios, including retail sales, insurance costs, medical diagnostics, customer retention, and music industry data.

## Note on Code Attribution

The base notebooks, starter code, and dataset scaffolding for most of these exercises were provided by the course instructors as part of the class curriculum. My contribution consists of completing the analysis, interpreting the results, extending the code where required by each assignment, and documenting the statistical and business reasoning behind each decision. This repository is shared as a record of applied learning, not as original tooling or production software.

**Exceptions:**
- The *Feature Engineering & Preprocessing Pipeline — Music Industry Sales Data* project was independently developed by me, applying the concepts learned in class to a self-selected dataset and building the full preprocessing pipeline from scratch.
- The *Customer Retention Diagnostic — TechLearn Case Study* was completed as a **group project** (5 members), representing 15% of the course grade. It is included here as it best demonstrates my applied understanding of the full classification workflow, including hyperparameter tuning and model comparison.

## Technologies

- Python
- Pandas / NumPy
- Scikit-learn
- Databricks / Apache Spark (PySpark)
- Matplotlib / Seaborn
- SciPy (statistical testing)
- Statsmodels (time series analysis)

## Featured Project

### Customer Retention Diagnostic — TechLearn Case Study (Group Project)
The most comprehensive project in this repository. Diagnoses the drivers of customer churn using an adapted Telco Customer Churn dataset (Kaggle), applying:
- Logistic Regression and Random Forest models, compared side by side
- Hyperparameter tuning via `GridSearchCV` and `StratifiedKFold` cross-validation
- Full evaluation suite: accuracy, precision, recall, F1-score, ROC-AUC, ROC curve, precision-recall curve, and confusion matrix
- Business-oriented recommendations translating model findings into retention strategy

## Contents

- **Diagnostic analysis case studies** — hypothesis testing, correlation analysis, and exploratory data analysis on business datasets.
- **Predictive modeling examples** — linear/ridge regression, logistic regression classification, ensemble methods (Random Forest), and model evaluation (R², MAE, RMSE, accuracy, precision, recall, F1-score, ROC-AUC).
- **Time series forecasting** — stationarity testing (ADF), ACF/PACF analysis, moving averages, and ARIMA models for demand and price forecasting.
- **Data preprocessing and feature engineering** — cleaning pipelines, scaling, encoding, and feature selection using scikit-learn `Pipeline` and `ColumnTransformer`.
- **Big Data techniques** — distributed data processing and machine learning workflows using Databricks and PySpark (data cleaning, clustering with MLlib, aggregation at scale).

