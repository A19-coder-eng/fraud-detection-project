
# Bank Fraud Detection Using Machine Learning

This project detects fraudulent banking transactions using a large real-world dataset (1 million entries with 32 features). The target variable is fraud_bool, a binary indicator of fraud. The goal is to help financial institutions identify suspicious behavior and mitigate risk using supervised learning models.


## Table of Contents
* Getting Started

* Dataset Overview

* Installation

* Project 

* Modeling

* Evaluation Metrics

* Results

* Contact
## Getting Started

**Prerequisites**

Install the following Python libraries:

```bash
pip install numpy pandas scikit-learn xgboost
```






##  Dataset Overview
* **Shape:** (1,000,000 rows × 32 columns)

* **Target Column:** fraud_bool

* **Class Distribution:**

  * **Non-Fraud:** 988,971

  * **Fraud:** 11,029
## Features
* **Numerical**: income, velocity_6h, customer_age, etc.

* **Categorical**: payment_type, employment_status, housing_status, device_os, source

* No missing values initially, but some columns use -1 as a placeholder, handled during preprocessing.
## Project Pipeline
**Custom Preprocessing**

A `PreprocessingCleaner` class:

* Drops irrelevant or redundant columns

* Replaces -1 with NaN in select columns

**Feature Engineering**

Built using `Pipeline` and `ColumnTransformer`:

* **Numerical**: imputed with median, scaled using StandardScaler

* **Categorical**: imputed with most frequent, encoded using OneHotEncoder

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

 base_pipeline = Pipeline([
    ('cleaner', PreprocessingCleaner()),
    ('preprocessor', preprocessor)
])
```
##  Modeling
**Models Trained:**

* Logistic Regression

* Random Forest

* XGBoost Classifier

##  Modeling
**Models Trained:**

* Logistic Regression

* Random Forest

* XGBoost Classifier

## Evaluation Metrics
**Evaluated using:**

* **Accuracy**

* **F1 Score** (especially important for imbalanced data)

* **ROC AUC Score**

* **Overfitting Check** (comparing train vs test)
## Results
##  Model Performance Comparison

| Model               | Train Accuracy | Test Accuracy | Train F1 Score | Test F1 Score | ROC AUC Score | Overfitting |
| ------------------- | -------------- | ------------- | -------------- | ------------- | ------------- | ----------- |
| Logistic Regression | 0.9890         | 0.9890        | 0.0181         | 0.0249        | 0.8724        |  No        |
| Random Forest       | 0.9999         | 0.9889        | 0.9993         | 0.0054        | 0.8335        |  No        |
| XGBoost             | 0.9898         | 0.9888        | 0.1711         | 0.0775        | 0.8865        |  No        |

## Insights
* All models generalize well (no overfitting).

* Due to class imbalance, F1 Scores are very low, especially for fraud class.

* XGBoost performed the best overall in terms of F1 Score and AUC.
## Contact
**Author:** **Sarita Jangid**

-  Email: [saritajangid197@gmail.com](mailto:saritajangid197@gmail.com)  
-  LinkedIn: [linkedin.com/in/sarita-jangid-749146308](https://www.linkedin.com/in/sarita-jangid-749146308/)  
-  GitHub: [github.com/A19-coder-eng](https://github.com/A19-coder-eng)
