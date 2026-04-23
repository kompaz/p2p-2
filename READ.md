# Customer Churn Prediction and API Service

## Project Overview
This project aims to predict whether a telecom customer is likely to leave the company using the Telco Customer Churn dataset. In addition to building a machine learning model, the project provides a working API service for churn prediction.

## Project Objective
The main goal of this project is to analyze customer-related features, identify patterns connected to customer churn, train multiple machine learning models, and expose the final selected model through a FastAPI-based prediction endpoint.

## Dataset
The dataset used in this project is the **Telco Customer Churn** dataset.

It includes customer-related information such as:
- demographic details
- subscription information
- billing and payment data
- additional service usage
- churn status

## Exploratory Data Analysis (EDA)
During the analysis process, the following key findings were observed:

- Customer churn is imbalanced in the dataset.
- Customers with **month-to-month contracts** are more likely to leave.
- Customers with **shorter tenure** show a higher churn tendency.
- Customers with **higher monthly charges** are more likely to churn.
- Customers using **Fiber optic** internet service tend to have higher churn.
- Customers without **TechSupport** show higher churn.
- Customers using **Electronic check** as a payment method are more likely to leave.

## Data Preprocessing
The following preprocessing steps were applied:

- `customerID` was removed because it does not provide predictive value.
- `TotalCharges` was converted to numeric format.
- Rows with invalid `TotalCharges` values were removed.
- `Churn` values were encoded as:
  - `Yes -> 1`
  - `No -> 0`
- Categorical variables were transformed using **One-Hot Encoding**.
- Numerical variables were processed with imputation and scaling inside a pipeline.

## Models Tested
The following models were trained and compared:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Extra Trees

## Final Model
The final selected model is:

**Random Forest Classifier**

The trained model was saved as:

`artifacts/model_pipeline.pkl`

## Project Structure
```text
p2p-2/
│
├── app/
│   ├── main.py
│   └── schemas.py
│
├── artifacts/
│   └── model_pipeline.pkl
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── model/
│   └── train.py
│
├── notebooks/
│   └── eda.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore