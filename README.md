# Customer Churn Prediction and API Service

## Project Overview
This project predicts whether a telecom customer is likely to leave the company by using the Telco Customer Churn dataset. In addition to building a machine learning model, the project also provides a working FastAPI service for churn prediction and Docker support for containerized execution.

## Project Objective
The main objective of this project is to:
- analyze the factors related to customer churn,
- preprocess the dataset properly,
- train and compare multiple machine learning models,
- test alternative improvement strategies,
- select a final model,
- and expose that model through an API endpoint.

## Dataset
The dataset used in this project is the **Telco Customer Churn** dataset.

It contains information such as:
- customer demographics
- subscription details
- billing and payment information
- additional services
- churn status

## Exploratory Data Analysis
The following main findings were observed during EDA:

- Customer churn is imbalanced in the dataset.
- Customers with **month-to-month contracts** are more likely to leave.
- Customers with **shorter tenure** have a higher churn tendency.
- Customers with **higher monthly charges** are more likely to churn.
- Customers using **Fiber optic** internet service show higher churn.
- Customers without **TechSupport** show higher churn.
- Customers using **Electronic check** are more likely to leave the company.

## Data Preprocessing
The following preprocessing steps were applied:

- `customerID` was removed because it does not contribute to prediction.
- `TotalCharges` was converted to numeric format.
- Rows with invalid `TotalCharges` values were removed.
- `Churn` values were encoded as:
  - `Yes -> 1`
  - `No -> 0`
- Categorical columns were transformed using **One-Hot Encoding**.
- Numerical columns were processed through imputation and scaling inside a preprocessing pipeline.

## Baseline Models Tested
The following baseline models were trained and evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Extra Trees

## Additional Model Experiments
To improve the baseline performance, the following additional models were tested in separate experiment files:

- XGBoost
- LightGBM
- CatBoost

These experiments were compared against the baseline Random Forest model.

## Sampling Experiments
Because the dataset is imbalanced, the following resampling methods were tested:

- RandomOverSampler
- RandomUnderSampler
- SMOTE

These methods were evaluated to see whether recall and overall churn detection performance could be improved.

## Feature Engineering Experiments
Additional engineered features were also tested, including:

- `IsMonthToMonth`
- `IsElectronicCheck`
- `HasFiberOptic`
- `ServiceCount`
- `HasSupportBundle`
- `TenureGroup`
- `MonthlyChargePerTenure`

A smaller minimal feature engineering set was also tested separately.

## Hyperparameter Tuning
Random Forest hyperparameter tuning was performed using `RandomizedSearchCV` in order to improve model performance.

## Final Model
After comparing:
- baseline models,
- advanced boosting models,
- sampling strategies,
- feature engineering experiments,
- and hyperparameter tuning results,

the final selected model remained:

**Baseline Random Forest Classifier**

This model provided the most balanced overall performance for the churn prediction task.

The trained model is saved at:

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
│   ├── train.py
│   ├── utils.py
│   ├── experiments_models.py
│   ├── experiments_sampling.py
│   ├── experiments_feature_engineering.py
│   ├── experiments_feature_engineering_minimal.py
│   └── experiments_tuning.py
│
├── notebooks/
│   └── eda.ipynb
│
├── app/
│   ├── main.py
│   └── schemas.py
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── README.md
└── .gitignore