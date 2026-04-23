import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


DATA_PATH = "./data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_data(file_path=DATA_PATH):
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def split_features_target(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def get_column_types(X):
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numerical_cols = X.select_dtypes(exclude="object").columns.tolist()
    return categorical_cols, numerical_cols


def build_preprocessor(categorical_cols, numerical_cols):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ]
    )

    return preprocessor


def evaluate_model(model_name, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    print(f"\n{model_name}")
    print("-" * 50)
    print(f"accuracy:  {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"f1_score:  {metrics['f1_score']:.4f}")
    print(f"roc_auc:   {metrics['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics, pipeline