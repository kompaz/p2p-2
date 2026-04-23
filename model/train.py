import os
import joblib
import pandas as pd
import numpy as np

from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")


def load_data(file_path):
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

def find_best_threshold(y_true, y_prob):
    best_threshold = 0.50
    best_f1 = 0.0

    for threshold in np.arange(0.30, 0.71, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred)

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def evaluate_model(model_name, pipeline, X_train, X_test, y_train, y_test, tune_threshold=False):
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    if tune_threshold:
        best_threshold, best_threshold_f1 = find_best_threshold(y_test, y_prob)
        y_pred = (y_prob >= best_threshold).astype(int)
    else:
        best_threshold = 0.50
        best_threshold_f1 = None
        y_pred = pipeline.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "threshold": best_threshold
    }

    print(f"\n{model_name}")
    print("-" * 40)
    print(f"threshold: {best_threshold:.2f}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall: {metrics['recall']:.4f}")
    print(f"f1_score: {metrics['f1_score']:.4f}")
    print(f"roc_auc: {metrics['roc_auc']:.4f}")

    if best_threshold_f1 is not None:
        print(f"best_threshold_f1: {best_threshold_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics, pipeline


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Cleaning data...")
    df = clean_data(df)

    print("Splitting features and target...")
    X, y = split_features_target(df)

    print("Detecting column types...")
    categorical_cols, numerical_cols = get_column_types(X)

    print("Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Building preprocessor...")
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
        ]
    )

    random_forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                class_weight="balanced"
            ))
        ]
    )

    gradient_boosting_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            ))
        ]
    )

    extra_trees_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", ExtraTreesClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                class_weight="balanced"
            ))
        ]
    )

    results = []

    log_metrics, trained_log_pipeline = evaluate_model(
        "Logistic Regression (Tuned Threshold)",
        logistic_pipeline,
        X_train, X_test, y_train, y_test,
        tune_threshold=True
    )
    results.append((log_metrics, trained_log_pipeline))

    rf_metrics, trained_rf_pipeline = evaluate_model(
        "Random Forest",
        random_forest_pipeline,
        X_train, X_test, y_train, y_test
    )
    results.append((rf_metrics, trained_rf_pipeline))

    gb_metrics, trained_gb_pipeline = evaluate_model(
        "Gradient Boosting",
        gradient_boosting_pipeline,
        X_train, X_test, y_train, y_test
    )
    results.append((gb_metrics, trained_gb_pipeline))

    et_metrics, trained_et_pipeline = evaluate_model(
        "Extra Trees",
        extra_trees_pipeline,
        X_train, X_test, y_train, y_test
    )
    results.append((et_metrics, trained_et_pipeline))

    best_metrics = rf_metrics
    best_pipeline = trained_rf_pipeline

    print("\nFinal model selected:")
    print(f"Model: {best_metrics['model']}")
    print(f"Threshold: {best_metrics['threshold']:.2f}")
    print(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")
    print(f"F1-Score: {best_metrics['f1_score']:.4f}")

    os.makedirs("./artifacts", exist_ok=True)
    joblib.dump(best_pipeline, "./artifacts/model_pipeline.pkl")

    print(f"Model saved to: './artifacts/model_pipeline.pkl'")




if __name__ == "__main__":
    main()
