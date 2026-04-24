"""
train_v2.py — Metrik bazlı otomatik model seçimi ile eğitim scripti.

Orijinal `train.py`'dan farkları:
1. LightGBM pipeline'ı eklendi (class_weight balanced)
2. Final model seçimi metrik bazlı 
3. Tüm modellerin metriklerini JSON olarak kaydeder (model_metadata.json)
4. Seçilen modelin adı ve kazanma gerekçesi log'lanır

Kullanım:
    python -m model.train_v2
"""
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model_pipeline.pkl")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "model_metadata.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model seçim kriteri — hangi metrik ile en iyi model seçilecek
# Alternatifler: "roc_auc" (ranking quality), "f1_score" (precision-recall balance)
# Churn için F1 genelde daha anlamlı çünkü retention kampanya bütçesi sınırlı,
# hem precision hem recall önemli.
PRIMARY_METRIC = "f1_score"
SECONDARY_METRIC = "roc_auc"

# Tüm modellerde threshold tuning uygulansın mı?
# True: fair karşılaştırma, her model kendi optimal threshold'u ile değerlendirilir
# False: orijinal davranış (sadece LogReg tuned, diğerleri 0.5 default)
TUNE_THRESHOLD_FOR_ALL = True

# ============================================================================
# Data processing — orijinal train.py ile aynı mantık
# ============================================================================
def load_data(file_path):
    return pd.read_csv(file_path)


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
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


# ============================================================================
# Evaluation
# ============================================================================
def find_best_threshold(y_true, y_prob):
    best_threshold, best_f1 = 0.50, 0.0
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
        best_threshold, _ = find_best_threshold(y_test, y_prob)
        y_pred = (y_prob >= best_threshold).astype(int)
    else:
        best_threshold = 0.50
        y_pred = pipeline.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "threshold": float(best_threshold),
    }

    print(f"\n{model_name}")
    print("-" * 50)
    for k in ["threshold", "accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        print(f"{k:12s}: {metrics[k]:.4f}")

    return metrics, pipeline


# ============================================================================
# Model selection — metrik bazlı otomatik kazanan seçimi
# ============================================================================
def select_best_model(results, primary_metric=PRIMARY_METRIC, secondary_metric=SECONDARY_METRIC):
    """
    En iyi modeli primary metriğe göre seçer, eşitlik durumunda secondary kırar.

    Returns
    -------
    best_metrics : dict
    best_pipeline : sklearn.Pipeline
    winner_rationale : str
        Hangi modelin neden seçildiğini açıklayan insan-okunur metin.
    """
    def sort_key(item):
        metrics, _ = item
        return (metrics[primary_metric], metrics[secondary_metric])

    sorted_results = sorted(results, key=sort_key, reverse=True)
    best_metrics, best_pipeline = sorted_results[0]
    runner_up_metrics, _ = sorted_results[1]

    primary_delta = best_metrics[primary_metric] - runner_up_metrics[primary_metric]
    secondary_delta = best_metrics[secondary_metric] - runner_up_metrics[secondary_metric]

    rationale = (
        f"{best_metrics['model']} seçildi. "
        f"{primary_metric}={best_metrics[primary_metric]:.4f} "
        f"(ikinci sıradaki {runner_up_metrics['model']}'dan "
        f"{primary_delta:+.4f} fark). "
        f"{secondary_metric}={best_metrics[secondary_metric]:.4f} "
        f"(fark {secondary_delta:+.4f})."
    )

    return best_metrics, best_pipeline, rationale


def save_metadata(all_metrics, best_metrics, rationale):
    """Tüm model sonuçlarını ve kazanan bilgisini JSON olarak kaydet."""
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "primary_metric": PRIMARY_METRIC,
        "secondary_metric": SECONDARY_METRIC,
        "threshold_tuning_applied_to_all": TUNE_THRESHOLD_FOR_ALL,
        "selected_model": best_metrics["model"],
        "selection_rationale": rationale,
        "all_models": all_metrics,
        "best_model_metrics": best_metrics,
    }

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nMetadata saved to: {METADATA_PATH}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("Train v2 — Metrik Bazlı Otomatik Model Seçimi")
    print(f"Primary metric: {PRIMARY_METRIC}")
    print(f"Secondary metric (tie-breaker): {SECONDARY_METRIC}")
    print(f"Threshold tuning for all: {TUNE_THRESHOLD_FOR_ALL}")
    print("=" * 60)

    print("\nLoading data...")
    df = load_data(DATA_PATH)

    print("Cleaning data...")
    df = clean_data(df)

    print("Splitting features and target...")
    X, y = split_features_target(df)

    categorical_cols, numerical_cols = get_column_types(X)
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Numerical features: {len(numerical_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    preprocessor = build_preprocessor(categorical_cols, numerical_cols)

    # ------------------------------------------------------------------
    # Model zoo — orijinal 4 model + LightGBM
    # ------------------------------------------------------------------
    pipelines = {
        "Logistic Regression (Tuned Threshold)": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(
                    max_iter=2000, class_weight="balanced", C=0.5
                )),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(
                    n_estimators=500, max_depth=12, min_samples_split=10,
                    min_samples_leaf=4, random_state=RANDOM_STATE,
                    class_weight="balanced",
                )),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.05, max_depth=3,
                    random_state=RANDOM_STATE,
                )),
            ]
        ),
        "Extra Trees": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", ExtraTreesClassifier(
                    n_estimators=500, max_depth=12, min_samples_split=10,
                    min_samples_leaf=4, random_state=RANDOM_STATE,
                    class_weight="balanced",
                )),
            ]
        ),
        "LightGBM (Tuned)": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.03,
                    max_depth=7,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbose=-1,
                )),
            ]
        ),
    }

    # ------------------------------------------------------------------
    # Eğitim + değerlendirme
    # ------------------------------------------------------------------
    results = []
    all_metrics = []

    for name, pipe in pipelines.items():
        metrics, trained = evaluate_model(
            name, pipe, X_train, X_test, y_train, y_test,
            tune_threshold=TUNE_THRESHOLD_FOR_ALL,
        )
        results.append((metrics, trained))
        all_metrics.append(metrics)

    # ------------------------------------------------------------------
    # Metrik bazlı kazanan seçimi
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MODEL SEÇİM SÜRECİ")
    print("=" * 60)

    # Karşılaştırma tablosu
    comparison_df = pd.DataFrame(all_metrics).set_index("model")
    print("\n", comparison_df.round(4).to_string())

    best_metrics, best_pipeline, rationale = select_best_model(results)

    print("\n" + "=" * 60)
    print("FINAL MODEL")
    print("=" * 60)
    print(rationale)

    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    save_metadata(all_metrics, best_metrics, rationale)

    print("\nDone.")


if __name__ == "__main__":
    main()