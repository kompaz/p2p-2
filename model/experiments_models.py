from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from utils import (
    load_data,
    clean_data,
    split_features_target,
    get_column_types,
    build_preprocessor,
    evaluate_model
)


def main():
    print("Loading data...")
    df = load_data()

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

    xgboost_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            ))
        ]
    )

    lightgbm_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                verbosity=-1
))
        ]
    )

    catboost_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=5,
                verbose=0,
                random_state=42
            ))
        ]
    )

    results = []

    gb_metrics, _ = evaluate_model(
        "Gradient Boosting",
        gradient_boosting_pipeline,
        X_train, X_test, y_train, y_test
    )
    results.append(gb_metrics)

    xgb_metrics, _ = evaluate_model(
        "XGBoost",
        xgboost_pipeline,
        X_train, X_test, y_train, y_test
    )
    results.append(xgb_metrics)

    lgbm_metrics, _ = evaluate_model(
        "LightGBM",
        lightgbm_pipeline,
        X_train, X_test, y_train, y_test
    )
    results.append(lgbm_metrics)

    cat_metrics, _ = evaluate_model(
        "CatBoost",
        catboost_pipeline,
        X_train, X_test, y_train, y_test
    )
    results.append(cat_metrics)

    print("\nModel Comparison Summary")
    print("-" * 50)
    for result in results:
        print(
            f"{result['model']}: "
            f"accuracy={result['accuracy']:.4f}, "
            f"precision={result['precision']:.4f}, "
            f"recall={result['recall']:.4f}, "
            f"f1_score={result['f1_score']:.4f}, "
            f"roc_auc={result['roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()