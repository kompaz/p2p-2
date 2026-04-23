from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

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

    baseline_pipeline = Pipeline(
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

    oversampling_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("sampler", RandomOverSampler(random_state=42)),
            ("classifier", RandomForestClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42
            ))
        ]
    )

    undersampling_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("sampler", RandomUnderSampler(random_state=42)),
            ("classifier", RandomForestClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42
            ))
        ]
    )

    smote_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("sampler", SMOTE(random_state=42)),
            ("classifier", RandomForestClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42
            ))
        ]
    )

    results = []

    baseline_metrics, _ = evaluate_model(
        "Random Forest Baseline",
        baseline_pipeline,
        X_train, X_test, y_train, y_test
    )
    results.append(baseline_metrics)

    over_metrics, _ = evaluate_model(
        "Random Forest + RandomOverSampler",
        oversampling_pipeline,
        X_train, X_test, y_train, y_test
    )
    results.append(over_metrics)

    under_metrics, _ = evaluate_model(
        "Random Forest + RandomUnderSampler",
        undersampling_pipeline,
        X_train, X_test, y_train, y_test
    )
    results.append(under_metrics)

    smote_metrics, _ = evaluate_model(
        "Random Forest + SMOTE",
        smote_pipeline,
        X_train, X_test, y_train, y_test
    )
    results.append(smote_metrics)

    print("\nSampling Comparison Summary")
    print("-" * 60)
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