from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from utils import (
    load_data,
    clean_data,
    split_features_target,
    get_column_types,
    build_preprocessor
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

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                random_state=42,
                class_weight="balanced"
            ))
        ]
    )

    param_dist = {
        "classifier__n_estimators": [200, 300, 500, 700],
        "classifier__max_depth": [8, 10, 12, 15, None],
        "classifier__min_samples_split": [2, 5, 10, 15],
        "classifier__min_samples_leaf": [1, 2, 4, 6],
        "classifier__max_features": ["sqrt", "log2", None]
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1",
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    print("Running RandomizedSearchCV...")
    search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(search.best_params_)

    print("\nBest CV F1 Score:")
    print(search.best_score_)

    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

    print("\nTest Results")
    print("-" * 50)
    print(f"accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"precision: {precision_score(y_test, y_pred):.4f}")
    print(f"recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"f1_score:  {f1_score(y_test, y_pred):.4f}")
    print(f"roc_auc:   {roc_auc_score(y_test, y_prob):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()