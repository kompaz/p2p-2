from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from utils import (
    load_data,
    clean_data,
    split_features_target,
    get_column_types,
    build_preprocessor,
    evaluate_model
)


def add_minimal_engineered_features(df):
    df = df.copy()

    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
    df["IsElectronicCheck"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["HasFiberOptic"] = (df["InternetService"] == "Fiber optic").astype(int)

    return df


def main():
    print("Loading data...")
    df = load_data()

    print("Cleaning data...")
    df = clean_data(df)

    print("Adding minimal engineered features...")
    df = add_minimal_engineered_features(df)

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

    rf_pipeline = Pipeline(
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

    evaluate_model(
        "Random Forest + Minimal Feature Engineering",
        rf_pipeline,
        X_train, X_test, y_train, y_test
    )


if __name__ == "__main__":
    main()