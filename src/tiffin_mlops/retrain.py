import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from src.tiffin_mlops.train import (
    FEATURES,
    TARGET,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
    calculate_sha256,
)

V1_VERSION = "1"
IMPROVEMENT_THRESHOLD = 0.92


def retrain_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = MlflowClient()

    # 1. Load original cleaned training data.
    original = pd.read_csv(
        "data/processed/train_cleaned.csv"
    )

    # 2. Load later-period data chronologically.
    later = pd.read_csv("data/raw/later.csv")

    later["date"] = pd.to_datetime(
        later["date"], errors="raise"
    )

    later = later.sort_values(
        ["date", "record_id"], kind="stable"
    ).reset_index(drop=True)

    split_index = int(len(later) * 0.70)

    later_train = later.iloc[:split_index].copy()
    later_test = later.iloc[split_index:].copy()

    # Ensure the chronological split is valid.
    assert (
        later_train["date"].max()
        <= later_test["date"].min()
    )

    combined = pd.concat(
        [original, later_train],
        ignore_index=True,
    )

    X_train = combined[FEATURES]
    y_train = combined[TARGET]

    X_test = later_test[FEATURES]
    y_test = later_test[TARGET]

    print("Original rows:", len(original))
    print("Later retraining rows:", len(later_train))
    print("Later test rows:", len(later_test))
    print("Combined training rows:", len(combined))

    # 3. Load registered Version 1.
    v1 = mlflow.sklearn.load_model(
        f"models:/{REGISTERED_MODEL_NAME}/{V1_VERSION}"
    )

    v1_predictions = v1.predict(X_test)
    v1_mae = mean_absolute_error(
        y_test, v1_predictions
    )

    # 4. Clone the original pipeline and retrain.
    v2 = clone(v1)
    v2.fit(X_train, y_train)

    v2_predictions = v2.predict(X_test)
    v2_mae = mean_absolute_error(
        y_test, v2_predictions
    )

    required_mae = IMPROVEMENT_THRESHOLD * v1_mae
    promote_v2 = v2_mae <= required_mae

    # 5. Log the retraining experiment.
    with mlflow.start_run(
        run_name="Retrained_LinearRegression_v2"
    ) as run:

        mlflow.log_params({
            "model_name": "LinearRegression",
            "original_rows": len(original),
            "later_train_rows": len(later_train),
            "later_test_rows": len(later_test),
            "retraining_rows": len(combined),
            "later_split_ratio": 0.70,
            "promotion_threshold": 0.92,
            "data_sha256": calculate_sha256(
                "data/raw/train.csv"
            ),
            "later_data_sha256": calculate_sha256(
                "data/raw/later.csv"
            ),
        })

        mlflow.log_metrics({
            "v1_later_mae": v1_mae,
            "v2_later_mae": v2_mae,
            "required_v2_mae": required_mae,
            "improvement_pct": (
                (v1_mae - v2_mae) / v1_mae
            ) * 100,
        })

        model_info = mlflow.sklearn.log_model(
            sk_model=v2,
            name="model",
            serialization_format="cloudpickle",
        )

        print("\nRetraining run ID:", run.info.run_id)

    # 6. Register Version 2, regardless of promotion.
    registered_v2 = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    # 7. Apply the 8% promotion rule.
    champion_version = (
        registered_v2.version if promote_v2
        else V1_VERSION
    )

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=champion_version,
    )

    print("\n" + "=" * 55)
    print("RETRAINING AND PROMOTION RESULTS")
    print("=" * 55)

    print(f"V1 later MAE: {v1_mae:.4f} minutes")
    print(f"V2 later MAE: {v2_mae:.4f} minutes")
    print(
        f"Required V2 MAE: {required_mae:.4f} minutes"
    )
    print(
        "Improvement: "
        f"{((v1_mae - v2_mae) / v1_mae) * 100:.2f}%"
    )
    print("Registered V2 version:", registered_v2.version)
    print("Promotion approved:", promote_v2)
    print("Champion version:", champion_version)


if __name__ == "__main__":
    retrain_model()