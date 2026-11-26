import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

NUMERIC_FEATURES = [
    "distance_km",
    "tiffins",
    "handovers",
    "wait_at_station_min",
    "rider_age",
    "temp_c",
]

CATEGORICAL_FEATURES = [
    "transport",
    "route_zone",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "delivery_min"

EXPERIMENT_NAME = "MP005_Tiffin_Relay"
REGISTERED_MODEL_NAME = "MP005_Tiffin_Relay_Model"


# --------------------------------------------------
# DATASET SHA-256
# --------------------------------------------------

def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# --------------------------------------------------
# PREPROCESSING + MODEL PIPELINE
# --------------------------------------------------

def build_pipeline(estimator):

    # Missing numeric values are filled using
    # the median learned from the training split.
    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ])

    # Missing categorical values are filled
    # using the most frequent category.
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    return pipeline


# --------------------------------------------------
# MODEL TRAINING + MLFLOW TRACKING
# --------------------------------------------------

def train_models():

    cleaned_path = Path(
        "data/processed/train_cleaned.csv"
    )

    raw_path = Path(
        "data/raw/train.csv"
    )

    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {cleaned_path}"
        )

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {raw_path}"
        )

    # Load cleaned dataset
    df = pd.read_csv(cleaned_path)

    print("Dataset shape:", df.shape)
    print("\nMissing values before preprocessing:")
    print(df[FEATURES].isnull().sum())

    # Separate features and target
    X = df[FEATURES]
    y = df[TARGET]

    # Split before fitting preprocessing
    # to prevent data leakage.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("\nTraining samples:", len(X_train))
    print("Validation samples:", len(X_val))

    # Models to compare
    models = {
        "LinearRegression": LinearRegression(),

        "DecisionTree": DecisionTreeRegressor(
            max_depth=8,
            random_state=42,
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        ),
    }

    # MLflow configuration
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Hash original raw training data
    data_sha256 = calculate_sha256(raw_path)

    print("\nOriginal dataset SHA-256:")
    print(data_sha256)

    results = []

    # Train each model
    for model_name, estimator in models.items():

        pipeline = build_pipeline(estimator)

        print(f"\nTraining {model_name}...")

        with mlflow.start_run(
            run_name=model_name
        ) as run:

            # Step 1: Preprocessing
            # Step 2: Model training
            pipeline.fit(X_train, y_train)

            # Validation predictions
            predictions = pipeline.predict(X_val)

            # Evaluate in minutes
            mae = mean_absolute_error(
                y_val,
                predictions,
            )

            # Log experiment parameters
            mlflow.log_param(
                "model_name",
                model_name,
            )

            mlflow.log_param(
                "data_sha256",
                data_sha256,
            )

            mlflow.log_param(
                "test_size",
                0.2,
            )

            mlflow.log_param(
                "random_state",
                42,
            )

            # Log model hyperparameters
            mlflow.log_params(
                estimator.get_params()
            )

            # Log performance
            mlflow.log_metric(
                "mae",
                mae,
            )

            # FIX:
            # Use cloudpickle instead of skops
            # to avoid numpy.dtype trust errors.
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle",
            )

            results.append({
                "model": model_name,
                "mae": mae,
                "run_id": run.info.run_id,
            })

            print(
                f"{model_name}: "
                f"MAE = {mae:.4f} minutes"
            )

    # Rank models by validation MAE
    results.sort(
        key=lambda item: item["mae"]
    )

    print("\n" + "=" * 45)
    print("MODEL COMPARISON RESULTS")
    print("=" * 45)

    for result in results:
        print(
            f"{result['model']}: "
            f"MAE = {result['mae']:.4f} minutes"
        )

    print("\n" + "=" * 45)
    print("BEST MODEL")
    print("=" * 45)

    best = results[0]

    print("Model:", best["model"])
    print("MAE:", round(best["mae"], 4))
    print("Run ID:", best["run_id"])

    return results


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    train_models()