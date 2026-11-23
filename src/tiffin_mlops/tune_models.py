import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
)
from sklearn.metrics import mean_absolute_error

from src.tiffin_mlops.train import (
    FEATURES,
    TARGET,
    EXPERIMENT_NAME,
    build_pipeline,
    calculate_sha256,
)


def tune_models():
    df = pd.read_csv("data/processed/train_cleaned.csv")

    X = df[FEATURES]
    y = df[TARGET]

    # Same split as baseline for comparison.
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    candidates = {
        "Ridge": (
            Ridge(),
            {
                "model__alpha": [0.1, 1.0, 10.0, 100.0],
            },
        ),
        "Lasso": (
            Lasso(max_iter=20000),
            {
                "model__alpha": [0.001, 0.01, 0.1, 1.0],
            },
        ),
        "ElasticNet": (
            ElasticNet(max_iter=20000),
            {
                "model__alpha": [0.001, 0.01, 0.1],
                "model__l1_ratio": [0.2, 0.5, 0.8],
            },
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
    }

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    data_hash = calculate_sha256("data/raw/train.csv")

    results = []

    for name, (estimator, param_grid) in candidates.items():
        print(f"\nTuning {name}...")

        search = GridSearchCV(
            estimator=build_pipeline(estimator),
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=1,
            refit=True,
        )

        # GridSearchCV fits preprocessing separately
        # inside each training fold.
        search.fit(X_train, y_train)

        cv_mae = -search.best_score_
        val_predictions = search.best_estimator_.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_predictions)

        with mlflow.start_run(run_name=f"Tuned_{name}") as run:
            mlflow.log_param("model_name", name)
            mlflow.log_param("data_sha256", data_hash)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("random_state", 42)
            mlflow.log_params(search.best_params_)

            mlflow.log_metric("cv_mae", cv_mae)
            mlflow.log_metric("mae", val_mae)

            mlflow.sklearn.log_model(
                sk_model=search.best_estimator_,
                name="model",
                serialization_format="cloudpickle",
            )

            results.append({
                "model": name,
                "cv_mae": cv_mae,
                "val_mae": val_mae,
                "run_id": run.info.run_id,
            })

        print("Best parameters:", search.best_params_)
        print(f"CV MAE: {cv_mae:.4f} minutes")
        print(f"Validation MAE: {val_mae:.4f} minutes")

    print("\n" + "=" * 50)
    print("TUNED MODEL COMPARISON")
    print("=" * 50)

    for result in sorted(results, key=lambda x: x["cv_mae"]):
        print(
            f"{result['model']}: "
            f"CV MAE = {result['cv_mae']:.4f}, "
            f"Validation MAE = {result['val_mae']:.4f}"
        )

    best = min(results, key=lambda x: x["cv_mae"])

    print("\nBEST TUNED MODEL (BY CROSS-VALIDATION)")
    print("Model:", best["model"])
    print("CV MAE:", round(best["cv_mae"], 4))
    print("Validation MAE:", round(best["val_mae"], 4))
    print("Run ID:", best["run_id"])

    print("\nOriginal Linear Regression validation MAE: 2.3536")

    return results


if __name__ == "__main__":
    tune_models()