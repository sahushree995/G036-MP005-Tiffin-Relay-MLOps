import mlflow
from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "MP005_Tiffin_Relay_Model"

BEST_RUN_ID = "ff8501c06a6d4992b0d63d187ac2d497"


def register_best_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    client = MlflowClient()

    # Verify the selected baseline run.
    run = client.get_run(BEST_RUN_ID)

    if run.data.tags.get("mlflow.runName") != "LinearRegression":
        raise ValueError(
            "The selected run is not LinearRegression."
        )

    print("Best run:", BEST_RUN_ID)
    print("Validation MAE:", run.data.metrics["mae"])

    # MLflow 3.x stores logged model metadata separately.
    logged_models = client.search_logged_models(
        experiment_ids=[run.info.experiment_id],
    )

    matching_models = []

    for model in logged_models:
        source_run_id = getattr(
            model,
            "source_run_id",
            None,
        )

        if source_run_id == BEST_RUN_ID:
            matching_models.append(model)

    if not matching_models:
        raise RuntimeError(
            "No logged model found for the selected run."
        )

    model_id = matching_models[0].model_id

    print("Logged model ID:", model_id)

    # Register the already logged model.
    registered = mlflow.register_model(
        model_uri=f"models:/{model_id}",
        name=REGISTERED_MODEL_NAME,
    )

    print("\nMODEL REGISTERED SUCCESSFULLY")
    print("Name:", REGISTERED_MODEL_NAME)
    print("Version:", registered.version)

    # Champion alias will be assigned after
    # v1 vs v2 retraining comparison.


if __name__ == "__main__":
    register_best_model()