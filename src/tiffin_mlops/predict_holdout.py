from pathlib import Path

import joblib
import pandas as pd

from src.tiffin_mlops.train import FEATURES


def generate_holdout_predictions():
    model_path = Path("models/model.joblib")
    input_path = Path("data/raw/holdout_features.csv")
    output_path = Path("predictions/holdout_predictions.csv")

    model = joblib.load(model_path)
    holdout = pd.read_csv(input_path)

    required_columns = ["record_id"] + FEATURES
    missing_columns = [
        col for col in required_columns
        if col not in holdout.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    predictions = model.predict(holdout[FEATURES])

    output = pd.DataFrame({
        "record_id": holdout["record_id"],
        "prediction": predictions,
    })

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    output.to_csv(output_path, index=False)

    print("Holdout rows:", len(holdout))
    print("Predictions generated:", len(output))
    print("Output:", output_path)
    print("\nFirst 5 predictions:")
    print(output.head())


if __name__ == "__main__":
    generate_holdout_predictions()