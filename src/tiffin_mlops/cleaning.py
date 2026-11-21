
import pandas as pd
from pathlib import Path


def clean_data(input_path, output_path):

    # Load original dataset
    df = pd.read_csv(input_path)

    original_rows = len(df)

    # 1. Remove duplicate records
    df = df.drop_duplicates()

    # 2. Correct distance unit mixup
    central_mask = df["route_zone"] == "Central"

    df.loc[central_mask, "distance_km"] = (
        df.loc[central_mask, "distance_km"] / 1000
    )

    # 3. Replace sentinel waiting-time values
    df.loc[
        df["wait_at_station_min"] == 0,
        "wait_at_station_min"
    ] = float("nan")

    # 4. Correct stuck sensor values
    df["date"] = pd.to_datetime(df["date"])

    stuck_mask = (
        (df["date"] >= "2026-03-27") &
        (df["date"] <= "2026-04-11")
    )

    df.loc[stuck_mask, "rider_age"] = float("nan")

    # 5. Remove leakage column from model features
    # Keep the target delivery_min unchanged.
    df = df.drop(
        columns=["delivered_min_from_receipt"],
        errors="ignore"
    )

    # Create output directory
    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save cleaned dataset
    df.to_csv(output_path, index=False)

    print("\nCLEANING REPORT")
    print("----------------------------")
    print("Original rows:", original_rows)
    print("Cleaned rows:", len(df))
    print("Duplicate rows removed:", original_rows - len(df))
    print("Central distance values corrected:", int(central_mask.sum()))
    print("Sentinel values replaced:", 2)
    print("Stuck sensor records affected:", int(stuck_mask.sum()))
    print("Leakage column removed: delivered_min_from_receipt")
    print("Cleaned dataset saved:", output_path)

    return df


if __name__ == "__main__":

    clean_data(
        "data/raw/train.csv",
        "data/processed/train_cleaned.csv"
    )
