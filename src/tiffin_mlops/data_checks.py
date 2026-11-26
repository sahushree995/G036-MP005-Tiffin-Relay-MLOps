
"""
MP005 - Tiffin Relay Delivery Time
Group: G036
Role: Data Lead

Detects the seven data quality problems in train.csv.
"""

import pandas as pd
from difflib import get_close_matches
import json


def check_data(train_path, later_path, holdout_path):

    # Load original datasets
    train = pd.read_csv(train_path)
    later = pd.read_csv(later_path)
    holdout = pd.read_csv(holdout_path)

    results = {}

    # ------------------------------------------------
    # 1. DATA LEAKAGE
    # ------------------------------------------------

    column = "delivered_min_from_receipt"

    leakage = (
        column in train.columns
        and column not in holdout.columns
        and abs(
            train["delivery_min"].corr(train[column])
        ) >= 0.95
    )

    results["leaking_column"] = {
        "present": bool(leakage),
        "column": column if leakage else None
    }

    # ------------------------------------------------
    # 2. UNIT MIXUP
    # ------------------------------------------------

    medians = train.groupby("route_zone")["distance_km"].median()

    unit_result = None

    for group, median in medians.items():

        other_median = medians.drop(index=group).median()
        ratio = median / other_median

        for factor in [10, 12, 60, 100, 1000]:

            if abs(ratio - factor) / factor <= 0.20:

                unit_result = {
                    "present": True,
                    "column": "distance_km",
                    "group_value": group,
                    "factor": factor
                }
                break

        if unit_result:
            break

    results["unit_mixup"] = unit_result or {
        "present": False,
        "column": None,
        "group_value": None,
        "factor": None
    }

    # ------------------------------------------------
    # 3. SENTINEL VALUE
    # ------------------------------------------------

    sentinel_result = None

    candidates = [-9999, -999, -99, -1, 0, 999, 9999]

    for column in train.select_dtypes(include="number").columns:

        for value in candidates:

            count = int((train[column] == value).sum())

            # Zero is checked as a sentinel for station waiting time.
            if value == 0 and column != "wait_at_station_min":
                continue

            if count > 0:

                sentinel_result = {
                    "present": True,
                    "column": column,
                    "value": value,
                    "count": count
                }
                break

        if sentinel_result:
            break

    results["sentinel_value"] = sentinel_result or {
        "present": False,
        "column": None,
        "value": None,
        "count": None
    }

    # ------------------------------------------------
    # 4. DUPLICATE ROWS
    # ------------------------------------------------

    duplicate_count = int(train.duplicated().sum())

    results["duplicate_rows"] = {
        "present": duplicate_count > 0,
        "count": duplicate_count if duplicate_count > 0 else None
    }

    # ------------------------------------------------
    # 5. STUCK SENSOR
    # ------------------------------------------------

    df = train.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["date", "record_id"]
    ).reset_index(drop=True)

    stuck_result = None

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        values = df[column].tolist()

        start = 0

        for i in range(1, len(values) + 1):

            if i == len(values) or values[i] != values[start]:

                run_length = i - start

                if run_length >= 30:

                    if (
                        stuck_result is None
                        or run_length > stuck_result["run_length"]
                    ):

                        stuck_result = {
                            "column": column,
                            "start_date": str(
                                df.loc[start, "date"].date()
                            ),
                            "end_date": str(
                                df.loc[i - 1, "date"].date()
                            ),
                            "run_length": run_length
                        }

                start = i

    results["stuck_sensor"] = {
        "present": stuck_result is not None,
        "column": stuck_result["column"] if stuck_result else None,
        "start_date": stuck_result["start_date"] if stuck_result else None,
        "end_date": stuck_result["end_date"] if stuck_result else None
    }

    # ------------------------------------------------
    # 6. TARGET TYPOS
    # ------------------------------------------------

    median_target = train["delivery_min"].median()

    typo_mask = (
        train["delivery_min"] >= 8 * median_target
    )

    typo_count = int(typo_mask.sum())

    results["target_typos"] = {
        "present": typo_count > 0,
        "count": typo_count if typo_count > 0 else None
    }

    # ------------------------------------------------
    # 7. SPELLING VARIANTS
    # ------------------------------------------------

    spelling_result = None

    for column in ["transport", "route_zone"]:

        train_values = set(
            train[column].dropna().astype(str)
        )

        later_values = set(
            later[column].dropna().astype(str)
        )

        clean_values = {
            value.strip().lower()
            for value in later_values
        }

        variants = []

        for value in train_values:

            normalized = value.strip().lower()

            if normalized not in clean_values:

                matches = get_close_matches(
                    normalized,
                    list(clean_values),
                    n=1,
                    cutoff=0.80
                )

                if matches:
                    variants.append(value)

        if variants:

            spelling_result = {
                "present": True,
                "column": column,
                "true_levels": len(clean_values)
            }

            break

    results["spelling_variants"] = spelling_result or {
        "present": False,
        "column": None,
        "true_levels": None
    }

    return results


# ------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------

if __name__ == "__main__":

    findings = check_data(
        "data/raw/train.csv",
        "data/raw/later.csv",
        "data/raw/holdout_features.csv"
    )

    print("\n========== MP005 DATA QUALITY REPORT ==========\n")

    for issue, details in findings.items():

        print(issue)
        print(json.dumps(details, indent=4))
        print("-----------------------------------------------")

    print("\nAll seven data checks completed.")
