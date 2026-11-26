import json

import numpy as np
import pandas as pd

from src.tiffin_mlops.monitoring import evaluate, population_stability_index
from src.tiffin_mlops.train import FEATURES, NUMERIC_FEATURES


class ConstantPredictor:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


def test_psi_is_zero_for_identical_distributions():
    values = pd.Series(range(1, 101))
    assert population_stability_index(values, values) == 0.0


def test_psi_increases_when_current_values_shift():
    reference = pd.Series(range(100))
    current = pd.Series(range(100, 200))
    assert population_stability_index(reference, current) > 1.0


def test_evaluate_reports_zone_mae_and_applies_promotion_rule(tmp_path):
    reference = pd.DataFrame({feature: range(1, 11) for feature in NUMERIC_FEATURES})
    rows = []
    for index in reversed(range(10)):
        row = {
            "record_id": f"R{index:03d}",
            "date": f"2026-07-{index + 1:02d}",
            "delivery_min": float(index),
            "transport": "bicycle",
            "route_zone": "Central" if index % 2 == 0 else "Suburb",
        }
        row.update({feature: float(index + 1) for feature in NUMERIC_FEATURES})
        rows.append(row)
    later = pd.DataFrame(rows)
    reference_path = tmp_path / "reference.csv"
    later_path = tmp_path / "later.csv"
    report_path = tmp_path / "report.json"
    reference.to_csv(reference_path, index=False)
    later.to_csv(later_path, index=False)

    report = evaluate(
        reference_path,
        later_path,
        v1_model=ConstantPredictor(0),
        v2_model=ConstantPredictor(8),
        output_path=report_path,
    )

    assert report["data"]["later_rows"] == 10
    assert report["data"]["holdout_rows"] == 3
    assert set(report["v1_mae_by_route_zone_minutes"]) == {"Central", "Suburb"}
    assert report["promotion_evaluation"]["promotion_approved"] is True
    assert report["promotion_evaluation"]["decision"] == "promote_v2"
    assert json.loads(report_path.read_text()) == report
