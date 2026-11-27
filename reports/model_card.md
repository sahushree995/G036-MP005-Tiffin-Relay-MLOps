# Model Card - MP005 Tiffin Relay

## Model Overview

- Project: Tiffin Relay Delivery Time Prediction, Lucknow
- Group: G036
- Model Lead: Shree Sahu
- Task: Regression
- Target: delivery_min
- Evaluation Metric: Mean Absolute Error (MAE)
- Champion Version: V2
- Model Registry: MP005_Tiffin_Relay_Model
- Model Artifact: models/model.joblib

## Data Preprocessing

- Median imputation for numerical features.
- Most frequent imputation for categorical features.
- One-hot encoding with unknown category handling.
- Exclusion of the target leakage column.
- Initial training and validation split: 80:20.
- Random seed: 42.

## Model Selection

| Model | Validation MAE (minutes) |
|---|---:|
| Linear Regression | 2.3536 |
| Decision Tree | 5.9329 |
| Random Forest | 3.4925 |
| Ridge | 2.3669 |
| Lasso | 2.3538 |
| ElasticNet | 2.3628 |
| Gradient Boosting | 2.6500 |

Linear Regression was selected for V1 because it achieved
the lowest recorded validation MAE while maintaining simplicity.

## Retraining and Champion Promotion

The later dataset contained 300 records and was split
chronologically into 210 training and 90 testing records.

The combined retraining dataset contained 1210 records.

| Metric | Result |
|---|---:|
| V1 later MAE | 13.3911 minutes |
| V2 later MAE | 8.0950 minutes |
| Required improvement | 8% |
| Achieved improvement | 39.55% |

V2 satisfied the promotion criterion and was assigned
the champion alias in the MLflow Model Registry.

## Holdout Predictions

- Total predictions: 250
- Missing predictions: 0
- Duplicate record IDs: 0
- Output: predictions/holdout_predictions.csv

## Intended Use

The model estimates tiffin delivery duration using
available delivery and operational features.

## Limitations

- Performance may decrease when delivery patterns change.
- Unseen categorical values may reduce prediction reliability.
- Linear models may not capture complex nonlinear relationships.
- Holdout prediction accuracy cannot be established without labels.
- Predictions should be monitored for data and performance drift.

## Maintenance

Monitor incoming data and prediction errors periodically.
Evaluate retrained candidates against the existing champion
before promoting a new model version.
