# G036-MP005 — Tiffin Relay Delivery Time Prediction

**Group:** G036 | **Problem:** MP005 — Tiffin Relay Delivery Time (Lucknow) | **Faculty:** Amit Soni | **Batch:** 72

MLOps project for predicting tiffin delivery time in Lucknow from delivery, rider, weather, transport, and route-zone features.

## Team and contributions

| Member | Enrolment no. | Role | Branch |
| --- | --- | --- | --- |
| Lakshay Godara | S24CSEU2725 | Data Lead | `data-lead` |
| Shree Sahu | S24CSEU2730 | Model Lead | `model-lead` |
| Aryan Sharma | S24CSEU2711 | Service Lead | `service-lead` |
| Ashutosh Tiwari | S24CSEU1561 | Ops Lead | `ops-lead` |

## Project structure

- `data/raw/`: DVC pointer files and holdout input; source CSVs are not committed.
- `data/processed/train_cleaned.csv`: cleaned training data.
- `src/tiffin_mlops/`: data cleaning and validation, training, retraining, prediction export, and monitoring.
- `models/model.joblib`: exported champion pipeline (V2).
- `predictions/holdout_predictions.csv`: predictions for 250 holdout records.
- `mlflow.db`, `mlruns/`: experiment tracking, registry, and model artifacts.
- `params.yaml`: promotion rule and registered-model name, read by `src/tiffin_mlops/config.py`.
- `app/`: FastAPI inference application.
- `tests/`: model, API, and monitoring tests.
- `k8s/`: Kubernetes Deployment and Service manifests.
- `.github/workflows/ci.yml`: GitHub Actions test workflow.
- `reports/`: data/model findings and Ops monitoring report.

## Data preparation and tracking

The Data Lead checks seven quality issues and removes duplicate rows, corrects the Central-zone distance unit mix-up (metres to km), blanks the two zero waiting-time readings, and blanks the stuck rider-age sensor values (2026-03-27 to 2026-04-11). No sentinel value is present in `train.csv`: zero is a valid waiting time (the column ranges from 0 to 25), but the cleaning step treats those two rows as missing and the pipeline imputes the median, which has a negligible effect. The leakage field `delivered_min_from_receipt` is removed from model features. Numeric missing values are imputed in the model pipeline; categorical values are imputed and one-hot encoded with unknown categories ignored.

Raw `train.csv` and `later.csv` are DVC-tracked and ignored by Git. The checked-in `.dvc/config` has no storage remote configured. To reproduce from DVC, obtain the team's remote configuration, install DVC, then run:

```sh
dvc pull data/raw/train.csv data/raw/later.csv
```

The CSVs used for the Ops report were available in the local working copy and their MD5 hashes match the checked-in DVC pointers. The raw CSVs are not included in the Git repository; a fresh clone needs the DVC remote or copies of those datasets to rerun data checks and monitoring.

Run the Data Lead validation and cleaning scripts from the repository root when their raw inputs are available:

```sh
python -m src.tiffin_mlops.data_checks
python -m src.tiffin_mlops.cleaning
```

## Model results

The selected V1 is a Linear Regression pipeline. The recorded validation MAE is 2.3536 minutes. Retraining with the first 70% of date-sorted later-period data produced V2; the final 30% (90 records) was used for the promotion comparison.

- V1 later-period MAE: 13.3911 minutes.
- V2 later-period MAE: 8.0950 minutes.
- Improvement: 39.55%.
- Promotion rule: `v2_mae <= 0.92 * v1_mae`.
- Decision: promote V2; registered champion version 2.

The exported champion model is `models/model.joblib`. Holdout output has `record_id` and `prediction` columns for 250 rows. Training, tuning, and retraining scripts are in `src/tiffin_mlops/`; retraining and registration can create MLflow runs or model versions, so do not rerun them for routine inference checks.

## FastAPI service

Install dependencies and start the service from the repository root:

```sh
python -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Implemented endpoints:

- `GET /health` returns `{"status": "ok", "model_loaded": true}`.
- `POST /predict` accepts the eight model features and returns `prediction_minutes`.
- `GET /docs` provides FastAPI's generated Swagger UI.

There is no `/model-info` endpoint in the current application. Pydantic rejects missing or extra fields and enforces the current `tiffins <= 100` and `temp_c >= 5` constraints. Successful predictions are emitted as structured JSON to stdout.

Example request:

```sh
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"distance_km":6.9,"tiffins":71,"handovers":3,"wait_at_station_min":22.1,"rider_age":46,"temp_c":22.4,"transport":"bicycle","route_zone":"Suburb"}'
```

## Ops monitoring

Run after the raw `later.csv` is available and from the repository root:

```sh
python -m src.tiffin_mlops.monitoring
```

The script calculates PSI for all six numeric features against cleaned training data, v1 MAE by `route_zone`, and v1/v2 MAE on the final 30% after sorting `later.csv` by date and record ID. It writes `reports/ops_monitoring.json`; actual results are also recorded in `reports/findings.yaml`.

Recorded PSI values: `distance_km` 0.0477, `tiffins` 5.3845, `handovers` 0.0097, `wait_at_station_min` 0.0150, `rider_age` 0.0793, and `temp_c` 0.0178. V1 zone MAE is Central 4.6107, Old City 22.9428, and Suburb 5.1050 minutes. The repository/faculty checklist provided no PSI alert threshold, so PSI is reported for review without an automated drift alert. The promotion decision uses the documented 8% MAE improvement rule and recommends V2.

## CI, tests, and deployment

GitHub Actions runs pytest on pushes and pull requests. Run the same suite locally:

```sh
python -m pytest -v
```

Integration verification on `final-integration` (2026-11-26): **18 tests passed, 0 failed, 0 skipped** on Python 3.14.2. Pytest emitted one Starlette/httpx deprecation warning. The seven data checks completed with results matching the recorded data findings. Registered MLflow V2 and `models/model.joblib` produced matching predictions on all 250 holdout records; the API response also matched the exported model on a holdout example. GitHub Actions has not yet run for `final-integration`.

Build and deploy the API to a Kubernetes cluster with the manifests in `k8s/`:

```sh
docker build -t mp005-api:1.0 .
# For a local Kind cluster, make the image available with:
kind load docker-image mp005-api:1.0
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl rollout status deployment/mp005-api
kubectl get deployment mp005-api
kubectl get service mp005-svc
```

The deployment specifies three replicas on container port 8000; the Service `mp005-svc` listens on port 80 and forwards to 8000. Readiness and liveness probes call `/health`. The Service is internal (`ClusterIP`); use `kubectl port-forward service/mp005-svc 8000:80` for a local check. These manifests are prepared, but no live cluster deployment is claimed here.

## Individual contributions

- **Lakshay Godara (Data Lead, `data-lead`):** DVC tracking, the seven data-quality checks, cleaning code and the data part of `findings.yaml`.
- **Shree Sahu (Model Lead, `model-lead`):** baseline and tuned models, MLflow tracking and registry, retraining (V2 champion), holdout predictions and the model tests.
- **Aryan Sharma (Service Lead, `service-lead`):** FastAPI service, request validation rules, API tests and the Dockerfile.
- **Ashutosh Tiwari (Ops Lead, `ops-lead`):** GitHub Actions, Kubernetes manifests, prediction logging, PSI and route-zone MAE monitoring, and the final integration into `main`.

## Limitations and submission notes

- A DVC storage remote is not configured in the repository; fresh clones need remote configuration or the raw CSVs.
- The report has no PSI alert threshold because none was specified; it reports feature drift values for review.
- The API currently implements `/health` and `/predict`, not `/model-info`.
- A Kubernetes cluster and image distribution are needed to perform a live deployment.
- The local test result above does not establish a green remote GitHub Actions run.
