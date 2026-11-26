# G036-MP005-Tiffin-Relay-MLOps
MLOps pipeline for predicting tiffin relay delivery time in Lucknow

## Ops Lead runbook

### Continuous integration

GitHub Actions runs on every push and pull request. It installs the pinned project requirements, then runs the pytest suite. A green `CI / test` check is the code validation gate.

### Build and run the API with Kubernetes

Build the container from the repository root and make that image available to your cluster (for a local Kind cluster, load it with `kind load docker-image mp005-api:latest`). Then apply the manifests:

```sh
docker build -t mp005-api:latest .
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl rollout status deployment/mp005-api
kubectl get deployment mp005-api
kubectl get service mp005-svc
```

The deployment maintains three replicas. The service listens on port 8000 and sends traffic to container port 8000. Kubernetes readiness and liveness probes call `/health`. For a local smoke check, forward the service port with `kubectl port-forward service/mp005-svc 8000:8000` and call `http://127.0.0.1:8000/health`.

### Prediction logs

Each successful `/predict` request writes one JSON object to stdout. The object includes the event name, model inputs, and predicted delivery minutes. Kubernetes collects container stdout; use your cluster's configured log collector to search on `event=prediction`.

### Drift, MAE, and promotion report

Once DVC has restored `data/raw/later.csv`, run:

```sh
python -m src.tiffin_mlops.monitoring
```

The command compares the six numeric features in the cleaned training data and later-period data using PSI, reports v1 MAE for every `route_zone`, then compares v1 and v2 MAE on the chronologically sorted final 30%. It writes `reports/ops_monitoring.json`. The report applies the 8% rule (`v2 MAE <= 0.92 * v1 MAE`), and records either `promote_v2` or `retain_v1`. By default it loads v1 from MLflow model version 1 and v2 from `models/model.joblib`. If your model artifacts use another location, pass `--v1-model` and `--v2-model` with joblib paths.

The repository currently has a DVC pointer for `later.csv`, but no configured DVC storage remote and no copy of that dataset. Set the team's DVC remote and run `dvc pull data/raw/later.csv.dvc` before producing the drift and route-zone metrics. Existing `reports/findings.yaml` records the model-lead promotion result (v1 MAE 13.3911, v2 MAE 8.0950, 39.55% improvement, promotion approved); the new monitoring command recomputes and saves the detailed Ops Lead report from source rows.
