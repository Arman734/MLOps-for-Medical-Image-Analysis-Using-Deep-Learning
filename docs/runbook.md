# Operations runbook

## Topology

| Component | Port | Notes |
| --- | --- | --- |
| Clinical console + API | 8000 | `make serve` or the `api` compose service |
| MLflow UI | 5000 | `make mlflow` or the `mlflow` compose service |
| Prometheus | 9090 | compose only; scrapes `api:8000/metrics` |
| Grafana | 3000 | compose only; `admin` / `medmlops` |

The API loads every `*.pt` checkpoint with a matching `*.json` sidecar from
`artifacts/models` at startup. Reload after training with `POST /api/models/refresh`
(the console does this automatically when a background job finishes).

## First-line checks

```bash
curl -s localhost:8000/api/health | jq
curl -s localhost:8000/api/models | jq '.segmentation_ready, .classification_ready, .load_errors'
curl -s localhost:8000/api/monitoring/summary | jq '.feedback, .recent_latency_p95_ms'
python scripts/smoke_test.py
```

`load_errors` should be empty. A non-empty list means a corrupt checkpoint or a missing
architecture dependency — retrain or restore the previous artifact.

## Alert playbook

### `medmlops_model_ready` == 0
1. Check `GET /api/models` → `load_errors`.
2. Confirm `artifacts/models/*.pt` and the `.json` sidecar exist.
3. Reload: `POST /api/models/refresh`.
4. If the file is corrupt, promote the previous registry version or retrain.

### Inference latency p95 rising
1. Check `device` in `/api/health`. Falling back to CPU roughly 10×s latency.
2. Check concurrent training jobs — they compete for the GPU.
3. Grad-CAM and flip-TTA are the expensive options; callers can disable them
   (`explain=false`, `tta=false`) for triage.
4. Inspect `medmlops_prediction_duration_seconds` in Grafana for the per-task split.

### Drift status `watch` / `alert`
1. Open the **Monitoring** tab or `GET /api/monitoring/drift?task=...`.
2. Identify which feature moved (brightness/contrast/entropy/saturation/colorfulness).
3. Compare recent uploads against a demo case — new scanner, different preprocessing or a
   different population are the usual causes.
4. Short-term: keep the model, tighten review. Long-term: retrain on data that includes the new
   distribution and let champion promotion decide.
5. PSI thresholds: `< 0.1` stable, `0.1–0.25` watch, `> 0.25` alert.
6. Status `insufficient_data` means fewer than 150 predictions in the window; a 16-bin PSI needs
   that many samples before the score is meaningful (Laplace smoothing tames sparse bins). Baselines
   are sampled across the whole train split, so `watch` reflects a real feature shift, not row order.

### Feedback “incorrect” rate increasing
1. Export the ledger: `GET /api/monitoring/export.csv`.
2. Group by model version and label; a single version standing out means a bad promotion.
3. Roll back the champion (below) and file the affected cases for the next training set.

### API returns 503 on prediction
No checkpoint for that task. Train one (`make train-seg` / `make train-clf`) or point
`MEDMLOPS_MODEL_DIR` at a directory containing the artifact.

## Champion rollback

```python
import mlflow

client = mlflow.MlflowClient()
client.set_registered_model_alias("isic-lesion-unet", "champion", "3")  # previous good version
```

Then copy the corresponding checkpoint back into `artifacts/models` (MLflow artifact
`checkpoints/`) and call `POST /api/models/refresh`.

CLI equivalent after switching the checkpoint:

```bash
.venv/Scripts/python -m medmlops.cli registry promote --config configs/isic_unet.yaml --run-id <run_id>
```

## Retraining

* Console: **Experiments → Train**, choose a config, add overrides such as `training.epochs=5`;
  the log streams live and models reload on success.
* CLI: `make train-seg` / `make train-clf`.
* CI: run the **Retrain** workflow manually (target + epochs) or wait for the weekly cron.

A new run is promoted to `champion` only when it beats the current champion on the configured
validation metric, so retraining is safe by default.

## Data refresh

```bash
make download      # idempotent; skips existing files
make data          # rebuilds npz splits and the drift reference statistics
```

Rebuilding `reference_stats.json` rebases drift monitoring to the new training cohort — do this
only when the training data itself changed.
