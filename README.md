# MedMLOps

**A production-shaped MLOps platform for medical imaging** — U-Net lesion segmentation, DenseNet-121
chest X-ray triage, MLflow tracking + registry, a custom clinical console, live drift monitoring and
a deployment path that runs from a laptop to a container stack.

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](#requirements)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![License](https://img.shields.io/badge/license-MIT-2ea44f)](LICENSE)

> [!WARNING]
> **Research and education only — not a medical device.** Outputs are assistive signals and must be
> reviewed by a qualified clinician before any diagnostic or treatment decision.

**Jump to:** [Why this project](#why-this-project) · [Architecture](#architecture) ·
[Quickstart](#quickstart) · [Results](#verified-results) · [Console](#the-console) ·
[MLflow & DagsHub](#mlflow--dagshub-workflow) · [API](#serving-api) · [Deployment](#deployment) ·
[Operations](#operations) · [Troubleshooting](#troubleshooting) · [Configuration](#configuration-reference)

---

## Why this project

Most portfolio MLOps repos stop at “train + log a metric”. MedMLOps follows the whole lifecycle and
keeps the operational seams honest:

| # | Capability | What it actually does |
| --- | --- | --- |
| 1 | **Two imaging modalities, one service** | U-Net segmentation (ISIC 2018) and DenseNet-121 classification (PneumoniaMNIST) share one predictor, ledger and console. |
| 2 | **Clinical console, not a notebook dump** | React 18 + TypeScript: drag-drop intake, canvas overlays, uncertainty layer, Grad-CAM, live training log, drift dashboard. |
| 3 | **Registry-driven deployment** | Models are registered in MLflow with a moving `champion` alias. `POST /api/models/sync` (or startup bootstrap) mirrors champions into the serving directory — no retraining required. |
| 4 | **One-flag DagsHub switch** | `MEDMLOPS_TRACKER=dagshub` or `--dagshub` streams runs, artifacts and registry entries to DagsHub-hosted MLflow. |
| 5 | **Drift + human feedback loop** | Every prediction logs interpretable image features; PSI against training baselines; clinician verdicts are joined back to ledger rows. |
| 6 | **Operable SQLite ledger** | WAL mode, indexes, online backup, retention pruning, streaming CSV export, readiness probe and database ping. |
| 7 | **Deployment-ready containers** | Multi-stage API/web/MLflow images, healthchecks, `depends_on: service_healthy`, nginx reverse proxy, Prometheus + provisioned Grafana. |
| 8 | **Optional API-key security** | Set `MEDMLOPS_API_KEY` and every `/api` route except health/ready requires `X-API-Key`; the console prompts for it. |

---

## Architecture

```mermaid
flowchart LR
    subgraph Data
        A[ISIC 2018<br/>images + masks] --> C[prepare<br/>npz splits]
        B[PneumoniaMNIST<br/>chest X-rays] --> C
        C --> D[reference_stats.json<br/>drift baseline]
    end

    subgraph Train
        C --> E[U-Net<br/>BCE + Dice · AMP]
        C --> F[DenseNet-121<br/>ImageNet transfer]
        E --> G[MLflow<br/>runs · artifacts · registry]
        F --> G
        G --> H[(champion alias)]
    end

    subgraph Serve
        G -->|sync / bootstrap| I[artifacts/models]
        I --> J[FastAPI]
        J --> K[React console<br/>overlays · Grad-CAM · drift]
        J --> L[(SQLite ledger<br/>feedback · features)]
        J --> M[/metrics]
        M --> N[Prometheus → Grafana]
        H -.-> K
    end
```

The lifecycle in one line: **DVC prepares data → training logs to MLflow and registers versions →
champion promotion picks the best run → the API mirrors champions → the console and ledger close the
loop with feedback → drift and metrics tell you when to retrain.**

<details>
<summary><b>Repository layout</b> (click to expand)</summary>

```
configs/              YAML for both tasks: data, model, training, output, MLflow
src/medmlops/
  data/               ISIC + MedMNIST loaders, transforms, drift baselines
  models/             U-Net and DenseNet-121
  train.py            shared engine: AMP, warmup+cosine, early stop, MLflow, model cards
  evaluate.py         held-out evaluation from a checkpoint
  tracking.py         MLflow, registry, champion alias, DagsHub, registry mirroring
  monitor.py          prediction ledger, feedback, PSI drift, backup/prune
  predictor.py        inference logic shared by API and CLI
  explain.py          Grad-CAM and flip-TTA uncertainty
  cli.py              data · train · evaluate · predict · registry · db · serve
api/                  FastAPI app, routers, background jobs, Prometheus, auth guard
web/                  React 18 + Vite + TypeScript console
docker/               Dockerfiles, nginx, Prometheus + Grafana provisioning
dvc.yaml              Reproducible pipeline
tests/                Unit, integration and API contract tests (61 tests)
scripts/smoke_test.py Live end-to-end verification
docs/                 Runbook + model-card notes
```

</details>

---

## Quickstart

### Requirements

- Python **3.11+** (3.12 recommended), [uv](https://docs.astral.sh/uv/), Node **20+**
- Optional: Docker Desktop / Engine + Compose v2 for the container stack; NVIDIA GPU for fast training
  (the service also runs on CPU)

On Windows use **Git Bash**; the Makefile selects `.venv/Scripts/python.exe` automatically.

### Path A — run the platform locally

```bash
uv venv --python 3.12 .venv
make install                 # Python extras + web dependencies
make download                # ISIC mirror + PneumoniaMNIST (~3 GB total)
make data                    # npz splits, manifests, drift baselines
make train-seg               # U-Net (GPU strongly recommended)
make train-clf               # DenseNet-121
make web-build               # type-check and bundle the console
make serve                   # http://127.0.0.1:8000
```

Open **http://127.0.0.1:8000** for the console, **/docs** for OpenAPI, **/metrics** for Prometheus.
`make mlflow` serves the MLflow UI on **:5000**.

### Path B — develop with hot reload

```bash
make api        # terminal 1: FastAPI on :8000
make dev        # terminal 2: Vite on :5173 (proxies /api)
```

### Path C — container stack

```bash
cp .env.example .env
docker compose up --build
# console     http://localhost:8080
# API docs    http://localhost:8000/docs
# MLflow      http://localhost:5000
# Grafana     http://localhost:3000   (admin / medmlops)
# Prometheus  http://localhost:9090
```

### Verify everything

```bash
make test                    # pytest (61 tests, CPU-only, tiny checkpoints)
make lint                    # ruff
python scripts/smoke_test.py # live inference, ledger, export, readiness
make db-stats                # ledger rows, size, WAL mode
```

`smoke_test.py` accepts `--base-url` and `--api-key` (defaults to `MEDMLOPS_API_KEY`), so it works
against local, compose and protected deployments alike.

---

## Verified results

Both models were trained end to end on a single RTX 3050 Laptop (6 GB), registered in MLflow and
promoted to `champion`. Metrics are from the held-out test split and reproducible with `dvc repro`:

| Task | Model | Test set | Headline metrics | Training |
| --- | --- | --- | --- | --- |
| Lesion segmentation (ISIC 2018) | U-Net, base 32 · 7.76M params | 412 images | **Dice 0.875** · IoU 0.801 · precision 0.892 · recall 0.898 · pixel acc 0.955 | 20 epochs · 23.7 min |
| Pneumonia triage (PneumoniaMNIST) | DenseNet-121, ImageNet init · 6.96M params | 624 images | **Accuracy 0.9615** · AUC 0.9837 · sensitivity 0.9795 · specificity 0.9316 · ECE 0.0356 | 6 epochs (best 2) · 9.3 min |

Raw numbers land in `metrics/*.json`; curves, ROC, confusion matrices and prediction grids in
`artifacts/figures/`; model cards and full run metadata live inside each MLflow run.

<details>
<summary><b>Datasets</b> (click to expand)</summary>

- **ISIC 2018** — 2,594 lesion images with expert masks. Default source is a fast Hugging Face mirror
  (`sergiomadrid/isic_segmentation`); set `data.source: official` for the ~11 GB official archive.
- **PneumoniaMNIST** — 5,856 pediatric chest X-rays, downloaded via the `medmnist` package. Other
  MedMNIST datasets work by changing `data.name`.

Both prep stages write an `npz` split bundle, `manifest.json` (counts, class balance) and
`reference_stats.json` (per-feature histograms consumed by drift monitoring).

</details>

---

## The console

| Tab | What you can do |
| --- | --- |
| **Analyze** | Drag-drop or load a held-out demo case; toggle flip-TTA and Grad-CAM; inspect overlay, hard mask, uncertainty and probability bars; submit clinician feedback. |
| **Models** | Loaded checkpoints, test metrics, parameter counts, run metadata, registry champions, **Sync registry** to pull champions from MLflow. |
| **Experiments** | MLflow runs with metric sparklines, parameter diffs, run links; **Train** launches a background job with a live log console and auto-reloads models on success. |
| **Monitoring** | Volume, p95 latency, agreement, drift status; PSI bars per feature; label mix; ledger storage panel (WAL, rows, size, per-task export, retention prune). |

The interface is responsive, keyboard-reachable, and uses a dark clinical theme with accessible
contrast. When `MEDMLOPS_API_KEY` is set, a banner asks for the key once and stores it locally.

---

## MLflow & DagsHub workflow

### Local (default)

No server required: runs live in `mlflow.db` (SQLite) with artifacts under `mlartifacts/`.

```bash
make mlflow      # UI on :5000 with artifact proxying (--serve-artifacts)
```

Every run logs parameters, per-epoch metrics, plots, the checkpoint, generated `model_card.md`, and
registers a model version. A run becomes **champion** only if it beats the current champion on
`val_dice` / `val_auc`.

```bash
python -m medmlops.cli registry list
python -m medmlops.cli registry champion --config configs/isic_unet.yaml
python -m medmlops.cli registry promote  --config configs/isic_unet.yaml --run-id <run_id>
python -m medmlops.cli registry export   --config configs/isic_unet.yaml --dest artifacts/models
python -m medmlops.cli registry sync     --dest artifacts/models
```

### MLflow server

Point both training and serving at the same server:

```dotenv
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Compose starts a server backed by a volume and configured with `--serve-artifacts`, so clients never
need shared filesystem access.

### DagsHub

This project uses [Arman734/MLOpsProject](https://dagshub.com/Arman734/MLOpsProject) for both
the DVC data remote and hosted MLflow. Put credentials in the ignored `.env` file:

```dotenv
DAGSHUB_USER_NAME=Arman734
DAGSHUB_REPO_NAME=MLOpsProject
DAGSHUB_USER_TOKEN=********
MEDMLOPS_TRACKER=dagshub
```

The DVC remote is versioned in `.dvc/config`; add credentials to the ignored local config and upload
the pipeline outputs:

```bash
set -a
source .env
set +a
dvc remote modify --local dagshub auth basic
dvc remote modify --local dagshub user "$DAGSHUB_USER_NAME"
dvc remote modify --local dagshub password "$DAGSHUB_USER_TOKEN"
dvc repro
dvc push
```

Train as usual — `dagshub.init(...)` resolves the hosted MLflow URI; runs, artifacts and registry
entries stream to DagsHub:

```bash
python -m medmlops.cli train --config configs/isic_unet.yaml --dagshub
```

The GitHub Actions **Retrain models** workflow also requires repository secrets
`DAGSHUB_USER_NAME`, `DAGSHUB_REPO_NAME`, and `DAGSHUB_USER_TOKEN`. It prepares and pushes DVC data,
logs training and evaluation to DagsHub MLflow, and commits updated `dvc.lock` metadata. The
optional `HF_TOKEN` secret is used only when a dataset host requires it.

Deploy from the registry: the API only needs `MLFLOW_TRACKING_URI` (resolved from DagsHub) plus
`MEDMLOPS_BOOTSTRAP_FROM_REGISTRY=1`, and it downloads champion checkpoints at startup.

Under the hood, artifact locations are backend-aware: local SQLite/file stores get an explicit
`MLFLOW_ARTIFACT_ROOT`, while remote servers and DagsHub own their artifact store (passing a
client-side path there would break uploads). Missing DagsHub credentials fail fast with a clear error
instead of silently falling back to SQLite.

---

## Serving API

Base URL: `http://127.0.0.1:8000`. Interactive schema at **/docs**; machine schema at
**/openapi.json**.

<details open>
<summary><b>Endpoint reference</b></summary>

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Liveness, device, model readiness, database ping, auth mode |
| GET | `/api/ready` | Readiness probe (200 ready / 503 not ready) for orchestrators |
| GET | `/api/system` | Versions, tracking backend, dataset summaries |
| GET | `/api/models` | Loaded checkpoints + registry versions |
| POST | `/api/models/refresh` | Reload checkpoints from disk |
| POST | `/api/models/sync` | Mirror registry champions into the model directory |
| POST | `/api/models/promote` | Point `champion` at a run |
| GET | `/api/models/champion/{name}` | Champion metrics |
| GET | `/api/demo/{task}` | Held-out test image for instant demos |
| POST | `/api/predict/segmentation` | Mask, overlay, uncertainty, drift feature vector |
| POST | `/api/predict/classification` | Probabilities and optional Grad-CAM |
| POST | `/api/feedback` | Clinician verdict (`correct` / `incorrect` / `uncertain`) + comment |
| GET | `/api/predictions/recent` | Prediction ledger (optional `task` filter) |
| GET | `/api/predictions/{id}` | One ledger row with features, metadata and feedback |
| GET | `/api/monitoring/summary` | Volumes, latency p95, label mix, drift status |
| GET | `/api/monitoring/drift` | Per-feature PSI with thresholds |
| GET | `/api/monitoring/storage` | DB path, size, rows, journal mode (WAL) |
| POST | `/api/monitoring/prune` | Retention: delete predictions older than `days` |
| GET | `/api/monitoring/export.csv` | Streaming CSV with latest clinician verdict |
| GET | `/api/experiments` · `/api/experiments/{run_id}` | MLflow runs and history |
| GET | `/api/registry` | Registered models and champions |
| GET | `/api/pipeline/configs` | Trainable YAML configs |
| POST | `/api/pipeline/train` | Launch a background training job |
| GET | `/api/pipeline/jobs` · `/api/pipeline/jobs/{id}` | Job list / status + logs |
| POST | `/api/pipeline/jobs/{id}/cancel` | Cancel a job |
| GET | `/metrics` | Prometheus exposition |

</details>

### Authentication

Auth is **off by default** for local development. Enable it for any shared deployment:

```bash
export MEDMLOPS_API_KEY=change-me
make serve
```

- Every `/api` route except `/api/health` and `/api/ready` then requires `X-API-Key: <key>`
  (or `Authorization: Bearer <key>`).
- Requests are compared with `hmac.compare_digest`; responses carry security headers
  (`nosniff`, `DENY` framing, no-referrer, restrictive permissions policy).
- The console detects the 401 and shows an unlock form; the key is stored in `localStorage`.

```bash
curl -H "X-API-Key: $MEDMLOPS_API_KEY" http://127.0.0.1:8000/api/system
python scripts/smoke_test.py --api-key "$MEDMLOPS_API_KEY"
```

### Limits and safety

- Uploads are streamed with a pixel-count guard and decompression-bomb protection; oversized bodies
  return **413** before they reach the model.
- Both responses include latency and model version; the ledger stores image features for drift.
- Prediction endpoints return **503** with a clear message when no checkpoint exists for the task.

---

## Deployment

### Option 1 — Docker Compose (single host)

```bash
cp .env.example .env          # set MEDMLOPS_API_KEY, optionally DAGSHUB_*
docker compose up --build -d
docker compose ps             # every service should be healthy
```

- `api` waits for a healthy `mlflow`; `web`, `prometheus` wait for a healthy `api`.
- Checkpoints and the ledger persist in `./artifacts`; prepared data is mounted read-only for the
  console’s **Train** action.
- The API image bundles CPU PyTorch for inference. Console-launched retraining therefore runs on CPU
  inside compose — use a GPU host, the CLI, or the **Retrain** workflow for heavy jobs.
- The web container (nginx) proxies `/api`, `/docs`, `/redoc`, `/openapi.json` and `/metrics` to the
  API and serves the SPA for everything else.
- Keep Grafana credentials out of source: `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD`.

### Option 2 — stateless service + external registry

This is the production-shaped path: the container holds **no model artifacts**, only code.

```dotenv
MEDMLOPS_API_KEY=********
MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow   # or an MLflow server
DAGSHUB_USER_NAME=<user>
DAGSHUB_REPO_NAME=<repo>
DAGSHUB_USER_TOKEN=********
MEDMLOPS_BOOTSTRAP_FROM_REGISTRY=1
MEDMLOPS_REGISTRY_NAMES=isic-lesion-unet,pneumonia-densenet
```

On startup the API downloads each champion’s `checkpoints/*.pt` + sidecar from the registry into
`MEDMLOPS_MODEL_DIR`, refreshes predictors and Prometheus gauges. Readiness stays **503** until both
models load, so a load balancer will not send traffic too early.

For DagsHub URIs, `DAGSHUB_USER_NAME` / `DAGSHUB_USER_TOKEN` are reused as MLflow basic auth
automatically; for any other protected server set `MLFLOW_TRACKING_USERNAME` /
`MLFLOW_TRACKING_PASSWORD` explicitly.

> Keep the SQLite ledger on a persistent volume (`MEDMLOPS_DB_PATH`) and back it up with
> `medmlops db backup` or `POST /api/monitoring/prune` for retention.

### Option 3 — training job + deployment handoff

Training runs (CLI, console or `retrain.yml`) register a version and may promote it to `champion`.
Deployment code does not change: restart the API (or call `POST /api/models/sync`) to adopt the new
champion. Roll back by moving the alias to a previous version and syncing again.

<details>
<summary><b>Production checklist</b> (click to expand)</summary>

- [ ] `MEDMLOPS_API_KEY` set and stored as a secret
- [ ] TLS terminated at the ingress/reverse proxy; `--proxy-headers` is enabled in the API image
- [ ] `MEDMLOPS_BOOTSTRAP_FROM_REGISTRY=1` or a mounted artifact volume
- [ ] Ledger volume backed up on a schedule; retention policy applied
- [ ] `/api/ready` used as the readiness probe, `/api/health` as liveness
- [ ] Prometheus scraping `/metrics`; Grafana dashboard provisioned
- [ ] `MEDMLOPS_DEVICE` pinned (`cpu` in containers without a GPU runtime)
- [ ] Model cards and metrics reviewed before promoting a champion
- [ ] Console served over HTTPS; API key rotated on team changes

</details>

### CI/CD

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `ci.yml` | push / PR to `main` or `master` | Ruff, pytest (CPU torch), frontend type-check + build, `docker compose config -q` |
| `docker.yml` | push to `main`/`master`, tags, manual | Builds and pushes API, web and MLflow images to GHCR |
| `retrain.yml` | weekly cron (`0 3 * * 0`) or manual | Prepares data, retrains (DagsHub when the secret exists), evaluates, uploads artifacts |

> Docker images cannot be built in this development environment (no Docker CLI); the compose file
> is validated statically in CI, and the images are built on GitHub Actions.

---

## Operations

| Task | Command |
| --- | --- |
| Health / readiness | `curl /api/health` · `curl /api/ready` |
| Ledger stats | `make db-stats` or `python -m medmlops.cli db stats` |
| Consistent backup | `make db-backup` (SQLite online backup API, safe while serving) |
| Retention | `POST /api/monitoring/prune?days=30` or `python -m medmlops.cli db prune --days 30` |
| Export ledger | `GET /api/monitoring/export.csv` |
| Sync champions | `make sync-models` or **Models → Sync registry** |
| Roll back champion | Move the alias with `mlflow.MlflowClient().set_registered_model_alias(...)`, then sync |
| Drift response | Monitoring tab → identify feature → tighten review → retrain |

Operational playbooks: [docs/runbook.md](docs/runbook.md). Metric names:
`medmlops_api_requests_total`, `medmlops_api_request_duration_seconds`,
`medmlops_predictions_total`, `medmlops_prediction_duration_seconds`, `medmlops_model_ready`.

### Monitoring & drift, precisely

- **Ledger** — every prediction stores model id/version, latency, label, confidence, image features
  and optional feedback.
- **Drift** — PSI compares live feature windows to training baselines: `< 0.10` stable,
  `0.10–0.25` watch, `> 0.25` alert. Scores remain `insufficient_data` until 150 predictions
  accumulate (a 16-bin PSI needs that many samples). Baselines are sampled across the whole train
  split, not the first N rows, and features are extracted at each model’s training resolution.
- **Feedback** — verdicts join to ledger rows and feed the agreement statistic in the console.
- **Storage** — WAL mode, `synchronous=NORMAL`, `busy_timeout`, task/feedback indexes, online backup,
  retention prune and streaming export keep SQLite predictable under concurrent reads.

---

## Testing & reproducibility

```bash
make test        # 61 tests: config, models, metrics, drift, transforms, explainability, API, tracking
make lint        # ruff (E, F, I, UP, B)
make smoke       # live end-to-end smoke test
```

The API suite builds tiny real checkpoints on CPU, so it exercises actual inference, explainability
and ledger behaviour without downloading the datasets.

```bash
make repro           # DVC: download → prepare → train → evaluate
dvc status           # what is stale (with the venv active)
dvc metrics show     # metrics/*.json
```

Any config key can be overridden without editing YAML:

```bash
python -m medmlops.cli train --config configs/isic_unet.yaml \
  --set training.epochs=2 data.max_samples=128 training.batch_size=8
```

---

## Troubleshooting

<details>
<summary><b>MLflow ignores my MLFLOW_* / DAGSHUB_* values</b></summary>

`.env` is loaded at import time as process-environment defaults (`load_env_defaults`), and real
environment variables always win. If a value still looks ignored, confirm the variable is not empty
in `.env` and restart the process — environment overrides are read once per process.
</details>

<details>
<summary><b>“DagsHub tracking requested but … not set”</b></summary>

DagsHub is enabled (`MEDMLOPS_TRACKER=dagshub` or `--dagshub`) but one of
`DAGSHUB_USER_NAME`, `DAGSHUB_REPO_NAME`, `DAGSHUB_USER_TOKEN` is missing. This is deliberate: the
run fails fast instead of silently writing to local SQLite.
</details>

<details>
<summary><b>Predictions return 503 “train one first”</b></summary>

No checkpoint for that task in `MEDMLOPS_MODEL_DIR`. Either train (`make train-seg` / `make
train-clf`), sync from the registry (`make sync-models`), or enable
`MEDMLOPS_BOOTSTRAP_FROM_REGISTRY=1` with a reachable tracking URI.
</details>

<details>
<summary><b>The console asks for an API key</b></summary>

`MEDMLOPS_API_KEY` is set on the server. Paste the same value into the banner; it is stored in
`localStorage`. Remove the key from the server environment to disable auth again.
</details>

<details>
<summary><b>The MLflow UI shows experiments but artifacts are missing</b></summary>

Local experiments store artifacts under `MLFLOW_ARTIFACT_ROOT` (default `mlartifacts/`). When using a
remote server, run it with `--serve-artifacts` (the bundled image does) and leave the client artifact
root unset. DagsHub manages artifacts remotely by design.
</details>

<details>
<summary><b>Compose services stay in “starting”</b></summary>

The stack gates on healthchecks. Check `docker compose logs mlflow` and `docker compose logs api`.
Common causes: a slow first model load (readiness 503 until both models exist), a missing `artifacts`
volume, or a port already in use.
</details>

<details>
<summary><b>SQLite files: `monitoring.db-wal` / `-shm`</b></summary>

Those files are expected in WAL mode. Do not copy the live `.db` file; use
`make db-backup` / `POST /api/monitoring/prune` or the CLI, which use SQLite’s online backup API.
</details>

<details>
<summary><b>Training is slow / CUDA is not detected</b></summary>

`/api/health` reports the active device. Install a CUDA build of PyTorch matching your driver, or set
`MEDMLOPS_DEVICE=cpu` in containers. Training jobs compete with inference for the GPU — expect p95
latency to rise while a job runs.
</details>

---

## Configuration reference

Copy `.env.example` to `.env`. Environment variables override `.env`, which overrides defaults.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | Tracking backend (SQLite, HTTP server or DagsHub URI) |
| `MLFLOW_ARTIFACT_ROOT` | `./mlartifacts` | Artifact store for local file/SQLite backends |
| `MEDMLOPS_TRACKER` | `local` | `dagshub` enables the DagsHub workflow |
| `DAGSHUB_USER_NAME` / `DAGSHUB_REPO_NAME` / `DAGSHUB_USER_TOKEN` | — | DagsHub credentials |
| `MEDMLOPS_API_KEY` | empty | Enables API-key auth when set |
| `MEDMLOPS_MODEL_DIR` | `./artifacts/models` | Serving checkpoint directory |
| `MEDMLOPS_DB_PATH` | `./artifacts/monitoring.db` | SQLite prediction ledger |
| `MEDMLOPS_BOOTSTRAP_FROM_REGISTRY` | `0` | Pull champions at startup when the model dir is empty |
| `MEDMLOPS_REGISTRY_NAMES` | `isic-lesion-unet,pneumonia-densenet` | Registry names to mirror |
| `MEDMLOPS_MAX_UPLOAD_MB` | `15` | Upload limit (images are validated before inference) |
| `MEDMLOPS_DEVICE` | `auto` | `auto`, `cuda`, `cpu` |
| `MEDMLOPS_CORS_ORIGINS` | local Vite origins | Comma-separated console origins |
| `MEDMLOPS_TRACKING_URI` | empty | In-cluster MLflow URL used by compose |
| `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD` | `admin` / `medmlops` | Compose Grafana credentials |

---

## Limitations & next steps

- Validation is retrospective on public research datasets; there is no prospective clinical study.
- Drift monitoring covers input-feature shift (PSI), not label shift or silent concept drift.
- SQLite is the right ledger for a single instance; move to Postgres for horizontal scale-out.
- Explainability is Grad-CAM/flip-TTA, useful for review but not a causal explanation.
- Natural next steps: DICOM ingestion, PACS integration, background quality checks, and
  per-site drift baselines.

## License

MIT. Datasets retain their original licenses (ISIC 2018, MedMNIST). Model weights and figures are
generated artifacts and are not committed.
