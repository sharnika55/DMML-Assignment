# RecoMart Recommendation System

This repository contains a modular end-to-end recommendation pipeline for RecoMart. It includes:

- data ingestion from the MovieLens 100k public dataset (with local fallback)
- validation with automated quality checks and PDF reporting
- preprocessing and EDA plot generation
- feature engineering for collaborative-style recommendations
- a simple feature store with metadata
- model training using NMF and experiment metadata
- an orchestration entry point for the full pipeline

## 1) Python environment setup

Recommended Python versions:
- `3.10` or `3.11` for full local stack (including Airflow package import)
- `3.12+` can run core pipeline, tests, DVC, and MLflow parts, but Airflow should be run via Docker Compose

From project root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notes:
- `requirements.txt` installs Airflow only on supported Python versions (`<3.12`).
- If Airflow is skipped due to Python version, use Docker for Airflow UI/DAG execution.

## 2) Download public dataset (required)

```bash
python -m ingestion.download_public_dataset
```

Expected dataset location:
- `data/raw/public/ml-100k/ml-100k/u.data`
- `data/raw/public/ml-100k/ml-100k/u.item`

## 3) Run the full pipeline

```bash
python -m orchestration.pipeline
```

## 4) Validate outputs

Check logs:
- `logs/ingestion.log`
- `logs/pipeline.log`

Check data artifacts:
- `data/raw/ingestion_manifest.json` (latest partition paths)
- `data/processed/prepared_interactions.csv`
- `data/features/feature_table.csv`
- `data/features/feature_store_metadata.json`
- `data/models/model_metadata.json`
- `data/models/recommendation_model.pkl`

Check reports:
- `reports/validation_report.json`
- `reports/data_quality_report.pdf`
- `reports/plots/rating_distribution.png`
- `reports/plots/feature_correlation.png`

Run tests:

```bash
python -m pytest -q
```

Validate DVC stage reproducibility:

```bash
python -m dvc repro
```

Optional Airflow import check (when using Python 3.10/3.11):

```bash
python -c "import airflow; print('airflow_import_ok')"
```

## 5) Inference interface

After training, run batch inference for a user via:

```bash
python -m model.inference --user-id 1 --k 5
```

This returns top-k recommended product ids in JSON format.

## Key outputs

- Raw data partitions in data/raw/
- Prepared datasets in data/processed/
- Engineered features in data/features/
- Model artifacts in data/models/
- Logs in logs/
- Reports in reports/

## Local Airflow & MLflow (quick start)

Start a local Airflow UI and MLflow server using Docker Compose (this uses `airflow standalone` for simplicity):

```bash
cd RecoMart-Recommendation-System
docker-compose up --build
```

- Airflow UI: http://localhost:8080 — open the `recomart_end_to_end` DAG to see per-task status and XComs.
- MLflow UI: http://localhost:5000 — view experiments and logged artifacts.

Notes:
- The DAG file is located at `dags/recomart_dag.py` and invokes the repository's ingestion/validation/prepare/feature/train modules.
- The ingestion task returns a manifest path and file list to XCom so you can inspect which datasets were used for that run.

