# RecoMart Recommendation System

This repository contains a modular end-to-end recommendation pipeline for RecoMart. It includes:

- data ingestion from local CSV and JSON sources
- validation with automated quality checks and PDF reporting
- preprocessing and EDA plot generation
- feature engineering for collaborative-style recommendations
- a simple feature store with metadata
- model training using NMF and experiment metadata
- an orchestration entry point for the full pipeline

## Run the pipeline

```bash
python main.py
```

## Public dataset option

For a more widely recognized recommendation dataset, the project can also download the MovieLens 100k dataset from GroupLens:

```bash
python ingestion/download_public_dataset.py
```

This stores the downloaded files under data/raw/public/ml-100k/.

## Key outputs

- Raw data partitions in data/raw/
- Prepared datasets in data/processed/
- Engineered features in data/features/
- Model artifacts in data/models/
- Logs in logs/
- Reports in reports/
