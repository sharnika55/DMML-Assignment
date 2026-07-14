# RecoMart Demo Walkthrough

## Duration
5-10 minutes

## Suggested structure
1. Introduce the project goal and business value.
2. Show the repository structure and explain the modular pipeline.
3. Run the pipeline locally using:
   ```bash
   python -m orchestration.pipeline
   ```
4. Highlight the key outputs:
   - raw data in data/raw/
   - prepared data in data/processed/
   - feature table in data/features/
   - validation report in reports/
   - model artifacts in data/models/
5. Explain the orchestration and experiment tracking:
   - Prefect flow run
   - MLflow experiment and metrics
6. Mention next steps such as deployment, real-time ingestion, and API integration.

## Talking points
- The pipeline covers ingestion, validation, preprocessing, feature engineering, feature store, model training, and orchestration.
- The model uses an NMF-based collaborative filtering approach.
- Validation and report generation are automated.
- MLflow records parameters, metrics, and artifacts for reproducibility.

## Expected evidence
- Terminal output showing the Prefect flow completed successfully.
- Generated plots in reports/plots/.
- Validation report PDF in reports/.
- Model metadata JSON in data/models/.
