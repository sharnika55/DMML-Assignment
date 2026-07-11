import logging
import os
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow
from prefect import flow, get_run_logger, task

from feature_engineering.engineer_features import engineer_features
from feature_store.feature_store import run_feature_store
from ingestion.ingest_data import run_ingestion
from model.train_model import train_model
from preprocessing.prepare_data import prepare_data
from project_config import ROOT_DIR, ensure_project_dirs
from validation.validate_data import run_validation


@task(retries=2, retry_delay_seconds=3)
def ingest_stage() -> Dict[str, Any]:
    return run_ingestion()


@task(retries=2, retry_delay_seconds=3)
def validate_stage(ingestion_output: Dict[str, Any]) -> Dict[str, Any]:
    return run_validation()


@task(retries=2, retry_delay_seconds=3)
def prepare_stage(validation_output: Dict[str, Any]) -> Dict[str, Any]:
    return prepare_data()


@task(retries=2, retry_delay_seconds=3)
def feature_stage(preparation_output: Dict[str, Any]) -> Dict[str, Any]:
    return engineer_features()


@task(retries=2, retry_delay_seconds=3)
def feature_store_stage(feature_output: Dict[str, Any]) -> Dict[str, Any]:
    return run_feature_store()


@task(retries=2, retry_delay_seconds=3)
def train_stage(feature_store_output: Dict[str, Any]) -> Dict[str, Any]:
    return train_model()


@flow(name="recomart-prefect-pipeline")
def run_prefect_pipeline() -> Dict[str, Any]:
    ensure_project_dirs()
    logger = get_run_logger()
    logger.info("Starting Prefect orchestration")

    mlflow.set_tracking_uri(f"file:///{ROOT_DIR / 'mlruns'}")
    mlflow.set_experiment("recomart-recommendation")

    ingestion_output = ingest_stage()
    validation_output = validate_stage(ingestion_output)
    preparation_output = prepare_stage(validation_output)
    feature_output = feature_stage(preparation_output)
    feature_store_output = feature_store_stage(feature_output)
    model_output = train_stage(feature_store_output)

    logger.info("Prefect orchestration completed")
    return {
        "ingestion": ingestion_output,
        "validation": validation_output,
        "preprocessing": preparation_output,
        "features": feature_output,
        "feature_store": feature_store_output,
        "model": model_output,
    }


if __name__ == "__main__":
    run_prefect_pipeline()
