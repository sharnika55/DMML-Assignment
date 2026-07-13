from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Any

from airflow import DAG
from airflow.decorators import task

ROOT_START = datetime(2026, 7, 1)

default_args = {
    "owner": "recomart",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


with DAG(
    dag_id="recomart_end_to_end",
    default_args=default_args,
    start_date=ROOT_START,
    schedule_interval="@daily",
    catchup=False,
    tags=["recomart"],
) as dag:

    @task
    def ingest() -> Dict[str, Any]:
        # imports inside tasks to avoid top-level Airflow import issues
        from ingestion.ingest_data import run_ingestion

        out = run_ingestion()
        # return manifest path and file list for visibility in the UI
        return {"manifest": str(out.get("manifest")), "files": {k: str(v) for k, v in out.items()}}

    @task
    def validate(ingest_result: Dict[str, Any]) -> Dict[str, Any]:
        from validation.validate_data import run_validation

        return run_validation()

    @task
    def prepare(validate_result: Dict[str, Any]) -> Dict[str, Any]:
        from preprocessing.prepare_data import prepare_data

        return prepare_data()

    @task
    def features(prepare_result: Dict[str, Any]) -> Dict[str, Any]:
        from feature_engineering.engineer_features import engineer_features

        return engineer_features()

    @task
    def register_features(features_result: Dict[str, Any]) -> Dict[str, Any]:
        from feature_store.feature_store import run_feature_store

        return run_feature_store()

    @task
    def train(register_result: Dict[str, Any]) -> Dict[str, Any]:
        from model.train_model import train_model

        return train_model()

    ingestion = ingest()
    validation = validate(ingestion)
    preparation = prepare(validation)
    engineered = features(preparation)
    fstore = register_features(engineered)
    model = train(fstore)

    ingestion >> validation >> preparation >> engineered >> fstore >> model
