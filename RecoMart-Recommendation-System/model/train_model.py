import json
import logging
import os
import pickle
from typing import Any, Dict, List

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
import mlflow
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

from project_config import FEATURES_PATH, MODEL_METADATA_PATH, MODEL_PATH, ROOT_DIR, ensure_project_dirs


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def precision_at_k(actual: List[str], recommended: List[str], k: int = 5) -> float:
    if not actual:
        return 0.0
    return len(set(recommended[:k]).intersection(actual)) / min(k, len(actual))


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def train_model() -> Dict[str, Any]:
    ensure_project_dirs()
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Training recommendation model")
    features = pd.read_csv(FEATURES_PATH)

    user_item_matrix = features.pivot_table(index="user_id", columns="product_id", values="rating", fill_value=0.0)
    matrix = user_item_matrix.to_numpy()
    nmf = NMF(n_components=3, random_state=42, max_iter=500)
    user_factors = nmf.fit_transform(matrix)
    item_factors = nmf.components_.T
    predictions = user_factors @ item_factors.T

    rmse = float(np.sqrt(mean_squared_error(matrix[matrix > 0], predictions[matrix > 0])))
    top_k = []
    for user_idx, user_id in enumerate(user_item_matrix.index):
        recommended = [str(user_item_matrix.columns[idx]) for idx in np.argsort(predictions[user_idx])[::-1][:5]]
        top_k.append({"user_id": str(user_id), "recommendations": recommended})

    metrics = {"rmse": rmse, "precision_at_5": 0.0}
    metadata = {
        "model": "NMF",
        "parameters": {"n_components": 3, "random_state": 42},
        "metrics": metrics,
        "recommendations": top_k,
    }

    with MODEL_PATH.open("wb") as handle:
        pickle.dump({"nmf": nmf, "user_item_matrix": user_item_matrix, "user_factors": user_factors, "item_factors": item_factors}, handle)

    serializable_metadata = to_serializable(metadata)
    MODEL_METADATA_PATH.write_text(json.dumps(serializable_metadata, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(f"file:///{ROOT_DIR / 'mlruns'}")
    mlflow.set_experiment("recomart-recommendation")
    mlflow.end_run()
    with mlflow.start_run(run_name="recomart-nmf-run") as run:
     
        metadata["mlflow_run_id"] = run.info.run_id

        MODEL_METADATA_PATH.write_text(
        json.dumps(to_serializable(metadata), indent=2),
        encoding="utf-8")
        mlflow.log_params(metadata["parameters"])
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(MODEL_METADATA_PATH))
        mlflow.log_artifact(str(FEATURES_PATH))
        metadata["mlflow_run_id"] = run.info.run_id

    logger.info("Model training completed")
    return metadata


if __name__ == "__main__":
    train_model()
