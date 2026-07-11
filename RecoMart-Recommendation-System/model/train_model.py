import json
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

from project_config import FEATURES_PATH, MODEL_METADATA_PATH, MODEL_PATH, ensure_project_dirs


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def precision_at_k(actual: List[str], recommended: List[str], k: int = 5) -> float:
    if not actual:
        return 0.0
    return len(set(recommended[:k]).intersection(actual)) / min(k, len(actual))


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
        recommended = [user_item_matrix.columns[idx] for idx in np.argsort(predictions[user_idx])[::-1][:5]]
        top_k.append((user_id, recommended))

    metrics = {"rmse": rmse, "precision_at_5": 0.0}
    metadata = {
        "model": "NMF",
        "parameters": {"n_components": 3, "random_state": 42},
        "metrics": metrics,
        "recommendations": top_k,
    }

    import pickle

    with MODEL_PATH.open("wb") as handle:
        pickle.dump({"nmf": nmf, "user_item_matrix": user_item_matrix, "user_factors": user_factors, "item_factors": item_factors}, handle)

    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Model training completed")
    return metadata


if __name__ == "__main__":
    train_model()
