import json
import logging
from typing import Any, Dict, List

import pandas as pd

from project_config import FEATURES_PATH, FEATURE_STORE_METADATA_PATH, ensure_project_dirs


class FeatureStore:
    def __init__(self, feature_path: str = str(FEATURES_PATH)) -> None:
        self.feature_path = feature_path
        self.metadata_path = FEATURE_STORE_METADATA_PATH
        ensure_project_dirs()

    def save_features(self, features: pd.DataFrame, version: str = "v1") -> Dict[str, Any]:
        features.to_csv(self.feature_path, index=False)
        metadata = {
            "version": version,
            "feature_columns": features.columns.tolist(),
            "rows": int(len(features)),
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata

    def get_feature_vector(self, user_id: str, product_id: str, version: str = "v1") -> Dict[str, Any]:
        features = pd.read_csv(self.feature_path)
        subset = features[(features["user_id"] == user_id) & (features["product_id"] == product_id)]
        if subset.empty:
            return {"status": "not_found"}
        row = subset.iloc[0].to_dict()
        row["version"] = version
        return row


def run_feature_store() -> Dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Registering feature store")
    store = FeatureStore()
    features = pd.read_csv(FEATURES_PATH)
    return store.save_features(features)


if __name__ == "__main__":
    run_feature_store()
