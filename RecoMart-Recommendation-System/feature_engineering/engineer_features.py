import logging
from typing import Dict, Any

import pandas as pd

from project_config import FEATURES_PATH, PROCESSED_INTERACTIONS_PATH, ensure_project_dirs


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def engineer_features() -> Dict[str, Any]:
    ensure_project_dirs()
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Engineering features")
    df = pd.read_csv(PROCESSED_INTERACTIONS_PATH)

    user_activity = df.groupby("user_id").size().rename("user_activity_frequency")
    user_rating_avg = df.groupby("user_id")["rating"].mean().rename("user_avg_rating")
    item_rating_avg = df.groupby("product_id")["rating"].mean().rename("item_avg_rating")
    item_popularity = df.groupby("product_id").size().rename("item_popularity")

    feature_frame = df[["user_id", "product_id", "rating", "price_scaled", "popularity_scaled", "category_code"]].copy()
    feature_frame = feature_frame.merge(user_activity, on="user_id", how="left")
    feature_frame = feature_frame.merge(user_rating_avg, on="user_id", how="left")
    feature_frame = feature_frame.merge(item_rating_avg, on="product_id", how="left")
    feature_frame = feature_frame.merge(item_popularity, on="product_id", how="left")
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_frame.to_csv(FEATURES_PATH, index=False)
    logger.info("Feature engineering completed")
    return {"feature_rows": int(len(feature_frame)), "path": str(FEATURES_PATH)}


if __name__ == "__main__":
    engineer_features()
