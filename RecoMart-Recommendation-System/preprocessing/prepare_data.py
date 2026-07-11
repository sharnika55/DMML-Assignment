import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd

from project_config import (
    EDA_PLOTS_DIR,
    PROCESSED_INTERACTIONS_PATH,
    RAW_DIR,
    ensure_project_dirs,
)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_latest_data() -> pd.DataFrame:
    manifest_path = RAW_DIR / "ingestion_manifest.json"
    manifest = pd.read_json(manifest_path)
    return pd.read_csv(RAW_DIR / "ratings" / "2026" / "07" / "11" / "interactions.csv")


def prepare_data() -> Dict[str, Any]:
    ensure_project_dirs()
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Preparing data")

    interactions = pd.read_csv(RAW_DIR / "ratings" / "2026" / "07" / "11" / "interactions.csv")
    products = pd.read_json(RAW_DIR / "api" / "2026" / "07" / "11" / "products.json")

    merged = interactions.merge(products, on="product_id", how="left")
    merged["event_timestamp"] = pd.to_datetime(merged["event_timestamp"], errors="coerce")
    merged["category_code"] = merged["category"].astype("category").cat.codes
    merged["price_scaled"] = (merged["price"] - merged["price"].mean()) / merged["price"].std(ddof=0)
    merged["popularity_scaled"] = (merged["popularity_score"] - merged["popularity_score"].mean()) / merged["popularity_score"].std(ddof=0)
    merged = merged.dropna(subset=["product_id"])

    PROCESSED_INTERACTIONS_PATH.write_text(merged.to_csv(index=False), encoding="utf-8")

    plt.figure(figsize=(8, 4))
    merged["rating"].hist(bins=[1, 2, 3, 4, 5], color="#4C78A8")
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR / "rating_distribution.png")

    plt.figure(figsize=(8, 6))
    corr = merged[["rating", "price_scaled", "popularity_scaled", "category_code"]].corr()
    plt.imshow(corr, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR / "feature_correlation.png")

    logger.info("Data preparation completed")
    return {"prepared_rows": int(len(merged)), "path": str(PROCESSED_INTERACTIONS_PATH)}


if __name__ == "__main__":
    prepare_data()
