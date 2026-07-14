import json
import logging
from pathlib import Path
from typing import Any, Dict
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from project_config import (
    EDA_PLOTS_DIR,
    PROCESSED_INTERACTIONS_PATH,
    RAW_DIR,
    ensure_project_dirs,
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def get_latest_paths():
    """
    Read the latest dataset locations from ingestion_manifest.json
    """
    manifest_path = RAW_DIR / "ingestion_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest file not found: {manifest_path}"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    ratings_path = None
    products_path = None

    for item in manifest["files"]:
        if item["source"] == "ratings":
            ratings_path = Path(item["path"])

        elif item["source"] == "api":
            products_path = Path(item["path"])

    if ratings_path is None:
        raise FileNotFoundError("Ratings file not found in manifest.")

    if products_path is None:
        raise FileNotFoundError("Products file not found in manifest.")

    return ratings_path, products_path


def load_latest_data() -> pd.DataFrame:
    ratings_path, _ = get_latest_paths()
    return pd.read_csv(ratings_path)


def prepare_data() -> Dict[str, Any]:
    ensure_project_dirs()
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Preparing data...")

    ratings_path, products_path = get_latest_paths()

    interactions = pd.read_csv(ratings_path)
    products = pd.read_json(products_path)

    merged = interactions.merge(
        products,
        on="product_id",
        how="left",
    )

    merged["event_timestamp"] = pd.to_datetime(
        merged["event_timestamp"],
        errors="coerce",
    )

    merged["category_code"] = (
        merged["category"]
        .astype("category")
        .cat.codes
    )

    merged["price_scaled"] = (
        merged["price"] - merged["price"].mean()
    ) / merged["price"].std(ddof=0)

    merged["popularity_scaled"] = (
        merged["popularity_score"]
        - merged["popularity_score"].mean()
    ) / merged["popularity_score"].std(ddof=0)

    merged = merged.dropna(subset=["product_id"])

    merged.to_csv(
        PROCESSED_INTERACTIONS_PATH,
        index=False,
    )

    plt.figure(figsize=(8, 4))
    merged["rating"].hist(bins=[1, 2, 3, 4, 5])
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(
        EDA_PLOTS_DIR / "rating_distribution.png"
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    corr = merged[
        [
            "rating",
            "price_scaled",
            "popularity_scaled",
            "category_code",
        ]
    ].corr()

    plt.imshow(corr, cmap="viridis")
    plt.colorbar()
    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=45,
    )
    plt.yticks(
        range(len(corr.columns)),
        corr.columns,
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(
        EDA_PLOTS_DIR / "feature_correlation.png"
    )
    plt.close()

    logger.info("Data preparation completed.")

    return {
        "prepared_rows": len(merged),
        "path": str(PROCESSED_INTERACTIONS_PATH),
    }


if __name__ == "__main__":
    prepare_data()