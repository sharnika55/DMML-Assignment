import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from project_config import (
    RAW_DIR,
    RAW_INTERACTIONS_PATH,
    RAW_PRODUCTS_PATH,
    ROOT_DIR,
    ensure_project_dirs,
)

LOG_PATH = ROOT_DIR / "logs" / "ingestion.log"
MANIFEST_PATH = RAW_DIR / "ingestion_manifest.json"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
    )


def write_partitioned_file(data: Any, source: str, file_name: str) -> Path:
    date_prefix = datetime.utcnow().strftime("%Y/%m/%d")
    destination_dir = RAW_DIR / source / date_prefix
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / file_name
    if isinstance(data, pd.DataFrame):
        data.to_csv(destination_path, index=False)
    else:
        destination_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return destination_path


def load_interactions(source_path: Path) -> pd.DataFrame:
    return pd.read_csv(source_path)


def load_products(source_path: Path) -> List[Dict[str, Any]]:
    with source_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_manifest(paths: List[Tuple[str, Path]]) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {"ingested_at": datetime.utcnow().isoformat(), "files": []}
    for source, path in paths:
        manifest["files"].append({"source": source, "path": str(path.relative_to(ROOT_DIR))})
    return manifest


def run_ingestion() -> Dict[str, Any]:
    ensure_project_dirs()
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting ingestion")

    interactions = load_interactions(RAW_INTERACTIONS_PATH)
    products = load_products(RAW_PRODUCTS_PATH)

    interaction_path = write_partitioned_file(interactions, "ratings", "interactions.csv")
    product_path = write_partitioned_file(products, "api", "products.json")

    manifest = build_manifest([("ratings", interaction_path), ("api", product_path)])
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Ingestion completed successfully")
    return {"interactions": interaction_path, "products": product_path, "manifest": MANIFEST_PATH}


if __name__ == "__main__":
    run_ingestion()
