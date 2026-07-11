import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from project_config import (
    RAW_DIR,
    REPORTS_DIR,
    ROOT_DIR,
    VALIDATION_REPORT_JSON,
    VALIDATION_REPORT_PDF,
    ensure_project_dirs,
)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_latest_raw_files() -> Tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = RAW_DIR / "ingestion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    interactions_path = ROOT_DIR / manifest["files"][0]["path"]
    products_path = ROOT_DIR / manifest["files"][1]["path"]
    interactions = pd.read_csv(interactions_path)
    products = pd.read_json(products_path)
    return interactions, products


def validate_interactions(df: pd.DataFrame) -> Dict[str, Any]:
    issues: List[str] = []
    metrics = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if df["rating"].between(1, 5).all():
        metrics["rating_range_ok"] = True
    else:
        issues.append("Ratings outside allowed range 1-5")
        metrics["rating_range_ok"] = False
    if df["user_id"].isna().any():
        issues.append("Missing user_id values")
    if df["product_id"].isna().any():
        issues.append("Missing product_id values")
    if df["event_timestamp"].isna().any():
        issues.append("Missing event_timestamp values")
    return {"dataset": "interactions", "metrics": metrics, "issues": issues}


def validate_products(df: pd.DataFrame) -> Dict[str, Any]:
    issues: List[str] = []
    metrics = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if df["product_id"].isna().any():
        issues.append("Missing product_id values")
    if df["category"].isna().any():
        issues.append("Missing category values")
    return {"dataset": "products", "metrics": metrics, "issues": issues}


def generate_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    report = {"summary": {"datasets_checked": len(results), "passed": all(not item["issues"] for item in results)}}
    report["datasets"] = results
    return report


def write_pdf(report: Dict[str, Any]) -> None:
    document = SimpleDocTemplate(str(VALIDATION_REPORT_PDF), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("RecoMart Data Quality Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Summary: {report['summary']}", styles["BodyText"]))
    story.append(Spacer(1, 12))
    for dataset in report["datasets"]:
        story.append(Paragraph(dataset["dataset"].capitalize(), styles["Heading2"]))
        rows = [["Metric", "Value"], ["Rows", dataset["metrics"]["rows"]], ["Missing Values", dataset["metrics"]["missing_values"]], ["Duplicate Rows", dataset["metrics"]["duplicate_rows"]]]
        table = Table(rows, hAlign="LEFT")
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F81BD")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
        story.append(table)
        story.append(Spacer(1, 12))
        if dataset["issues"]:
            story.append(Paragraph("Issues: " + "; ".join(dataset["issues"]), styles["BodyText"]))
        else:
            story.append(Paragraph("Issues: None", styles["BodyText"]))
        story.append(Spacer(1, 12))
    document.build(story)


def run_validation() -> Dict[str, Any]:
    ensure_project_dirs()
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting validation")
    interactions, products = load_latest_raw_files()
    results = [validate_interactions(interactions), validate_products(products)]
    report = generate_report(results)
    VALIDATION_REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_pdf(report)
    logger.info("Validation completed")
    return report


if __name__ == "__main__":
    run_validation()
