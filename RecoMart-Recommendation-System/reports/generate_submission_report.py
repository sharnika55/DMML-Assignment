import json
import os
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Image

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "RecoMart_Assignment_Report.pdf"
PLOTS_DIR = ROOT / "reports" / "plots"
MODEL_METADATA_PATH = ROOT / "data" / "models" / "model_metadata.json"
VALIDATION_REPORT_PATH = ROOT / "reports" / "validation_report.json"
VIDEO_LINK = os.getenv("RECOMART_VIDEO_LINK", "Not provided")
ZIP_LINK = os.getenv("RECOMART_ZIP_LINK", "Not provided")


def build_report() -> None:
    doc = SimpleDocTemplate(str(REPORT_PATH), pagesize=letter, rightMargin=0.75 * inch, leftMargin=0.75 * inch, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("End-to-End Data Management Pipeline for a Recommendation System", styles["Title"]))
    story.append(Paragraph("RecoMart Recommendation System", styles["Heading1"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Team Member Details: Student Group / RecoMart Data Platform Team", styles["BodyText"]))
    story.append(Paragraph(f"Submission Date: {date.today().strftime('%d.%m.%Y')}", styles["BodyText"]))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("1. Problem Statement", styles["Heading2"]))
    story.append(Paragraph("RecoMart wants to create a scalable recommendation engine that uses clickstream, purchase, and product metadata to provide personalized product suggestions and improve engagement and conversion rates.", styles["BodyText"]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("2. Objectives", styles["Heading2"]))
    story.append(Paragraph("- Ingest interactions and product data from multiple sources", styles["BodyText"]))
    story.append(Paragraph("- Validate data quality and monitor issues", styles["BodyText"]))
    story.append(Paragraph("- Prepare and transform data for downstream modeling", styles["BodyText"]))
    story.append(Paragraph("- Build a feature store and train a recommendation model", styles["BodyText"]))
    story.append(Paragraph("- Orchestrate the pipeline with Airflow and track experiments with MLflow", styles["BodyText"]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("3. Methodology / Pipeline", styles["Heading2"]))
    story.append(Paragraph("The implemented workflow follows ingestion -> validation -> preparation -> feature engineering -> feature store -> model training.", styles["BodyText"]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("4. Implementation Details", styles["Heading2"]))
    story.append(Paragraph("- Ingestion scripts download and process the MovieLens 100k public dataset and write timestamp-partitioned raw data plus ingestion manifest lineage.", styles["BodyText"]))
    story.append(Paragraph("- Validation checks verify missing values, duplicate rows, and rating ranges.", styles["BodyText"]))
    story.append(Paragraph("- Preparation joins product metadata with interaction data, encodes categories, and normalizes attributes.", styles["BodyText"]))
    story.append(Paragraph("- Feature engineering creates user activity frequency, average ratings, and popularity-based features.", styles["BodyText"]))
    story.append(Paragraph("- Airflow orchestrates the workflow while MLflow records parameters, metrics, and artifacts.", styles["BodyText"]))
    story.append(Spacer(1, 0.1 * inch))

    try:
        model_meta = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        model_meta = None
    try:
        validation_report = json.loads(VALIDATION_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        validation_report = None

    story.append(Paragraph("5. Results and Outputs", styles["Heading2"]))
    if model_meta:
        metrics_data = [["Metric", "Value"], ["Model", model_meta.get("model")], ["NMF Components", model_meta.get("parameters", {}).get("n_components")], ["RMSE", round(model_meta.get("metrics", {}).get("rmse", 0.0), 4)], ["Precision@5", round(model_meta.get("metrics", {}).get("precision_at_5", 0.0), 4)], ["Recall@5", round(model_meta.get("metrics", {}).get("recall_at_5", 0.0), 4)], ["NDCG@5", round(model_meta.get("metrics", {}).get("ndcg_at_5", 0.0), 4)]]
    else:
        metrics_data = [["Metric", "Value"], ["Model", "N/A"], ["NMF Components", "N/A"], ["RMSE", "N/A"], ["Precision@5", "N/A"], ["Recall@5", "N/A"], ["NDCG@5", "N/A"]]
    table = Table(metrics_data, colWidths=[2.2 * inch, 3.2 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F81BD")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.15 * inch))

    if validation_report:
        story.append(Paragraph(f"Validation summary: {validation_report['summary']['datasets_checked']} datasets checked, passed = {validation_report['summary']['passed']}", styles["BodyText"]))
    else:
        story.append(Paragraph("Validation summary: Not available", styles["BodyText"]))
    story.append(Spacer(1, 0.15 * inch))

    # Add MLflow and sample recommendations if available
    if model_meta and model_meta.get("mlflow_run_id"):
        story.append(Paragraph("Experiment Tracking", styles["Heading2"]))
        story.append(Paragraph(f"MLflow run id: {model_meta.get('mlflow_run_id')}", styles["BodyText"]))
        story.append(Paragraph(f"Local mlruns path: {ROOT / 'mlruns'}", styles["BodyText"]))
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph("Sample Recommendations (top 5 per user):", styles["Heading3"]))
        # show up to 5 users
        recs = model_meta.get("recommendations", [])
        for item in recs[:5]:
            if isinstance(item, dict):
                uid = item.get("user_id")
                rlist = item.get("recommendations", [])
                story.append(Paragraph(f"- {uid}: {', '.join(rlist[:5])}", styles["BodyText"]))
        story.append(Spacer(1, 0.15 * inch))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Sample plots generated:", styles["BodyText"]))
    story.append(Spacer(1, 0.05 * inch))
    img1 = Image(str(PLOTS_DIR / "rating_distribution.png"), width=4.5 * inch, height=2.5 * inch)
    img2 = Image(str(PLOTS_DIR / "feature_correlation.png"), width=4.5 * inch, height=2.5 * inch)
    story.append(img1)
    story.append(Spacer(1, 0.1 * inch))
    story.append(img2)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("6. Conclusion and Future Scope", styles["Heading2"]))
    story.append(Paragraph("The pipeline provides a modular and reproducible foundation for recommendation-system data management. Future work can extend it to near-real-time ingestion, richer feature engineering, and deployment via an inference API.", styles["BodyText"]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("7. Submission Links", styles["Heading2"]))
    story.append(Paragraph(f"Google Drive Link to Video Walkthrough: {VIDEO_LINK}", styles["BodyText"]))
    story.append(Paragraph(f"Google Drive Link to Deliverables ZIP: {ZIP_LINK}", styles["BodyText"]))
    story.append(Paragraph("Repository: https://github.com/sharnika55/DMML-Assignment", styles["BodyText"]))

    try:
        doc.build(story)
    except PermissionError:
        alt_path = REPORT_PATH.with_name(REPORT_PATH.stem + "_v2" + REPORT_PATH.suffix)
        alt_doc = SimpleDocTemplate(str(alt_path), pagesize=letter, rightMargin=0.75 * inch, leftMargin=0.75 * inch, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
        alt_doc.build(story)


if __name__ == "__main__":
    build_report()
