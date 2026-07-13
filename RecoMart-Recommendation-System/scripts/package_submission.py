import shutil
from pathlib import Path
import zipfile
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT_SCRIPT = ROOT / "reports" / "generate_submission_report.py"
OUTPUT_ZIP = ROOT / "RecoMart_Submission.zip"


def build_pdf() -> None:
    # run the report builder
    try:
        sys.path.insert(0, str(ROOT))
        from reports.generate_submission_report import build_report

        build_report()
    except Exception as e:
        print(f"Warning: failed to build PDF report: {e}")


def collect_files(zip_path: Path) -> None:
    include = [
        "README.md",
        "reports/RecoMart_Assignment_Report.pdf",
        "reports/validation_report.json",
        "reports/demo_walkthrough.md",
        "data/processed/prepared_interactions.csv",
        "data/features/feature_table.csv",
        "data/models/recommendation_model.pkl",
        "data/models/model_metadata.json",
        "dags/recomart_dag.py",
        "docker-compose.yml",
        "requirements.txt",
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for rel in include:
            p = ROOT / rel
            if p.exists():
                z.write(p, arcname=rel)
            else:
                print(f"Skipping missing file: {rel}")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    print("Building PDF report...")
    build_pdf()
    print(f"Creating submission ZIP at {OUTPUT_ZIP}")
    collect_files(OUTPUT_ZIP)
    print("Done. Verify RecoMart_Submission.zip in repository root.")


if __name__ == "__main__":
    main()
