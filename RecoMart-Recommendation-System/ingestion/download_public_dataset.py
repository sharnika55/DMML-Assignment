import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from project_config import RAW_DIR, ensure_project_dirs

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
TARGET_DIR = RAW_DIR / "public" / "ml-100k"


def download_movie_lens_100k() -> Path:
    ensure_project_dirs()
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = TARGET_DIR / "ml-100k.zip"

    if not archive_path.exists():
        print(f"Downloading {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, archive_path)

    if not (TARGET_DIR / "u.data").exists():
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(TARGET_DIR)

    return TARGET_DIR


if __name__ == "__main__":
    download_movie_lens_100k()
