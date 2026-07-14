import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from project_config import RAW_DIR, RAW_INTERACTIONS_PATH, RAW_PRODUCTS_PATH, ensure_project_dirs


def load_movielens_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_project_dirs()
    RAW_INTERACTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PRODUCTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset_dir = RAW_DIR / "public" / "ml-100k" / "ml-100k"
    ratings_path = dataset_dir / "u.data"
    items_path = dataset_dir / "u.item"

    ratings_df = pd.read_csv(
        ratings_path,
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        header=None,
    )
    items_df = pd.read_csv(
        items_path,
        sep='|',
        encoding='latin-1',
        names=['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western'],
        header=None,
    )

    ratings_df['product_id'] = ratings_df['item_id'].astype(str)
    ratings_df['user_id'] = ratings_df['user_id'].astype(str)
    ratings_df['event_timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
    ratings_df['purchase_flag'] = 1
    ratings_df['session_id'] = ratings_df['user_id'] + '-session'
    ratings_df = ratings_df[['user_id', 'product_id', 'rating', 'event_timestamp', 'purchase_flag', 'session_id']]

    products_df = pd.DataFrame({
        'product_id': items_df['movie_id'].astype(str),
        'name': items_df['title'],
        'category': 'Movies',
        'price': 9.99,
        'popularity_score': 3.5,
    })

    ratings_df.to_csv(RAW_INTERACTIONS_PATH, index=False)
    products_df.to_json(RAW_PRODUCTS_PATH, orient='records', indent=2)
    return ratings_df, products_df


if __name__ == "__main__":
    load_movielens_dataset()
