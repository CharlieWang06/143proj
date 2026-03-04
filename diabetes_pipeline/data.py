import os
from pathlib import Path

import kagglehub
import pandas as pd


DEFAULT_DATASET = "alexteboul/diabetes-health-indicators-dataset"
DEFAULT_CSV = "diabetes_binary_health_indicators_BRFSS2015.csv"


def download_dataset(data_dir: Path, dataset: str = DEFAULT_DATASET) -> Path:
    """Download dataset with kagglehub into project data directory."""
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(data_dir)
    return Path(kagglehub.dataset_download(dataset))


def find_existing_csv(project_root: Path, csv_name: str = DEFAULT_CSV) -> Path | None:
    """Find an already-downloaded CSV in project data folder."""
    matches = list((project_root / "data").rglob(csv_name))
    if matches:
        return matches[0]
    return None


def resolve_csv_path(
    project_root: Path,
    data_dir_name: str = "data",
    csv_name: str = DEFAULT_CSV,
    download_if_missing: bool = True,
) -> Path:
    """Resolve CSV path from local data; optionally download if missing."""
    existing = find_existing_csv(project_root, csv_name=csv_name)
    if existing is not None:
        return existing

    if not download_if_missing:
        raise FileNotFoundError(
            f"Could not find {csv_name} under {(project_root / data_dir_name)!s}."
        )

    downloaded_path = download_dataset(project_root / data_dir_name)
    csv_path = downloaded_path / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Downloaded dataset but missing expected file: {csv_path}")
    return csv_path


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load diabetes dataset CSV."""
    return pd.read_csv(csv_path)

