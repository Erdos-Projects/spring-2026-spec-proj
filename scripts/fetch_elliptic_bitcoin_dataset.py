"""
Fetch the Elliptic Bitcoin graph dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- Nodes: 203,769 Bitcoin transactions
- Edges: 234,355 directed payment flows
- Features: 166 node features; labels: licit / illicit / unknown

Setup:
  1. pip install kaggle pandas
  2. Get API token: Kaggle Account → Create New API Token (downloads kaggle.json)
  3. Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "elliptic_bitcoin_data"

def fetch_elliptic_dataset(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    dataset_slug: str = "ellipticco/elliptic-data-set",
    unzip: bool = True,
) -> Path:
    """
    Download the Elliptic Bitcoin graph dataset from Kaggle.

    Args:
        output_dir: Directory to save the dataset (default: elliptic_bitcoin_data).
        dataset_slug: Kaggle dataset identifier (default: ellipticco/elliptic-data-set).
        unzip: If True, unzip the downloaded archive (default: True).

    Returns:
        Path to the directory containing the dataset files.

    Raises:
        ImportError: If the kaggle package is not installed.
        Exception: If Kaggle API authentication fails (missing/invalid kaggle.json or env vars).
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "Kaggle API is required. Install with: pip install kaggle"
        ) from None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset: {dataset_slug}")
    api.dataset_download_files(
        dataset_slug,
        path=str(output_path),
        unzip=unzip,
        quiet=False,
    )

    # When unzip=True, Kaggle may extract into a subfolder or directly into output_path
    if unzip:
        for subdir_name in ("elliptic-data-set", "elliptic_data_set"):
            sub = output_path / subdir_name
            if sub.exists():
                print(f"Dataset extracted to: {sub.resolve()}")
                return sub
        # Files extracted directly into output_path
        print(f"Dataset saved to: {output_path.resolve()}")
        return output_path

    return output_path


def load_elliptic_graph(data_dir: str | Path):
    """
    Load the Elliptic dataset as DataFrames (and optionally as a graph).

    Expects the following files in data_dir (from Kaggle):
      - elliptic_txs_edgelist.csv
      - elliptic_txs_features.csv
      - elliptic_txs_classes.csv

    Args:
        data_dir: Path to the directory containing the CSV files.

    Returns:
        dict with keys: 'edgelist', 'features', 'classes' (pandas DataFrames).
    """
    import pandas as pd

    data_path = Path(data_dir)

    edgelist_path = data_path / "elliptic_txs_edgelist.csv"
    features_path = data_path / "elliptic_txs_features.csv"
    classes_path = data_path / "elliptic_txs_classes.csv"

    result = {}

    if edgelist_path.exists():
        result["edgelist"] = pd.read_csv(edgelist_path)
    else:
        result["edgelist"] = None

    if features_path.exists():
        # First column is node id; rest are features (no header in original)
        result["features"] = pd.read_csv(features_path, header=None)
    else:
        result["features"] = None

    if classes_path.exists():
        result["classes"] = pd.read_csv(classes_path)
    else:
        result["classes"] = None

    return result


def main():
    import argparse

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "elliptic_bitcoin_data"

    parser = argparse.ArgumentParser(
        description="Fetch Elliptic Bitcoin graph dataset from Kaggle"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help=f"Output directory for the dataset (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Do not unzip the downloaded archive",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="After download, load and print basic info about the graph data",
    )
    args = parser.parse_args()

    data_path = fetch_elliptic_dataset(
        output_dir=args.output_dir,
        unzip=not args.no_unzip,
    )

    if args.load:
        print("\nLoading dataset into DataFrames...")
        data = load_elliptic_graph(data_path)
        for name, df in data.items():
            if df is not None:
                print(f"  {name}: shape {df.shape}")
            else:
                print(f"  {name}: file not found")
        return data

    return data_path


if __name__ == "__main__":
    main()
