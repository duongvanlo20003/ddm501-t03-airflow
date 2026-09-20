"""Load a registered WDBC model from MLflow and score the saved test split."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

STAGING = Path(__file__).resolve().parents[1] / "data" / "staging"
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "wdbc-classifier")


def newest_date() -> str:
    folders = sorted(p.name for p in STAGING.glob("????-??-??")
                     if (p / "test.parquet").is_file())
    if not folders:
        raise SystemExit("No saved test split found. Run the Airflow DAG first.")
    return folders[-1]


def newest_version(client: mlflow.MlflowClient) -> int:
    versions = list(client.search_model_versions(f"name='{MODEL_NAME}'"))
    if not versions:
        raise SystemExit(f"No version of {MODEL_NAME!r} is registered yet.")
    return max(int(version.version) for version in versions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds", help="logical date, YYYY-MM-DD (default: newest saved split)")
    parser.add_argument("--version", type=int, help="registry version (default: newest)")
    parser.add_argument("--rows", type=int, default=5, help="number of rows to score")
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be at least 1")
    if args.version is not None and args.version < 1:
        parser.error("--version must be at least 1")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:15030"))
    version = args.version or newest_version(mlflow.MlflowClient())
    model_uri = f"models:/{MODEL_NAME}/{version}"
    model = mlflow.sklearn.load_model(model_uri)

    ds = args.ds or newest_date()
    test_file = STAGING / ds / "test.parquet"
    if not test_file.is_file():
        raise SystemExit(f"Saved test split does not exist: {test_file}")
    test = pd.read_parquet(test_file).head(args.rows)
    features = test.drop(columns=["sample_id", "diagnosis"])
    probabilities = model.predict_proba(features)[:, 1]

    print(f"Model: {model_uri}; test date: {ds}")
    for sample_id, actual, probability in zip(
            test["sample_id"], test["diagnosis"], probabilities):
        predicted = "M" if probability >= 0.5 else "B"
        print(f"{sample_id}: actual={actual}, predicted={predicted}, "
              f"p(M)={probability:.4f}")


if __name__ == "__main__":
    main()
