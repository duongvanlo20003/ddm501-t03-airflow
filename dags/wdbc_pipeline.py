"""
DDM501 Tutorial 03 — a data pipeline that runs

Seven tasks: ingest -> validate -> split -> scale -> train -> register -> report.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

log = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw" / "wdbc.csv"
STAGING = PROJECT / "data" / "staging"

FEATURES_MIN = 0.0                 # every WDBC measurement is a non-negative size
LABELS = {"M", "B"}
MAX_BAD_FRACTION = 0.05            # above this the extract is not worth using
TEST_FRACTION = 0.20
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:15030")
MLFLOW_EXPERIMENT = "wdbc-pipeline"
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "wdbc-classifier")


def run_dir(ds: str) -> Path:
    """One folder per logical date. Re-running a date overwrites its own folder
    and touches nothing else, which is what makes a re-run safe."""
    d = STAGING / ds
    d.mkdir(parents=True, exist_ok=True)
    return d


@dag(
    dag_id="wdbc_pipeline",
    description="Breast cancer extract: ingest, validate, split, scale, train, register",
    schedule="@daily",
    start_date=datetime(2026, 8, 20),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 3,
        "retry_delay": timedelta(seconds=10),
        "retry_exponential_backoff": True,
    },
    tags=["ddm501", "tutorial-03"],
)
def wdbc_pipeline():

    @task
    def ingest(ds: str = None) -> dict:
        """Copy the extract into this run's folder and freeze it there.

        Reading the source again in a later task would mean two tasks seeing
        two different files if the source changes mid-run. Snapshot once.
        """
        if not RAW.exists():
            raise AirflowFailException(f"source extract missing: {RAW}")
        frame = pd.read_csv(RAW)
        out = run_dir(ds) / "raw.parquet"
        frame.to_parquet(out, index=False)
        log.info("ingested %d rows, %d columns", len(frame), frame.shape[1])
        # Returned dicts travel as XCom, which is stored in the metadata
        # database. Keep them to counts and paths -- never a DataFrame.
        return {"rows": len(frame), "cols": frame.shape[1], "path": str(out)}

    @task
    def validate(meta: dict, ds: str = None) -> dict:
        """Quarantine bad rows; fail only if too many of them."""
        frame = pd.read_parquet(meta["path"])
        numeric = [c for c in frame.columns if c not in ("sample_id", "diagnosis")]

        problems = pd.DataFrame(index=frame.index)
        problems["null"] = frame[numeric].isna().any(axis=1)
        problems["negative"] = (frame[numeric] < FEATURES_MIN).any(axis=1)
        problems["bad_label"] = ~frame["diagnosis"].isin(LABELS)
        problems["duplicate"] = frame.duplicated(subset="sample_id", keep="first")
        # An area 20x the 99th percentile is a data-entry error, not a tumour.
        cutoff = frame["mean_area"].quantile(0.99) * 20
        problems["outlier"] = frame["mean_area"] > cutoff

        bad = problems.any(axis=1)
        counts = {k: int(v) for k, v in problems.sum().items()}
        fraction = float(bad.mean())
        log.info("validation: %s  (%.2f%% of rows rejected)", counts, fraction * 100)

        clean = frame[~bad]
        rejected = frame[bad]
        rejected.to_parquet(run_dir(ds) / "rejected.parquet", index=False)
        clean_path = run_dir(ds) / "clean.parquet"
        clean.to_parquet(clean_path, index=False)
        (run_dir(ds) / "validation_report.json").write_text(
            json.dumps({"counts": counts, "bad_fraction": fraction,
                        "clean_rows": len(clean)}, indent=2))

        if fraction > MAX_BAD_FRACTION:
            # AirflowFailException stops the run without burning the retries:
            # a malformed file will still be malformed on the third attempt.
            raise AirflowFailException(
                f"{fraction:.1%} of rows rejected, limit is {MAX_BAD_FRACTION:.0%}")
        return {"path": str(clean_path), "clean_rows": len(clean), **counts}

    @task
    def split(meta: dict, ds: str = None) -> dict:
        """Deterministic split by hashing the id -- no random seed involved.

        A seeded shuffle gives the same split only if the rows arrive in the
        same order. Hashing the id gives the same split for a given row
        forever, on any machine, even if tomorrow's extract adds rows.
        """
        frame = pd.read_parquet(meta["path"])

        def bucket(sample_id: str) -> int:
            digest = hashlib.sha256(sample_id.encode()).hexdigest()
            return int(digest[:8], 16) % 100

        is_test = frame["sample_id"].map(bucket) < TEST_FRACTION * 100
        for name, part in (("train", frame[~is_test]), ("test", frame[is_test])):
            part.to_parquet(run_dir(ds) / f"{name}_unscaled.parquet", index=False)
        log.info("split: %d train / %d test", (~is_test).sum(), is_test.sum())
        return {"train": int((~is_test).sum()), "test": int(is_test.sum())}

    @task
    def scale(meta: dict, ds: str = None) -> dict:
        """Fit the scaler on train only, then apply it to both."""
        train = pd.read_parquet(run_dir(ds) / "train_unscaled.parquet")
        test = pd.read_parquet(run_dir(ds) / "test_unscaled.parquet")
        numeric = [c for c in train.columns if c not in ("sample_id", "diagnosis")]

        mean, std = train[numeric].mean(), train[numeric].std().replace(0, 1)
        for name, part in (("train", train), ("test", test)):
            scaled = part.copy()
            scaled[numeric] = (part[numeric] - mean) / std
            scaled.to_parquet(run_dir(ds) / f"{name}.parquet", index=False)

        (run_dir(ds) / "scaler.json").write_text(json.dumps(
            {"mean": mean.round(6).to_dict(), "std": std.round(6).to_dict()}, indent=2))
        log.info("scaled with statistics from %d training rows", len(train))
        return {"scaled_columns": len(numeric), "fitted_on": len(train)}

    @task
    def train(scaling: dict, ds: str = None) -> dict:
        """Fit a classifier and log its model and test metrics to MLflow."""
        folder = run_dir(ds)
        training = pd.read_parquet(folder / "train.parquet")
        testing = pd.read_parquet(folder / "test.parquet")
        features = [c for c in training if c not in ("sample_id", "diagnosis")]
        x_train, x_test = training[features], testing[features]
        y_train = (training["diagnosis"] == "M").astype(int)
        y_test = (testing["diagnosis"] == "M").astype(int)

        if y_train.nunique() != 2 or y_test.nunique() != 2:
            raise AirflowFailException("both train and test must contain M and B labels")

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(x_train, y_train)
        accuracy = float(accuracy_score(y_test, model.predict(x_test)))
        roc_auc = float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=f"wdbc-{ds}") as run:
            mlflow.log_params({"logical_date": ds, "features": len(features),
                               "train_rows": len(training), "test_rows": len(testing)})
            mlflow.log_metrics({"accuracy": accuracy, "roc_auc": roc_auc})
            mlflow.sklearn.log_model(model, artifact_path="model", input_example=x_train.head(1))
            run_id = run.info.run_id

        log.info("MLflow run %s: accuracy=%.4f, roc_auc=%.4f", run_id, accuracy, roc_auc)
        return {"mlflow_run_id": run_id, "accuracy": round(accuracy, 4),
                "roc_auc": round(roc_auc, 4)}

    @task
    def register(training: dict) -> dict:
        """Publish the logged model as a new registry version for this run."""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.MlflowClient()
        # A task retry should not create another version for the same MLflow run.
        existing = [version for version in client.search_model_versions(
            f"name='{MLFLOW_MODEL_NAME}'") if version.run_id == training["mlflow_run_id"]]
        if existing:
            version_number = max(int(version.version) for version in existing)
        else:
            version = mlflow.register_model(
                f"runs:/{training['mlflow_run_id']}/model", MLFLOW_MODEL_NAME)
            version_number = int(version.version)
        log.info("registered %s version %s", MLFLOW_MODEL_NAME, version_number)
        return {**training, "model_version": version_number}

    @task
    def report(validation: dict, split_info: dict, scaling: dict,
               registered: dict, ds: str = None) -> str:
        """One line per run, appended to a log the whole pipeline shares."""
        summary = {"ds": ds, **validation, **split_info, **scaling, **registered}
        summary.pop("path", None)
        (run_dir(ds) / "summary.json").write_text(json.dumps(summary, indent=2))

        line = json.dumps(summary, sort_keys=True)
        history = STAGING / "history.jsonl"
        kept = [l for l in (history.read_text().splitlines() if history.exists() else [])
                if json.loads(l).get("ds") != ds]
        history.write_text("\n".join(kept + [line]) + "\n")
        log.info("summary: %s", line)
        return line

    ingested = ingest()
    validated = validate(ingested)
    split_info = split(validated)
    scaling = scale(validated)
    split_info >> scaling
    trained = train(scaling)
    registered = register(trained)
    report(validated, split_info, scaling, registered)


wdbc_pipeline()
