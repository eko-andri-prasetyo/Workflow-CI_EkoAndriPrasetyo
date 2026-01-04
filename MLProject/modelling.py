"""
Kriteria 2 - modelling.py (Basic - MLflow autolog)

Dataset: creditscoring_preprocessing/creditscoring_preprocessed.csv
Label: target
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.ensemble import RandomForestClassifier


# Kolom sesuai dataset creditscoring_preprocessed.csv versi kamu
NUM_COLS = [
    "age",
    "monthly_income",
    "loan_amount",
    "tenure_months",
    "num_credit_lines",
    "has_previous_default",
]
CAT_COLS = ["job_type", "education_level", "city", "marital_status"]


def build_pipeline(n_estimators: int, random_state: int) -> Pipeline:
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[("num", numeric, NUM_COLS), ("cat", categorical, CAT_COLS)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )

    return Pipeline(steps=[("preprocess", pre), ("model", model)])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-path",
        type=str,
        default="creditscoring_preprocessing/creditscoring_preprocessed.csv",
    )
    p.add_argument("--target-col", type=str, default="target")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--run-name", type=str, default="rf-pipeline-autolog")
    return p.parse_args()


def ensure_columns_exist(df: pd.DataFrame, target_col: str) -> None:
    missing = []
    for c in NUM_COLS + CAT_COLS + [target_col]:
        if c not in df.columns:
            missing.append(c)
    if missing:
        raise ValueError(
            "Kolom berikut tidak ditemukan di dataset: "
            + ", ".join(missing)
            + "\nCek kembali creditscoring_preprocessed.csv kamu."
        )


def main() -> None:
    args = parse_args()

    # Tracking URI: kalau secrets kosong, fallback local mlruns
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        tracking_uri = "file:./mlruns"

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "creditscoring-MSML")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {data_path} (pwd={Path.cwd()})"
        )

    df = pd.read_csv(data_path)
    ensure_columns_exist(df, args.target_col)

    X = df.drop(columns=[args.target_col])
    y = df[args.target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipe = build_pipeline(n_estimators=args.n_estimators, random_state=args.random_state)

    # Autolog (Basic)
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run(run_name=args.run_name):
        # log params tambahan biar jelas
        mlflow.log_param("data_path", str(data_path))
        mlflow.log_param("target_col", args.target_col)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("n_estimators", args.n_estimators)

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        # metrics eksplisit (biar aman walau autolog gak cover nama ini)
        mlflow.log_metric("test_accuracy", float(accuracy_score(y_test, preds)))
        mlflow.log_metric(
            "test_precision", float(precision_score(y_test, preds, zero_division=0))
        )
        mlflow.log_metric(
            "test_recall", float(recall_score(y_test, preds, zero_division=0))
        )
        mlflow.log_metric("test_f1", float(f1_score(y_test, preds, zero_division=0)))

        # artifact: confusion matrix
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax)
        ax.set_title("Confusion Matrix")
        out_png = "training_confusion_matrix.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(out_png)

        print("Done.")
        print(f"Tracking URI: {tracking_uri}")
        print(f"Experiment: {experiment_name}")


if __name__ == "__main__":
    main()
