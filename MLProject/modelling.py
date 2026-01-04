"""
Creditscoring - modelling.py

Dataset default: creditscoring_preprocessing/creditscoring_preprocessed.csv
Label default  : target

Advance:
- Enable MLflow autolog (optional) + manual logging additional metrics
- Log at least 2 additional artifacts (confusion matrix, roc curve, classification report, etc.)
- Compatible with `mlflow run . -P data_path=... -P target_col=...`
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

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
    classification_report,
    roc_auc_score,
    average_precision_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from sklearn.ensemble import RandomForestClassifier


# Kolom sesuai dataset Syaikh Eko (pastikan match)
NUM_COLS = [
    "age",
    "monthly_income",
    "loan_amount",
    "tenure_months",
    "num_credit_lines",
    "has_previous_default",
]
CAT_COLS = ["job_type", "education_level", "city", "marital_status"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-path",
        default="creditscoring_preprocessing/creditscoring_preprocessed.csv",
        help="Path CSV preprocessed",
    )
    p.add_argument("--target-col", default="target", help="Nama kolom target/label")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)

    # model params
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--min-samples-split", type=int, default=2)
    p.add_argument("--min-samples-leaf", type=int, default=1)

    # mlflow
    p.add_argument("--experiment-name", default="creditscoring-MSML")
    p.add_argument("--run-name", default="rf-pipeline-advance")
    p.add_argument("--use-autolog", action="store_true", help="Enable mlflow sklearn autolog")
    return p.parse_args()


def build_pipeline(args: argparse.Namespace) -> Pipeline:
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
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
    )

    return Pipeline(steps=[("preprocess", pre), ("model", model)])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    # Tracking URI: pakai ENV kalau ada; fallback lokal
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"CSV not found: {data_path.resolve()}")

    df = pd.read_csv(data_path)

    if args.target_col not in df.columns:
        raise ValueError(
            f"Target col '{args.target_col}' tidak ada. Kolom tersedia: {list(df.columns)}"
        )

    X = df.drop(columns=[args.target_col])
    y = df[args.target_col].astype(int)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if y.nunique() <= 20 else None,
    )

    pipe = build_pipeline(args)

    # Autolog optional (sesuai kebutuhan rubrik)
    if args.use_autolog:
        mlflow.sklearn.autolog(log_models=True)

    artifacts_dir = Path("artifacts")
    ensure_dir(artifacts_dir)

    with mlflow.start_run(run_name=args.run_name) as run:
        # ===== Manual Params (jelas untuk reviewer) =====
        mlflow.log_param("data_path", str(data_path))
        mlflow.log_param("target_col", args.target_col)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth if args.max_depth is not None else "None")
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("min_samples_leaf", args.min_samples_leaf)

        mlflow.log_param("n_rows", int(df.shape[0]))
        mlflow.log_param("n_features", int(X.shape[1]))

        # ===== Train =====
        pipe.fit(X_train, y_train)

        # ===== Predict =====
        preds = pipe.predict(X_test)

        # ===== Metrics utama =====
        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1", f1)

        # ===== Metrics tambahan (biasanya tidak “cukup” di autolog) =====
        # ROC AUC & PR AUC (Average Precision) untuk binary
        roc_auc = None
        pr_auc = None
        if y.nunique() == 2:
            try:
                proba = pipe.predict_proba(X_test)[:, 1]
                roc_auc = float(roc_auc_score(y_test, proba))
                pr_auc = float(average_precision_score(y_test, proba))
                mlflow.log_metric("test_roc_auc", roc_auc)
                mlflow.log_metric("test_pr_auc", pr_auc)
            except Exception:
                pass

        # ===== Artifacts (minimal 2 tambahan) =====

        # (1) Confusion matrix PNG
        fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax_cm)
        ax_cm.set_title("Confusion Matrix")
        cm_path = artifacts_dir / "training_confusion_matrix.png"
        fig_cm.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig_cm)
        mlflow.log_artifact(str(cm_path))

        # (2) Classification report TXT
        report_text = classification_report(y_test, preds, zero_division=0)
        report_path = artifacts_dir / "classification_report.txt"
        report_path.write_text(report_text, encoding="utf-8")
        mlflow.log_artifact(str(report_path))

        # (3) ROC Curve PNG (kalau binary)
        if roc_auc is not None:
            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
            RocCurveDisplay.from_predictions(y_test, proba, ax=ax_roc)
            ax_roc.set_title("ROC Curve")
            roc_path = artifacts_dir / "roc_curve.png"
            fig_roc.savefig(roc_path, dpi=150, bbox_inches="tight")
            plt.close(fig_roc)
            mlflow.log_artifact(str(roc_path))

        # (4) Precision-Recall Curve PNG (kalau binary)
        if pr_auc is not None:
            fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
            PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax_pr)
            ax_pr.set_title("Precision-Recall Curve")
            pr_path = artifacts_dir / "pr_curve.png"
            fig_pr.savefig(pr_path, dpi=150, bbox_inches="tight")
            plt.close(fig_pr)
            mlflow.log_artifact(str(pr_path))

        # (5) Metrics JSON (bonus)
        metrics_payload = {
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1,
            "test_roc_auc": roc_auc,
            "test_pr_auc": pr_auc,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
        }
        metrics_path = artifacts_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(metrics_path))

        # Log model explicitly juga (biar aman walau autolog off)
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        print(f"Run ID: {run.info.run_id}")
        print(f"Tracking URI: {tracking_uri}")
        print("Done. Metrics:", metrics_payload)


if __name__ == "__main__":
    main()
