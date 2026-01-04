"""Kriteria 2 - modelling.py (Basic - MLflow autolog)

Dataset: `creditscoring_preprocessing/creditscoring_preprocessed.csv`
Label: `target`
"""

from __future__ import annotations

import os
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier



mlflow.set_tracking_uri("https://dagshub.com/eko-andri-prasetyo/Membangun_model.mlflow")
mlflow.set_experiment("creditscoring-MSML")


DATA_PATH = Path("creditscoring_preprocessing/creditscoring_preprocessed.csv")
TARGET_COL = "target"

NUM_COLS = ['age', 'monthly_income', 'loan_amount', 'tenure_months', 'num_credit_lines', 'has_previous_default']
CAT_COLS = ['job_type', 'education_level', 'city', 'marital_status']

def build_pipeline() -> Pipeline:
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(
        transformers=[("num", numeric, NUM_COLS), ("cat", categorical, CAT_COLS)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    return Pipeline(steps=[("preprocess", pre), ("model", model)])

def main() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("creditscoring-MSML")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run(run_name="rf-pipeline-autolog"):
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mlflow.log_metric("test_accuracy", float(accuracy_score(y_test, preds)))
        mlflow.log_metric("test_precision", float(precision_score(y_test, preds, zero_division=0)))
        mlflow.log_metric("test_recall", float(recall_score(y_test, preds, zero_division=0)))
        mlflow.log_metric("test_f1", float(f1_score(y_test, preds, zero_division=0)))

        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax)
        ax.set_title("Confusion Matrix")
        fig.savefig("training_confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact("training_confusion_matrix.png")

        print("Done. Open MLflow UI at: http://127.0.0.1:5000")

if __name__ == "__main__":
    main()
