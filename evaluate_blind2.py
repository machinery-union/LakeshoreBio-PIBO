#!/usr/bin/env python3
"""Evaluate frozen Blind-2 predictions using only released censored labels.

Blind-2 continuous IC50 values and its 49-dimensional feature matrix are not part
of the public release. This script therefore evaluates predictions; it does not
attempt to reconstruct model inputs or refit a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
ALLOWED_CLASSES = {"<1", "1-10", ">10"}


def clean_nid(raw: object) -> str:
    return str(raw).strip().lstrip(">").split()[0].split("[")[0].strip().lower()


def pair_keys(df: pd.DataFrame) -> pd.Series:
    return df["antibody_id"].astype(str) + "|" + df["target_eprotein_id"].map(clean_nid)


def metric_summary(df: pd.DataFrame) -> dict:
    y = df["y_true"].to_numpy(int)
    p = df["proba_neutralizing"].to_numpy(float)
    pred = (p >= 0.5).astype(int)
    two_class = len(np.unique(y)) == 2
    return {
        "n": int(len(df)),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "AUC": float(roc_auc_score(y, p)) if two_class else None,
        "AUPRC": float(average_precision_score(y, p)),
        "Accuracy": float(accuracy_score(y, pred)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, pred)),
        "BalAcc": float(balanced_accuracy_score(y, pred)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "TN_FP_FN_TP": confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist(),
    }


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int,
                     rng: np.random.Generator) -> list[float]:
    values = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(roc_auc_score(y[idx], p[idx]))
    if not values:
        raise ValueError("bootstrap produced no two-class AUC sample")
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def advance_frozen_blind1_rng(data: Path, rng: np.random.Generator, n_boot: int) -> None:
    """Advance the shared RNG exactly as the frozen Blind-1/2 generator did."""
    blind1 = pd.read_csv(data / "blind1_predictions_300105.csv", encoding="utf-8-sig")
    y = blind1["y_true"].to_numpy(int)
    p = blind1["proba_neutralizing"].to_numpy(float)
    bootstrap_auc_ci(y, p, n_boot, rng)
    for _, group in blind1.groupby("target_eprotein_id", sort=True):
        gy = group["y_true"].to_numpy(int)
        if len(np.unique(gy)) < 2:
            continue
        bootstrap_auc_ci(gy, group["proba_neutralizing"].to_numpy(float), n_boot, rng)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.n_boot < 1:
        parser.error("--n-boot must be positive")

    data = args.data_dir
    labels = pd.read_csv(data / "PIBO_blind2_40.csv", encoding="utf-8-sig")
    predictions = pd.read_csv(data / "blind2_predictions_300105.csv", encoding="utf-8-sig")
    labels["_key"] = pair_keys(labels)
    predictions["_key"] = pair_keys(predictions)
    if not labels["_key"].is_unique or not predictions["_key"].is_unique:
        raise ValueError("Blind-2 antibody/target keys must be unique")
    predictions = predictions.merge(
        labels[["_key", "ic50_censored_class", "y_true"]].rename(
            columns={"y_true": "y_true_censored"}
        ),
        on="_key", how="left", validate="1:1", indicator=True,
    )
    if len(predictions) != len(labels) or not predictions["_merge"].eq("both").all():
        raise ValueError("Blind-2 censored-label and prediction keys do not match one-to-one")
    predictions = predictions.drop(columns="_merge")
    if not np.array_equal(
        predictions["y_true_censored"].to_numpy(int), predictions["y_true"].to_numpy(int)
    ):
        raise ValueError("Blind-2 censored-label and prediction labels differ")

    classes = predictions["ic50_censored_class"].astype(str)
    if not classes.isin(ALLOWED_CLASSES).all():
        raise ValueError(f"ic50_censored_class must use only {sorted(ALLOWED_CLASSES)}")
    derived_label = classes.eq("<1").astype(int).to_numpy()
    if not np.array_equal(derived_label, predictions["y_true_censored"].to_numpy(int)):
        raise ValueError("censored classes and y_true are inconsistent: only <1 is positive")

    y = predictions["y_true"].to_numpy(int)
    p = predictions["proba_neutralizing"].to_numpy(float)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("probabilities must be finite and within [0, 1]")
    if not np.array_equal(predictions["predicted_label"].to_numpy(int), (p >= 0.5).astype(int)):
        raise ValueError("predicted_label is inconsistent with the 0.5 threshold")

    per_antibody = {}
    for antibody_id, group in predictions.groupby("antibody_id", sort=True):
        gy = group["y_true"].to_numpy(int)
        gp = group["proba_neutralizing"].to_numpy(float)
        pred = (gp >= 0.5).astype(int)
        per_antibody[str(antibody_id)] = {
            "n": int(len(group)),
            "n_pos": int(gy.sum()),
            "n_correct": int((pred == gy).sum()),
            "Accuracy": float(accuracy_score(gy, pred)),
        }

    rng = np.random.default_rng(args.seed)
    advance_frozen_blind1_rng(data, rng, args.n_boot)
    overall_ci = bootstrap_auc_ci(y, p, args.n_boot, rng)

    result = {
        "overall": metric_summary(predictions),
        "overall_auc_95ci": overall_ci,
        "censored_class_counts": {
            str(k): int(v) for k, v in classes.value_counts().sort_index().items()
        },
        "per_antibody": per_antibody,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "ci_protocol": (
            "AUC bootstrap uses default_rng(seed).integers with n_boot resamples; "
            "the shared stream is advanced through frozen Blind-1 overall and "
            "two-class per-target bootstraps before the Blind-2 overall CI"
        ),
        "scope": "evaluation of frozen predictions from censored labels; no feature reconstruction",
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
