#!/usr/bin/env python3
"""Reproduce Blind-1 metrics and verify frozen predictions with the released RF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
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


def clean_nid(raw: object) -> str:
    return str(raw).strip().lstrip(">").split()[0].split("[")[0].strip().lower()


def pair_keys(df: pd.DataFrame) -> pd.Series:
    return df["antibody_id"].astype(str) + "|" + df["target_eprotein_id"].map(clean_nid)


def as_bool(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(bool)
    mapped = series.astype(str).str.strip().str.lower().map({
        "true": True, "1": True, "yes": True,
        "false": False, "0": False, "no": False,
    })
    if mapped.isna().any():
        raise ValueError("sequence_seen_in_training contains unrecognized values")
    return mapped.to_numpy(bool)


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


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int, seed: int) -> list[float]:
    """Match the frozen generator: fresh default_rng(seed), integers, 2000 draws."""
    rng = np.random.default_rng(seed)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probability-atol", type=float, default=5.1e-7)
    args = parser.parse_args()
    if args.n_boot < 1:
        parser.error("--n-boot must be positive")

    data = args.data_dir
    cfg = json.loads((data / "final_model_config_49d.json").read_text(encoding="utf-8"))
    feature_names = list(cfg["feature_names"])
    features = pd.read_csv(data / "features_blind1_49d.csv", encoding="utf-8-sig")
    predictions = pd.read_csv(data / "blind1_predictions_300105.csv", encoding="utf-8-sig")

    if len(features) != len(predictions) or not pair_keys(features).equals(pair_keys(predictions)):
        raise ValueError("Blind-1 feature and prediction rows are not aligned")
    if not np.array_equal(features["y_true"].to_numpy(int), predictions["y_true"].to_numpy(int)):
        raise ValueError("Blind-1 feature and prediction labels differ")
    y = predictions["y_true"].to_numpy(int)
    p = predictions["proba_neutralizing"].to_numpy(float)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("probabilities must be finite and within [0, 1]")
    if not np.array_equal(predictions["predicted_label"].to_numpy(int), (p >= 0.5).astype(int)):
        raise ValueError("predicted_label is inconsistent with the 0.5 threshold")

    X = features.reindex(columns=feature_names).apply(pd.to_numeric, errors="raise")
    model = joblib.load(data / "final_model_rf49.joblib")
    recalculated = model.predict_proba(X)[:, 1]
    max_delta = float(np.max(np.abs(recalculated - p)))
    if max_delta > args.probability_atol:
        raise RuntimeError(
            f"released RF predictions differ from frozen Blind-1 predictions: max abs delta={max_delta:.9g}"
        )

    seen = as_bool(predictions["sequence_seen_in_training"])
    predictions = predictions.copy()
    predictions["sequence_seen_in_training"] = seen
    novel = predictions.loc[~seen]
    previously_seen = predictions.loc[seen]
    overlap_counts = {
        str(k): int(v) for k, v in predictions["target_overlap"].value_counts().sort_index().items()
    }
    novel_flag = predictions["target_overlap"].eq("training_unseen").to_numpy()
    if not np.array_equal(novel_flag, ~seen):
        raise ValueError("target_overlap and sequence_seen_in_training are inconsistent")
    if seen.any() and predictions.loc[seen, "target_overlap"].eq("training_unseen").any():
        raise ValueError("a training-seen target is marked training_unseen")

    per_serotype = {
        str(serotype): metric_summary(group)
        for serotype, group in predictions.groupby("target_serotype", sort=True)
    }
    per_overlap = {
        str(category): metric_summary(group)
        for category, group in predictions.groupby("target_overlap", sort=True)
    }

    result = {
        "model_prediction_check": {
            "max_abs_probability_difference": max_delta,
            "exact_match_after_rounding_6dp": bool(np.array_equal(
                np.round(recalculated, 6), np.round(p, 6)
            )),
            "probability_atol": args.probability_atol,
        },
        "overall": metric_summary(predictions),
        "overall_auc_95ci": bootstrap_auc_ci(y, p, args.n_boot, args.seed),
        "sequence_stratified": {
            "training_unseen": metric_summary(novel),
            "training_seen": metric_summary(previously_seen),
        },
        "target_overlap_counts": overlap_counts,
        "target_overlap_metrics": per_overlap,
        "per_serotype": per_serotype,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "overlap_note": (
            "training_unseen and training_seen describe exact E-protein sequence overlap; "
            "seen_in_own_panel pairs remain target-excluded during feature construction "
            "but are not independent of the antibody's measured training panel"
        ),
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
