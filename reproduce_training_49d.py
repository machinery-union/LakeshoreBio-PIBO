#!/usr/bin/env python3
"""Reproduce the final PIBO 49-dim LOGO-CV metrics from released files.

By default this script evaluates the released out-of-fold (OOF) probabilities.
With --retrain-logo it also refits the 194 antibody-level folds from the released
49-dimensional matrix and compares the new probabilities with the released OOF
artifact. It never replaces the frozen model or the released OOF file.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
EXACT_RETRAIN_SKLEARN = "1.6.1"


def clean_nid(raw: object) -> str:
    return str(raw).strip().lstrip(">").split()[0].split("[")[0].strip().lower()


def pair_keys(df: pd.DataFrame) -> pd.Series:
    return df["antibody_id"].astype(str) + "|" + df["eprotein_id"].map(clean_nid)


def validate_predictions(df: pd.DataFrame) -> None:
    y = df["y_true"].to_numpy(int)
    p = df["proba_neutralizing"].to_numpy(float)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("probabilities must be finite and within [0, 1]")
    expected = (p >= 0.5).astype(int)
    if not np.array_equal(df["predicted_label"].to_numpy(int), expected):
        raise ValueError("predicted_label is inconsistent with the 0.5 threshold")


def metric_summary(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "AUC": float(roc_auc_score(y, p)),
        "Accuracy": float(accuracy_score(y, pred)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, pred)),
        "BalAcc": float(balanced_accuracy_score(y, pred)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "TN_FP_FN_TP": confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist(),
    }


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(roc_auc_score(y[idx], p[idx]))
    if not values:
        raise ValueError("bootstrap produced no two-class AUC sample")
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def bootstrap_all_metrics(y: np.ndarray, p: np.ndarray, n_boot: int, seed: int) -> OrderedDict:
    """Match the frozen pipeline's shared-RNG, metric-major bootstrap order."""
    rng = np.random.default_rng(seed)
    n = len(y)
    pred = (p >= 0.5).astype(int)
    metric_fns = OrderedDict([
        ("AUC", lambda yt, yp, ypr: roc_auc_score(yt, yp) if len(np.unique(yt)) >= 2 else np.nan),
        ("F1", lambda yt, yp, ypr: f1_score(yt, ypr, zero_division=0)),
        ("MCC", lambda yt, yp, ypr: matthews_corrcoef(yt, ypr)),
        ("BalAcc", lambda yt, yp, ypr: balanced_accuracy_score(yt, ypr)),
        ("Precision", lambda yt, yp, ypr: precision_score(yt, ypr, zero_division=0)),
        ("Recall", lambda yt, yp, ypr: recall_score(yt, ypr, zero_division=0)),
    ])
    results = OrderedDict()
    for name, fn in metric_fns.items():
        values = []
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            if name == "AUC" and len(np.unique(y[idx])) < 2:
                continue
            values.append(fn(y[idx], p[idx], (p[idx] >= 0.5).astype(int)))
        ci = np.percentile(values, [2.5, 97.5]) if values else [np.nan, np.nan]
        results[name] = {
            "point": float(fn(y, p, pred)),
            "95CI": [float(ci[0]), float(ci[1])],
        }
    return results


def retrain_logo(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, params: dict) -> np.ndarray:
    probabilities = np.full(len(y), np.nan, dtype=float)
    splitter = LeaveOneGroupOut()
    for train_idx, test_idx in splitter.split(X, y, groups):
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            probabilities[test_idx] = float(y_train[0])
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            RandomForestClassifier(**params),
        )
        model.fit(X.iloc[train_idx], y_train)
        probabilities[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("LOGO-CV left unassigned probabilities")
    return probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="bootstrap repetitions per metric for the six-metric table")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrain-logo", action="store_true",
                        help="refit all antibody-level LOGO folds from the released matrix")
    parser.add_argument("--probability-atol", type=float, default=5.1e-7,
                        help="allowed OOF difference, accounting for six-decimal released values")
    args = parser.parse_args()
    if args.n_boot < 1:
        parser.error("--n-boot must be positive")
    if args.retrain_logo and sklearn.__version__ != EXACT_RETRAIN_SKLEARN:
        parser.error(
            f"--retrain-logo requires scikit-learn=={EXACT_RETRAIN_SKLEARN} for the exact "
            f"OOF comparison; found {sklearn.__version__}. Use a separate environment with "
            f"scikit-learn=={EXACT_RETRAIN_SKLEARN}; requirements.txt targets artifact "
            "evaluation, not exact retraining. No LOGO folds were run. Evaluation of the "
            "released OOF artifact without --retrain-logo does not require this version check."
        )

    data = args.data_dir
    cfg = json.loads((data / "final_model_config_49d.json").read_text(encoding="utf-8"))
    feature_names = list(cfg["feature_names"])
    if len(feature_names) != 49 or len(set(feature_names)) != 49:
        raise ValueError("configuration must contain 49 unique feature names")

    features = pd.read_csv(data / "features_train_49d.csv", encoding="utf-8-sig")
    oof = pd.read_csv(data / "oof_final_model_49d.csv", encoding="utf-8-sig")
    if len(features) != len(oof) or not pair_keys(features).equals(pair_keys(oof)):
        raise ValueError("feature and OOF rows are not aligned by antibody and normalized E-protein ID")
    if not np.array_equal(features["y_true"].to_numpy(int), oof["y_true"].to_numpy(int)):
        raise ValueError("feature and OOF labels differ")
    validate_predictions(oof)

    X = features.reindex(columns=feature_names).apply(pd.to_numeric, errors="raise")
    y = oof["y_true"].to_numpy(int)
    p = oof["proba_neutralizing"].to_numpy(float)
    groups = oof["antibody_id"].astype(str).to_numpy()

    result = {
        "evaluation_source": "released antibody-level LOGO-CV OOF probabilities",
        "environment": {
            "sklearn": sklearn.__version__,
            "exact_retrain_sklearn": EXACT_RETRAIN_SKLEARN,
        },
        "n_antibodies": int(len(np.unique(groups))),
        "overall": metric_summary(y, p),
        "paper_auc_95ci_1000_boot": bootstrap_auc_ci(y, p, n_boot=1000, seed=args.seed),
        "six_metric_bootstrap": {
            "n_boot_per_metric": args.n_boot,
            "seed": args.seed,
            "metrics": bootstrap_all_metrics(y, p, args.n_boot, args.seed),
        },
        "deployment_model_note": (
            "final_model_rf49.joblib is fitted on all training pairs; its training-set "
            "predictions are not a substitute for these OOF metrics"
        ),
    }

    if args.retrain_logo:
        retrained = retrain_logo(X, y, groups, dict(cfg["params"]))
        delta = np.abs(retrained - p)
        rounded_match = np.array_equal(np.round(retrained, 6), np.round(p, 6))
        result["logo_retraining_check"] = {
            "n_folds": int(len(np.unique(groups))),
            "sklearn": sklearn.__version__,
            "overall": metric_summary(y, retrained),
            "max_abs_probability_difference": float(delta.max()),
            "mean_abs_probability_difference": float(delta.mean()),
            "exact_match_after_rounding_6dp": bool(rounded_match),
            "probability_atol": args.probability_atol,
        }
        if float(delta.max()) > args.probability_atol:
            raise RuntimeError(
                f"retrained OOF differs from released OOF under scikit-learn {sklearn.__version__}: "
                f"max abs delta={delta.max():.9g}, mean abs delta={delta.mean():.9g}, "
                f"released AUC={result['overall']['AUC']:.9g}, "
                f"retrained AUC={result['logo_retraining_check']['overall']['AUC']:.9g}; "
                "check the remaining dependency versions and feature ordering"
            )

    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
