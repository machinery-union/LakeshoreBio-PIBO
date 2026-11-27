#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predict DENV neutralization from the fixed four-anchor IC50 panel.

The four values must be supplied in DENV1, DENV2, DENV3, DENV4 order,
using exactly one input mode:

  --ic50-ug-ml IC50_DENV1 IC50_DENV2 IC50_DENV3 IC50_DENV4
  --log10-ic50-ug-ml LOG10_DENV1 LOG10_DENV2 LOG10_DENV3 LOG10_DENV4

Targets are cache accessions supplied with --target, or every cache key with
--all-cached. No antibody or protein sequence is read by this script.

Example (synthetic values only):
  python predict_anchor_panel.py \
      --ic50-ug-ml 0.01 0.1 1.0 10.0 \
      --target ACF49259.1 AAC59275.1 \
      --out anchor_panel_predictions.csv
"""

import argparse
import math
import os

import pandas as pd

from pibo_inference import ANCHOR_LIST, PiboInference


ANCHOR_PANEL = {
    "DENV1": "ACF49259.1",
    "DENV2": "AAC59275.1",
    "DENV3": "ALS05358.1",
    "DENV4": "AAX48017.1",
}
THRESHOLD = 0.5
OUTPUT_COLUMNS = [
    "target_id",
    "target_nid",
    "target_serotype",
    "proba_neutralizing",
    "predicted_label",
    "n_panel",
    "target_excluded_from_panel",
]


class AnchorPanelError(ValueError):
    """An input or cache-resolution error that must be shown to the user."""


def script_relative_defaults():
    """Return frozen model/data paths relative to this script, not the cwd."""
    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(here, "data")
    return {
        "model": os.path.join(data, "final_model_rf49.joblib"),
        "config": os.path.join(data, "final_model_config_49d.json"),
        "bin_pooled": os.path.join(data, "eprotein", "bin_pooled.npz"),
        "center_mu": os.path.join(data, "center_mu.npz"),
        "coords": os.path.join(data, "eprotein", "e_coords_all.csv"),
    }


def build_parser():
    defaults = script_relative_defaults()
    parser = argparse.ArgumentParser(
        description=(
            "PIBO prediction from four measured anchor IC50 values. "
            "Values are ordered DENV1 DENV2 DENV3 DENV4; targets must be "
            "accessions in the frozen embedding cache."
        )
    )
    ic50_group = parser.add_mutually_exclusive_group(required=True)
    ic50_group.add_argument(
        "--ic50-ug-ml",
        nargs=4,
        type=float,
        metavar=("DENV1", "DENV2", "DENV3", "DENV4"),
        help="four positive, finite IC50 values in ug/mL",
    )
    ic50_group.add_argument(
        "--log10-ic50-ug-ml",
        nargs=4,
        type=float,
        metavar=("DENV1", "DENV2", "DENV3", "DENV4"),
        help="four finite log10(IC50 in ug/mL) values",
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target",
        dest="targets",
        nargs="+",
        metavar="ACCESSION",
        help="one or more target accessions in the frozen cache",
    )
    target_group.add_argument(
        "--all-cached",
        action="store_true",
        help="predict every key in the frozen cache (8,487 in this release)",
    )
    parser.add_argument("--model", default=defaults["model"], help="frozen RF model")
    parser.add_argument("--config", default=defaults["config"], help="49-feature config JSON")
    parser.add_argument(
        "--bin-pooled", default=defaults["bin_pooled"], help="frozen E-protein embedding cache"
    )
    parser.add_argument("--center-mu", default=defaults["center_mu"], help="frozen centering vector")
    parser.add_argument("--coords", default=defaults["coords"], help="accession-to-serotype table")
    parser.add_argument(
        "--out",
        default="anchor_panel_predictions.csv",
        help="per-target output CSV (default: ./anchor_panel_predictions.csv)",
    )
    return parser


def validated_log10_panel(args):
    """Validate the selected input mode and return values in DENV1-4 order."""
    if args.ic50_ug_ml is not None:
        values = args.ic50_ug_ml
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise AnchorPanelError(
                "--ic50-ug-ml requires four positive, finite values in DENV1-DENV4 order"
            )
        return [math.log10(value) for value in values]

    values = args.log10_ic50_ug_ml
    if any(not math.isfinite(value) for value in values):
        raise AnchorPanelError(
            "--log10-ic50-ug-ml requires four finite values in DENV1-DENV4 order"
        )
    return [float(value) for value in values]


def build_anchor_panel(inf, log10_values):
    """Resolve every anchor explicitly; never omit an unresolved anchor."""
    panel = []
    for serotype, log10_ic50 in zip(ANCHOR_LIST, log10_values):
        accession = ANCHOR_PANEL[serotype]
        nid = inf.resolve_nid(accession=accession)
        if nid is None:
            raise AnchorPanelError(
                f"panel anchor {accession} ({serotype}) is not present in the embedding cache"
            )
        panel.append((nid, serotype, log10_ic50))
    if len({nid for nid, _, _ in panel}) != len(ANCHOR_PANEL):
        raise AnchorPanelError("the four panel anchors did not resolve to four distinct cache keys")
    return panel


def resolve_targets(inf, requested_targets=None, all_cached=False):
    """Resolve and validate all targets before any model prediction."""
    if all_cached:
        requested = sorted(inf.bin_pooled)
        resolved = [(nid, nid) for nid in requested]
    else:
        requested = requested_targets or []
        resolved = []
        for accession in requested:
            nid = inf.resolve_nid(accession=accession)
            if nid is None:
                raise AnchorPanelError(
                    f"target {accession!r} is not present in the embedding cache"
                )
            resolved.append((str(accession), nid))

    if not resolved:
        raise AnchorPanelError("no targets were selected")

    validated = []
    for requested_id, nid in resolved:
        serotype = inf.get_serotype(nid)
        if serotype not in ANCHOR_LIST:
            raise AnchorPanelError(
                f"target {requested_id!r} resolved to {nid!r} but has no valid DENV1-DENV4 serotype"
            )
        validated.append((requested_id, nid, serotype))
    return validated


def predict_resolved_targets(inf, full_panel, targets):
    """Predict each target, excluding an overlapping anchor for that target."""
    metadata = []
    feature_rows = []
    for requested_id, target_nid, target_serotype in targets:
        panel = [entry for entry in full_panel if entry[0] != target_nid]
        target_excluded = len(panel) != len(full_panel)
        if not panel:
            raise AnchorPanelError(f"target {requested_id!r} leaves an empty panel")

        feature_rows.append(inf.build_features(panel, target_nid, target_serotype))
        metadata.append((requested_id, target_nid, target_serotype, len(panel), target_excluded))

    # Batch only the RF call; feature construction remains PiboInference's exact path.
    X = pd.DataFrame(feature_rows).reindex(columns=inf.feature_names)
    probabilities = inf.model.predict_proba(X)[:, 1]
    rows = []
    for (requested_id, target_nid, target_serotype, n_panel, excluded), probability in zip(
        metadata, probabilities
    ):
        probability = float(probability)
        if not math.isfinite(probability):
            raise AnchorPanelError(f"target {requested_id!r} produced a non-finite probability")
        rows.append(
            {
                "target_id": requested_id,
                "target_nid": target_nid,
                "target_serotype": target_serotype,
                "proba_neutralizing": probability,
                "predicted_label": int(probability >= THRESHOLD),
                "n_panel": n_panel,
                "target_excluded_from_panel": excluded,
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        log10_values = validated_log10_panel(args)
        inf = PiboInference(
            model_path=args.model,
            config_path=args.config,
            bin_pooled_path=args.bin_pooled,
            center_mu_path=args.center_mu,
            coords_path=args.coords,
            eprotein_fasta_path=None,
        )
        full_panel = build_anchor_panel(inf, log10_values)
        targets = resolve_targets(inf, args.targets, args.all_cached)
        predictions = predict_resolved_targets(inf, full_panel, targets)
    except AnchorPanelError as exc:
        parser.exit(2, f"[predict_anchor_panel] error: {exc}\n")

    predictions.to_csv(args.out, index=False)
    if not args.all_cached:
        print(
            predictions.to_string(
                index=False,
                formatters={"proba_neutralizing": lambda value: f"{value:.6f}"},
            )
        )

    n_positive = int(predictions["predicted_label"].sum())
    n_selected = len(predictions)
    fraction = n_positive / n_selected
    print(
        "selected-target predicted-positive fraction "
        f"(threshold {THRESHOLD:.1f}): {n_positive}/{n_selected} = {fraction:.6f}"
    )
    print(f"[predict_anchor_panel] wrote {n_selected} predictions -> {args.out}")


if __name__ == "__main__":
    main()
