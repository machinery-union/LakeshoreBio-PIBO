#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_predict.py — PIBO default cache-lookup inference demo
=============================================================
Demonstrates the "model availability" path of the public PIBO release: given an
antibody's IC50 panel (strains measured against the antibody) and a target
E-protein strain, predict the probability that the antibody neutralizes the
target — using ONLY the frozen 49-dim RandomForest model and the precomputed
ESM-2 epitope-bin embedding cache (`bin_pooled.npz`). No ESM-2 run, no antibody
sequence, no antibody structure.

Two demonstrations
------------------
  1. (default) *Bootstrap from the released training set* — pick a real antibody
     from `data/PIBO_train_859.csv` that has measured IC50s across all four
     DENV serotypes, leave-one-out predict each measured strain, and compare the
     predicted breadth against the true neutralization labels. This runs
     end-to-end with zero user input and is fully reproducible.

  2. *Four-representative-strain (anchor) panel* — the four anchor strains below
     are the recommended minimal panel (one DENV strain per serotype). Their
     E-protein embeddings are already in the 8,487-strain cache, so a user only
     needs to supply the four measured log10 IC50 values. See `predict_anchor_panel()`
     at the bottom of this file.

Recommended minimal panel (the four anchor strains)
---------------------------------------------------
    DENV1  ACF49259.1
    DENV2  AAC59275.1
    DENV3  ALS05358.1
    DENV4  AAX48017.1

Usage
-----
    python example_predict.py                          # bootstrap demo (default)
    python example_predict.py --antibody Group_0001    # pick a specific antibody
    python example_predict.py --bin-pooled /path/to/bin_pooled.npz   # if downloaded from Zenodo

Frozen artifacts used (all under data/ by default)
--------------------------------------------------
    data/final_model_rf49.joblib     trained 49-dim RF pipeline
    data/final_model_config_49d.json 49 feature names (model order)
    data/eprotein/bin_pooled.npz     8,487 strains x (6,1280) float32 embeddings
    data/center_mu.npz               ESM-2 global centering mean (1280-dim)
    data/eprotein/e_coords_all.csv   e_id -> serotype lookup
    data/eprotein/E_full_8487.fasta    E-protein sequences (for sequence->strain matching)
"""

import argparse
import os

import pandas as pd

from pibo_inference import PiboInference


# ---- recommended minimal panel: the four anchor strains ---------------------
# One representative strain per DENV serotype. Their 6-bin ESM-2 embeddings are
# already present in the 8,487-strain `bin_pooled.npz` cache, so no embedding
# regeneration is needed — the user only provides the measured log10 IC50 values.
ANCHOR_PANEL = {
    "DENV1": "ACF49259.1",
    "DENV2": "AAC59275.1",
    "DENV3": "ALS05358.1",
    "DENV4": "AAX48017.1",
}


def build_inference(bin_pooled_path):
    """Load the frozen model + embedding cache. All paths default to data/."""
    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(here, "data")
    return PiboInference(
        model_path=os.path.join(data, "final_model_rf49.joblib"),
        config_path=os.path.join(data, "final_model_config_49d.json"),
        bin_pooled_path=bin_pooled_path,
        center_mu_path=os.path.join(data, "center_mu.npz"),
        coords_path=os.path.join(data, "eprotein", "e_coords_all.csv"),
        eprotein_fasta_path=os.path.join(data, "eprotein", "E_full_8487.fasta"),
    )


def predict_anchor_panel(inf, ic50_log10, target_accession):
    """Predict neutralization of `target_accession` using the four-anchor panel.

    Parameters
    ----------
    inf : PiboInference
        loaded frozen model + cache.
    ic50_log10 : dict
        {serotype: log10 IC50} for the four anchor strains, e.g.
        {"DENV1": -1.2, "DENV2": -0.3, "DENV3": 0.4, "DENV4": 1.1}.
        These are the antibody's *measured* values — replace with real data.
    target_accession : str
        GenBank accession of the target strain (must be in the 8,487 cache).

    Returns
    -------
    float
        probability of neutralization P(neutralizing).
    """
    panel = []
    for sero, acc in ANCHOR_PANEL.items():
        nid = inf.resolve_nid(accession=acc)
        if nid is None:
            raise ValueError(f"anchor {acc} not found in bin_pooled cache")
        panel.append((nid, sero, float(ic50_log10[sero])))

    tnid = inf.resolve_nid(accession=target_accession)
    if tnid is None:
        raise ValueError(f"target {target_accession} not found in bin_pooled cache")
    tsero = inf.get_serotype(tnid)

    proba, _ = inf.predict_pair(panel, tnid, tsero)
    return proba


def bootstrap_demo(inf, train_csv, antibody=None):
    """Leave-one-out breadth prediction for a real training antibody.

    Picks a training antibody with measured IC50s across all four serotypes,
    predicts each of its measured strains leaving that strain out of the panel,
    and compares predicted vs. true neutralizing labels.
    """
    df = pd.read_csv(train_csv)

    # choose an antibody that covers all four serotypes
    coverage = df.groupby("antibody_id")["serotype"].nunique()
    full = coverage[coverage == 4].index.tolist()
    if antibody is not None:
        if antibody not in coverage.index:
            raise ValueError(f"antibody {antibody} not in training set")
        ab = antibody
    else:
        ab = full[0]

    rows = df[df["antibody_id"] == ab]
    print(f"\n=== bootstrap demo: antibody {ab} ===")
    print(f"    measured strains: {len(rows)}  "
          f"(serotypes {sorted(rows['serotype'].unique())})")

    # build the panel once (nid, serotype, log10_ic50) for this antibody
    panel = []
    for _, r in rows.iterrows():
        nid = inf.resolve_nid(accession=r["eprotein_id"], sequence=r.get("E_sequence"))
        if nid is None:
            continue
        sero = inf.get_serotype(nid, r.get("serotype"))
        panel.append((nid, sero, float(r["log10_ic50_ug_ml"])))

    print(f"    {'target':<16} {'sero':<6} {'P(neut)':>8}  {'true':>5}")
    n_correct = 0
    n_pred_pos = 0
    for nid, sero, ic50 in panel:
        # leave-one-out: exclude the target strain itself from the panel
        loo_panel = [(n, s, v) for n, s, v in panel if n != nid]
        proba, _ = inf.predict_pair(loo_panel, nid, sero)
        true = 1 if ic50 <= 0 else 0
        pred = 1 if proba >= 0.5 else 0
        n_correct += int(pred == true)
        n_pred_pos += pred
        print(f"    {nid:<16} {sero:<6} {proba:>8.3f}  {true:>5}")

    n_total = len(panel)
    true_breadth = sum(1 for _, _, v in panel if v <= 0) / n_total
    pred_breadth = n_pred_pos / n_total
    print(f"\n    LOO accuracy = {n_correct}/{n_total} = {n_correct/n_total:.3f}")
    print(f"    true breadth = {true_breadth:.3f}   "
          f"predicted breadth (threshold 0.5) = {pred_breadth:.3f}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="PIBO default cache-lookup inference demo")
    ap.add_argument("--bin-pooled",
                    default=os.path.join(here, "data", "eprotein", "bin_pooled.npz"),
                    help="path to bin_pooled.npz (default: data/eprotein/)")
    ap.add_argument("--antibody", default=None,
                    help="training antibody id (default: first covering all 4 serotypes)")
    ap.add_argument("--train-csv",
                    default=os.path.join(here, "data", "PIBO_train_859.csv"),
                    help="path to PIBO_train_859.csv")
    args = ap.parse_args()

    if not os.path.exists(args.bin_pooled):
        raise SystemExit(
            f"bin_pooled.npz not found at {args.bin_pooled}. "
            "Download it from the Zenodo deposit linked in the paper "
            "(it exceeds GitHub's 100 MB limit), or pass --bin-pooled <path>."
        )

    inf = build_inference(args.bin_pooled)

    print("\nRecommended minimal panel (four anchor strains):")
    for sero, acc in ANCHOR_PANEL.items():
        print(f"    {sero:<6} {acc}")

    bootstrap_demo(inf, args.train_csv, args.antibody)


if __name__ == "__main__":
    main()
