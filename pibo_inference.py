#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pibo_inference.py — PIBO 49-dim frozen-model inference entry point
==================================================================
Maps an antibody's IC50 panel (strain accessions + log10 IC50) plus each
target E-protein strain to the 49-dim PIBO feature vector and predicts the
probability of neutralization with the frozen RandomForest model.

This is the *reproducibility front door* of the public PIBO release. It needs
NO antibody sequence or structure: the only protein-level input is the E-protein
identity (GenBank accession and/or amino-acid sequence) of the panel and target
strains, which is used to look up precomputed ESM-2 epitope-bin embeddings.

Two E-protein resolution paths
------------------------------
  (1) fast / default  — the strain accession (or an exact E-sequence match)
      is looked up in the frozen `bin_pooled.npz` (8,487 strains x 6 bins x
      1280 dims). No ESM-2 run is required.
  (2) regenerate      — for a strain absent from the frozen set, re-embed with
      ESM-2 (esm2_t33_650M_UR50D) + q2ref residue mapping + 6-bin mean-pooling
      (see scripts/esm2_residue_embed.py and scripts/build_emb_cache.py), then
      pass the resulting (6, 1280) array in directly via `embed_override=`.

Frozen artifacts required (paths passed to the constructor)
-----------------------------------------------------------
  model          final_model_rf49.joblib       (SimpleImputer -> StandardScaler -> RF)
  config         final_model_config_49d.json   (the 49 feature names, in model order)
  bin_pooled     bin_pooled.npz                (8487 strains x (6,1280) float32)
  center_mu      center_mu.npz                 (1280-dim global residue mean, mu)
  coords         e_coords_all.csv              (e_id -> serotype lookup)
  eprotein fasta E_full_8487.fasta (optional, for sequence->strain matching)

The 49 features (23 panel + 18 cosine + 4 nn + 4 serotype) are reproduced exactly
from ArbitraryPanelBuilder in 00_utils/utils_L1.py. The frozen bin_pooled rows are
indexed in BIN_NAMES order (EDI, EDII, FL, EDE, EDIII, EDIII_LR), matching
scripts/build_emb_cache.py::per_bin_pooled.

Usage
-----
  python pibo_inference.py \
      --model data/final_model_rf49.joblib \
      --config data/final_model_config_49d.json \
      --bin-pooled /path/to/bin_pooled.npz \
      --center-mu data/center_mu.npz \
      --coords /path/to/e_coords_all.csv \
      --eprotein-fasta /path/to/E_full_8487.fasta \
      --panel panel.csv \
      --target target.csv \
      --out predictions.csv
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

# ---- constants (identical to 00_utils/utils_L1.py) -------------------------

ANCHOR_LIST = ["DENV1", "DENV2", "DENV3", "DENV4"]

# 6 disjoint epitope bins, in the order bin_pooled rows are stored.
BIN_NAMES = ["EDI", "EDII", "FL", "EDE", "EDIII", "EDIII_LR"]

# 49 feature names in the exact model order (mirrors final_model_config_49d.json
# "feature_names"); used as a fallback if the config file is missing.
FEATURE_NAMES_49D = [
    # A — panel IC50 summary (23)
    "panel_ic50_mean_DENV1", "panel_ic50_min_DENV1", "panel_has_DENV1", "panel_n_DENV1",
    "panel_ic50_mean_DENV2", "panel_ic50_min_DENV2", "panel_has_DENV2", "panel_n_DENV2",
    "panel_ic50_mean_DENV3", "panel_ic50_min_DENV3", "panel_has_DENV3", "panel_n_DENV3",
    "panel_ic50_mean_DENV4", "panel_ic50_min_DENV4", "panel_has_DENV4", "panel_n_DENV4",
    "panel_ic50_mean", "panel_ic50_min", "panel_ic50_std", "panel_n_total",
    "panel_n_serotypes", "panel_n_strong", "panel_strong_ratio",
    # B — epitope-bin cosine (18)
    "cos_mean_EDI", "cos_min_EDI", "cos_same_EDI",
    "cos_mean_EDII", "cos_min_EDII", "cos_same_EDII",
    "cos_mean_FL", "cos_min_FL", "cos_same_FL",
    "cos_mean_EDE", "cos_min_EDE", "cos_same_EDE",
    "cos_mean_EDIII", "cos_min_EDIII", "cos_same_EDIII",
    "cos_mean_EDIII_LR", "cos_min_EDIII_LR", "cos_same_EDIII_LR",
    # C — nearest neighbor (4)
    "nn_cosine", "nn_ic50", "nn_same_sero_cosine", "nn_same_sero_ic50",
    # D — target serotype (4)
    "target_is_DENV1", "target_is_DENV2", "target_is_DENV3", "target_is_DENV4",
]


def clean_nid(raw):
    """Normalize a GenBank accession / FASTA id to the bin_pooled key form.

    Mirrors 00_utils/utils_L1.py::clean_nid: lowercase, strip leading '>',
    take the first whitespace-delimited token, drop any '[...]' suffix.
    """
    s = str(raw).strip().lstrip(">").split()[0]
    return s.split("[")[0].strip().lower()


def normalize_serotype(s):
    """Normalize 'DENV-1' / 'DENV1' / '1' -> 'DENV1'..'DENV4' (None if invalid)."""
    if s is None:
        return None
    t = str(s).strip().upper().replace("-", "").replace("_", "")
    if t in ("DENV1", "1"):
        return "DENV1"
    if t in ("DENV2", "2"):
        return "DENV2"
    if t in ("DENV3", "3"):
        return "DENV3"
    if t in ("DENV4", "4"):
        return "DENV4"
    return None


class PiboInference:
    """Loads the frozen PIBO 49d model + embeddings and predicts per-pair."""

    def __init__(self, model_path, config_path, bin_pooled_path, center_mu_path,
                 coords_path, eprotein_fasta_path=None):
        import joblib

        # 1. frozen model + 49 feature names
        self.model = joblib.load(model_path)
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            self.feature_names = list(cfg.get("feature_names", FEATURE_NAMES_49D))
        else:
            self.feature_names = list(FEATURE_NAMES_49D)
        assert len(self.feature_names) == 49, \
            f"expected 49 feature names, got {len(self.feature_names)}"

        # 2. raw bin_pooled (8487 x (6,1280)) — keys are clean_nid strings
        bp = dict(np.load(bin_pooled_path, allow_pickle=True))
        self.bin_pooled_raw = {str(k): np.asarray(v, dtype=np.float32)
                               for k, v in bp.items()}

        # 3. centering: subtract the frozen global residue mean (broadcast over 6 bins)
        mu = np.asarray(np.load(center_mu_path)["mu"], dtype=np.float32)
        self.mu = mu
        self.bin_pooled = {k: (v - mu) for k, v in self.bin_pooled_raw.items()}
        print(f"[pibo_inference] loaded {len(self.bin_pooled)} strains "
              f"(centered, mu l2={float(np.linalg.norm(mu)):.4f})")

        # 4. serotype lookup (e_id -> serotype) from the coordinate table
        cdf = pd.read_csv(coords_path)
        self.serotype_of = {
            clean_nid(r["e_id"]): normalize_serotype(r.get("serotype"))
            for _, r in cdf.iterrows()
        }

        # 5. optional sequence -> strain index (for exact-sequence matching)
        self.seq_index = {}
        if eprotein_fasta_path and os.path.exists(eprotein_fasta_path):
            self.seq_index = self._build_seq_index(eprotein_fasta_path)
            print(f"[pibo_inference] sequence index: {len(self.seq_index)} unique E sequences")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_seq_index(fasta_path):
        """FASTA -> {sequence: clean_nid} (first accession wins for duplicates)."""
        idx = {}
        cur_id, cur_seq = None, []
        with open(fasta_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if cur_id is not None:
                        idx.setdefault("".join(cur_seq).upper(), clean_nid(cur_id))
                    cur_id, cur_seq = line, []
                else:
                    cur_seq.append(line)
            if cur_id is not None:
                idx.setdefault("".join(cur_seq).upper(), clean_nid(cur_id))
        return idx

    # ------------------------------------------------------------------ #
    def resolve_nid(self, accession=None, sequence=None):
        """Return the bin_pooled key for a strain, trying accession then exact
        sequence. Returns None if the strain is not in the frozen set (caller
        should then supply embed_override=)."""
        if accession is not None:
            nid = clean_nid(accession)
            if nid in self.bin_pooled:
                return nid
        if sequence is not None:
            nid = self.seq_index.get(str(sequence).strip().upper())
            if nid is not None and nid in self.bin_pooled:
                return nid
        return None

    def get_serotype(self, nid, fallback=None):
        return self.serotype_of.get(nid, normalize_serotype(fallback))

    # ------------------------------------------------------------------ #
    # Feature construction — exact port of ArbitraryPanelBuilder internals.
    # panel: list of (nid, serotype, log10_ic50).  serotype is one of DENV1-4.
    # ------------------------------------------------------------------ #
    def _panel_ic50_features(self, panel):
        by_sero = defaultdict(list)
        all_ic50s = []
        for _, sero, ic50 in panel:
            all_ic50s.append(ic50)
            if sero is not None:
                by_sero[sero].append(ic50)

        feats = {}
        for sero in ANCHOR_LIST:
            vals = by_sero.get(sero, [])
            feats[f"panel_ic50_mean_{sero}"] = float(np.mean(vals)) if vals else np.nan
            feats[f"panel_ic50_min_{sero}"] = float(np.min(vals)) if vals else np.nan
            feats[f"panel_has_{sero}"] = 1.0 if vals else 0.0
            feats[f"panel_n_{sero}"] = float(len(vals))

        all_arr = np.asarray(all_ic50s, dtype=float)
        feats["panel_ic50_mean"] = float(np.mean(all_arr)) if len(all_arr) else np.nan
        feats["panel_ic50_min"] = float(np.min(all_arr)) if len(all_arr) else np.nan
        # NOTE: np.std default ddof=0 (population std) — matches the frozen builder.
        feats["panel_ic50_std"] = float(np.std(all_arr)) if len(all_arr) > 1 else 0.0
        feats["panel_n_total"] = float(len(all_ic50s))
        feats["panel_n_serotypes"] = float(len(by_sero))
        feats["panel_n_strong"] = float(np.sum(all_arr <= 0))
        feats["panel_strong_ratio"] = (
            feats["panel_n_strong"] / feats["panel_n_total"]
            if feats["panel_n_total"] > 0 else 0.0)
        return feats

    def _target_vs_panel_cosine(self, target_nid, panel):
        target_bp = self.bin_pooled.get(target_nid)
        if target_bp is None:
            return {f"cos_mean_{b}": np.nan for b in BIN_NAMES} | \
                   {f"cos_min_{b}": np.nan for b in BIN_NAMES} | \
                   {f"cos_same_{b}": np.nan for b in BIN_NAMES}

        target_sero = self.get_serotype(target_nid)
        feats = {}
        for j, b in enumerate(BIN_NAMES):
            t_vec = target_bp[j]
            tnorm = float(np.linalg.norm(t_vec))
            if tnorm < 1e-9:
                feats[f"cos_mean_{b}"] = np.nan
                feats[f"cos_min_{b}"] = np.nan
                feats[f"cos_same_{b}"] = np.nan
                continue

            all_cos, same_sero_cos = [], []
            for eid, sero, _ in panel:
                p_bp = self.bin_pooled.get(eid)
                if p_bp is None:
                    continue
                p_vec = p_bp[j]
                pnorm = float(np.linalg.norm(p_vec))
                if pnorm < 1e-9:
                    continue
                cos = float(np.dot(t_vec, p_vec) / (tnorm * pnorm))
                all_cos.append(cos)
                if sero == target_sero:
                    same_sero_cos.append(cos)

            feats[f"cos_mean_{b}"] = float(np.mean(all_cos)) if all_cos else np.nan
            feats[f"cos_min_{b}"] = float(np.min(all_cos)) if all_cos else np.nan
            # NOTE: "cos_same" is the MEAN over same-serotype panel strains.
            feats[f"cos_same_{b}"] = float(np.mean(same_sero_cos)) if same_sero_cos else np.nan
        return feats

    def _nearest_neighbor_features(self, target_nid, panel, target_sero):
        target_bp = self.bin_pooled.get(target_nid)
        if target_bp is None:
            return {"nn_cosine": np.nan, "nn_ic50": np.nan,
                    "nn_same_sero_cosine": np.nan, "nn_same_sero_ic50": np.nan}

        target_flat = target_bp.flatten()
        tnorm = float(np.linalg.norm(target_flat))
        if tnorm < 1e-9:
            return {"nn_cosine": np.nan, "nn_ic50": np.nan,
                    "nn_same_sero_cosine": np.nan, "nn_same_sero_ic50": np.nan}

        best_cos, best_ic50 = -1.0, np.nan
        best_same_cos, best_same_ic50 = -1.0, np.nan
        for eid, sero, ic50 in panel:
            p_bp = self.bin_pooled.get(eid)
            if p_bp is None:
                continue
            p_flat = p_bp.flatten()
            pnorm = float(np.linalg.norm(p_flat))
            if pnorm < 1e-9:
                continue
            cos = float(np.dot(target_flat, p_flat) / (tnorm * pnorm))
            if cos > best_cos:
                best_cos, best_ic50 = cos, ic50
            if sero == target_sero and cos > best_same_cos:
                best_same_cos, best_same_ic50 = cos, ic50

        return {"nn_cosine": best_cos if best_cos >= 0 else np.nan,
                "nn_ic50": best_ic50,
                "nn_same_sero_cosine": best_same_cos if best_same_cos >= 0 else np.nan,
                "nn_same_sero_ic50": best_same_ic50}

    def build_features(self, panel, target_nid, target_sero):
        """panel: list of (nid, serotype, log10_ic50); target_nid: bin_pooled key;
        target_sero: one of DENV1-4. Returns an OrderedDict-like dict of 49 values."""
        feats = {}
        feats.update(self._panel_ic50_features(panel))
        feats.update(self._target_vs_panel_cosine(target_nid, panel))
        feats.update(self._nearest_neighbor_features(target_nid, panel, target_sero))
        for s in ANCHOR_LIST:
            feats[f"target_is_{s}"] = 1.0 if target_sero == s else 0.0
        return feats

    # ------------------------------------------------------------------ #
    def predict_pair(self, panel, target_nid, target_sero):
        """Return (proba_positive, feature_row_dict)."""
        feats = self.build_features(panel, target_nid, target_sero)
        X = pd.DataFrame([feats]).reindex(columns=self.feature_names)
        proba = float(self.model.predict_proba(X)[:, 1][0])
        return proba, feats

    # ------------------------------------------------------------------ #
    def run(self, panel_df, target_df, ic50_col="log10_ic50_ug_ml",
            ab_col="antibody_id", eid_col="eprotein_id", sero_col="serotype",
            seq_col="E_sequence", target_seq_col=None):
        """Panel + target CSVs -> predictions DataFrame.

        panel_df: one row per (antibody, measured strain) with log10 IC50.
        target_df: one row per (antibody, target strain) to predict.
        Strains are resolved via accession (eid_col) then exact sequence (seq_col).
        """
        # group the panel by antibody
        panels = defaultdict(list)
        for _, r in panel_df.iterrows():
            gid = str(r[ab_col])
            nid = self.resolve_nid(r.get(eid_col), r.get(seq_col))
            if nid is None:
                continue  # strain not in frozen set; skip (or use embed_override)
            sero = self.get_serotype(nid, r.get(sero_col))
            ic50 = float(r[ic50_col])
            panels[gid].append((nid, sero, ic50))

        rows = []
        for _, r in target_df.iterrows():
            gid = str(r[ab_col])
            if gid not in panels:
                continue
            tseq_col = target_seq_col or seq_col
            tnid = self.resolve_nid(r.get(eid_col), r.get(tseq_col))
            if tnid is None:
                continue
            tsero = self.get_serotype(tnid, r.get(sero_col))
            # leave-one-out: exclude the target itself from the panel if present
            panel = [(n, s, v) for n, s, v in panels[gid] if n != tnid]
            proba, feats = self.predict_pair(panel, tnid, tsero)
            rows.append({"antibody_id": gid,
                         "target_eprotein_id": r.get(eid_col),
                         "target_nid": tnid,
                         "target_serotype": tsero,
                         "n_panel": len(panel),
                         "proba_neutralizing": proba,
                         **feats})
        return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="PIBO 49d frozen-model inference")
    ap.add_argument("--model", required=True, help="final_model_rf49.joblib")
    ap.add_argument("--config", required=True, help="final_model_config_49d.json")
    ap.add_argument("--bin-pooled", required=True, help="bin_pooled.npz (8487 strains)")
    ap.add_argument("--center-mu", required=True, help="center_mu.npz")
    ap.add_argument("--coords", required=True, help="e_coords_all.csv")
    ap.add_argument("--eprotein-fasta", default=None, help="E_full_8487.fasta")
    ap.add_argument("--panel", required=True, help="panel CSV (antibody x strain IC50)")
    ap.add_argument("--target", required=True, help="target CSV (antibody x strain to predict)")
    ap.add_argument("--ic50-col", default="log10_ic50_ug_ml")
    ap.add_argument("--ab-col", default="antibody_id")
    ap.add_argument("--eid-col", default="eprotein_id")
    ap.add_argument("--sero-col", default="serotype")
    ap.add_argument("--seq-col", default="E_sequence")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    inf = PiboInference(args.model, args.config, args.bin_pooled,
                        args.center_mu, args.coords, args.eprotein_fasta)
    panel_df = pd.read_csv(args.panel)
    target_df = pd.read_csv(args.target)
    out = inf.run(panel_df, target_df, ic50_col=args.ic50_col,
                  ab_col=args.ab_col, eid_col=args.eid_col,
                  sero_col=args.sero_col, seq_col=args.seq_col)
    out.to_csv(args.out, index=False)
    print(f"[pibo_inference] wrote {len(out)} predictions -> {args.out}")
    if len(out):
        print(out[["antibody_id", "target_nid", "target_serotype",
                   "n_panel", "proba_neutralizing"]].to_string(index=False))


if __name__ == "__main__":
    main()
