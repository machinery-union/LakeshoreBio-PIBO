# PIBO — Panel-In, Breadth-Out

PIBO is a machine-learning **pipeline** that predicts whether a dengue (DENV)
antibody neutralizes a target strain from a small panel of IC₅₀ measurements,
without requiring antibody sequences or structures. E-protein identities are used
to retrieve precomputed ESM-2 epitope-bin embeddings.

This repository contains only the model-training/evaluation and inference scripts,
the frozen model, and the supporting files that are not distributed through the
journal Supplementary Data or Zenodo. Internal development files and intermediate
analysis pipelines are not included.

## 1. Three non-duplicated distribution channels

| Channel | Contents |
|---|---|
| Scientific Reports Supplementary Data | Nine CSV files: curated measurements, feature matrices, out-of-fold probabilities, frozen predictions, and the E-protein accession/status manifest |
| Zenodo | `bin_pooled.npz`: the complete 8,487-strain ESM-2 six-bin embedding cache (~244 MB) |
| GitHub | Six scripts, the frozen RF model and configuration, `center_mu.npz`, `e_coords_all.csv`, `E_full_8487.fasta`, documentation, and dependency information |

The Supplementary Data CSV files and `bin_pooled.npz` are **not duplicated in this repository**.
To run the scripts, assemble the three channels as described below.

Zenodo record: **[https://zenodo.org/records/24522271](https://zenodo.org/records/24522271)**

## 2. Repository contents

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── SHA256SUMS.txt
├── reproduce_training_49d.py
├── reproduce_blind1.py
├── evaluate_blind2.py
├── pibo_inference.py
├── predict_anchor_panel.py
├── example_predict.py
└── data/
    ├── final_model_rf49.joblib
    ├── final_model_config_49d.json
    ├── center_mu.npz
    └── eprotein/
        ├── e_coords_all.csv
        └── E_full_8487.fasta
```

- `reproduce_training_49d.py`: training LOGO-CV artifact evaluation, with optional refitting.
- `reproduce_blind1.py`: Blind-1 evaluation and frozen-prediction verification.
- `evaluate_blind2.py`: Blind-2 evaluation from the released censored labels and predictions.
- `pibo_inference.py`: 49-feature construction and prediction library/CLI.
- `predict_anchor_panel.py`: prediction from the fixed four-anchor panel.
- `example_predict.py`: a leave-one-out demonstration using a released training antibody.

The complete E-protein FASTA is included as a convenience copy. The original NCBI
Protein/Nucleotide records referenced by `E_full_8487_accession_manifest.csv` remain
the primary sequence sources.

## 3. Assemble the release

### 3.1 Add the Supplementary Data

Download the nine Supplementary Data CSV files from the journal and place them in
`data/` with their filenames unchanged.

| Supplementary file | Destination in this repository |
|---|---|
| `PIBO_train_859.csv` | `data/PIBO_train_859.csv` |
| `PIBO_blind1_82.csv` | `data/PIBO_blind1_82.csv` |
| `PIBO_blind2_40.csv` | `data/PIBO_blind2_40.csv` |
| `features_train_49d.csv` | `data/features_train_49d.csv` |
| `features_blind1_49d.csv` | `data/features_blind1_49d.csv` |
| `oof_final_model_49d.csv` | `data/oof_final_model_49d.csv` |
| `blind1_predictions_300105.csv` | `data/blind1_predictions_300105.csv` |
| `blind2_predictions_300105.csv` | `data/blind2_predictions_300105.csv` |
| `E_full_8487_accession_manifest.csv` | `data/E_full_8487_accession_manifest.csv` |

`E_full_8487_accession_manifest.csv` provides provenance and is not required by the
three evaluation commands.

### 3.2 Add the Zenodo cache

Download `bin_pooled.npz` from **[https://zenodo.org/records/24522271](https://zenodo.org/records/24522271)** and place it at:

```text
data/eprotein/bin_pooled.npz
```

The cache contains 8,487 keys, each holding an uncentered `(6, 1280)` float32
array. The loader subtracts `center_mu.npz` once; do not centre the arrays again.

## 4. Environment

The released commands were tested with Python 3.12.3 and:

```text
numpy==2.4.6
pandas==3.0.5
scipy==1.18.0
scikit-learn==1.9.0
joblib==1.5.3
```

Create an isolated environment and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The optional exact LOGO refit (`--retrain-logo`) requires a separate environment
with `scikit-learn==1.6.1`; the script enforces this requirement.

## 5. Reproduce the core evaluations

Run from the repository root after adding the Supplementary Data files required for
evaluation:

```bash
python reproduce_training_49d.py
python reproduce_blind1.py
python evaluate_blind2.py
```

Expected aggregate results:

| Evaluation | Pairs | AUC | AUC 95% CI | Accuracy |
|---|---:|---:|---:|---:|
| Training OOF (LOGO-CV) | 859 | 0.930289 | [0.913718, 0.945550] | 0.845169 |
| Blind-1 | 82 | 0.917193 | [0.843926, 0.970141] | 0.865854 |
| Blind-2 | 40 | 0.753247 | [0.548476, 0.921569] | 0.725000 |

These commands evaluate the released feature/prediction artifacts. They do not
rebuild ESM-2 embeddings or regenerate the 49-dimensional matrices. Blind-2 is
evaluation-only; its continuous measurements and 49-dimensional matrix are not
part of the release.

To refit all 194 antibody-level LOGO folds from `features_train_49d.csv` and compare
the probabilities with `oof_final_model_49d.csv`:

```bash
python reproduce_training_49d.py --retrain-logo
```

This optional command requires `scikit-learn==1.6.1` and does not replace the
frozen deployment model.

## 6. Run inference

After adding the Zenodo cache, supply four anchor IC₅₀ values in DENV1–DENV4 order:

```bash
python predict_anchor_panel.py \
    --ic50-ug-ml 0.01 0.1 1.0 10.0 \
    --target ACF49259.1 AAC59275.1 \
    --out anchor_panel_predictions.csv
```

The fixed anchor identifiers are:

```text
DENV1  ACF49259.1
DENV2  AAC59275.1
DENV3  ALS05358.1
DENV4  AAX48017.1
```

Use `--all-cached` to score every cache key. The output for a selected target set
is a predicted-positive fraction; it is not automatically labelled global breadth.

Run the leave-one-out demonstration after adding `PIBO_train_859.csv` and the Zenodo
cache:

```bash
python example_predict.py
python example_predict.py --antibody Group_0001
```

`pibo_inference.py` can also be imported as a library or run as a CLI. A panel
table contains `antibody_id`, `eprotein_id`, and `log10_ic50_ug_ml`; a target table
contains `antibody_id` and `eprotein_id`. Optional sequence/serotype columns can be
used as fallbacks. Targets absent from the 8,487-strain cache are skipped.

## 7. Reproduction boundary

The release supports:

1. artifact-level reproduction of the three reported core evaluations;
2. optional LOGO refitting from the released training feature matrix;
3. inference for strains present in the complete Zenodo cache.

It does not provide a turnkey ESM-2 regeneration pipeline for new sequences, and
it does not reproduce every intermediate analysis or figure in the internal
research directory. New out-of-cache targets require a compatible embedding to be
generated separately before feature construction.

## 8. Data dictionary and labels

`PIBO_train_859.csv` and `PIBO_blind1_82.csv` contain antibody and strain
identifiers, E-protein accession and sequence fields, serotype, IC₅₀/log10(IC₅₀),
and literature or patent provenance. The binary training label is `1` when
IC₅₀ ≤ 1 µg/mL and `0` otherwise.

`PIBO_blind2_40.csv` provides censored Blind-2 classes `<1`, `1-10`, and `>10`
(33/1/6 records); only `<1` is positive. `features_train_49d.csv` and
`features_blind1_49d.csv` contain the 49 model inputs. `oof_final_model_49d.csv`
contains antibody-level LOGO-CV probabilities, while `blind1_predictions_300105.csv`
and `blind2_predictions_300105.csv` contain frozen predictions.
`E_full_8487_accession_manifest.csv` maps the released identifiers to their NCBI
source records and record statuses.

## 9. Integrity and licence

From the repository root, before adding external files:

```bash
sha256sum -c SHA256SUMS.txt
```

This manifest covers only the GitHub payload. The Zenodo record has a separate
checksum manifest for the embedding cache.

The code and documentation are distributed under the MIT License. Third-party
models, dependencies, and source databases remain subject to their own terms.
Cite the PIBO manuscript and the original data sources when reusing this release.
