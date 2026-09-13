# NeuroSelect

NeuroSelect is a reproducible research codebase for non-invasive neural
signal-to-text experiments. It currently provides a compact character-level
CTC baseline over public SpanishBCBL/DECOMEG MEG data.

Research hypothesis for later phases:

> Can selectively conditioning an LLM on temporally localized neural evidence
> improve faithfulness of non-invasive brain-to-text decoding compared with
> uniformly providing neural representations to the language model?

This repository currently builds the data and baseline foundation only. It
does **not** implement the Evidence Selector, LLM conditioning, LoRA,
uncertainty estimation, active calibration, or product UI.

## Current status

| Area | Status |
|---|---|
| Synthetic CTC sanity check | Passed |
| Public dataset audit | Completed |
| Official SpanishBCBL event extraction | Passed on S22 subset |
| SpanishBCBL label mapping | Verified |
| Real-data signal extraction | Completed for 8 trials |
| One-trial forward pass + finite CTC loss | Passed |
| Real-data alignment audit | Completed |
| CTC geometry audit | All selected trials structurally feasible |
| Tiny real-data overfit | Not yet successful |
| Scientific real CER/WER benchmark | Not reported |
| Evidence Selector / LLM | Not implemented |

Current forensic conclusion:

**NOT YET ESTABLISHED** that the complete real-data problem is technically
learnable by the compact baseline. Mapping, event alignment, target encoding,
and CTC geometry are valid. A short-window filtering mismatch was found and
fixed. Model/optimization learnability remains unresolved.

## Dataset and access

Brain2Qwerty v2 data is documented as embargoed by its official repository.
NeuroSelect therefore uses the public SpanishBCBL/DECOMEG release:

- Dataset: <https://huggingface.co/datasets/bcbl190626/SpanishBCBL>
- License: CC BY-NC 4.0, subject to dataset terms
- Official code: <https://github.com/facebookresearch/brain2qwerty>
- Vendored official revision: `5f9889621d0df391c5aab37c996683d308e6e926`
- Full public archive: approximately 262 GB

Only one small MEG recording block was downloaded locally:

```text
MEG/FIF/22_9788/231214/block1.fif
MEG/logs/S22-session1_block1_list1.mat
```

Raw and processed neural recordings are excluded from Git by `.gitignore`.
Users must obtain data through permitted official access before running
real-data preparation.

## Real subset

The current local subset represents:

- Subject: S22
- Session: 1
- Block: block1
- Modality: MEG
- Original sampling rate: 1000 Hz
- Channels: 306 MEG
- Selected production trials: 8
- Processed sampling rate: 50 Hz
- Processed arrays: `(time, channels)` float32 `.npy`

Generated local artifacts:

```text
data/processed/spanishbcbl_subset/manifest.jsonl
data/manifests/spanishbcbl_meg_subset.csv
data/raw/spanishbcbl_s22/events_clean.pkl
results/real_data_alignment_examples.json
results/figures/spanishbcbl_trial_002.png
```

## Installation

### Conda

```powershell
conda env create -f environment.yml
conda activate neuroselect
$env:PYTHONPATH="src"
```

### pip / virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH="src"
```

Official event extraction additionally requires the installed
`neuralset`, `neuralfetch`, `exca`, `mne`, and related Brain2Qwerty
dependencies. The pinned upstream dependency specification is available at:

```text
vendor/brain2qwerty/pyproject.toml
vendor/brain2qwerty/requirements.lock
```

## Tests

Run NeuroSelect tests:

```powershell
$env:PYTHONPATH="src"
python -m pytest -q tests
```

The current NeuroSelect suite covers:

- generic vocabulary encoding/decoding
- verified SpanishBCBL mapping
- real-style JSONL manifest loading
- signal shape and finite-value checks
- Conv1D + BiGRU + CTC dimensions
- CTC loss compatibility
- CER/WER
- duplicate-text split protection
- repeated-label CTC geometry

Vendored Brain2Qwerty tests are separate. Some require optional dependencies
such as `Levenshtein` and `neuraltrain`; they are not part of the NeuroSelect
test gate.

## Baseline model

```text
(batch, time, channels)
        ↓
Conv1D + GELU
        ↓
Conv1D + GELU
        ↓
bidirectional GRU
        ↓
linear vocabulary head
        ↓
CTC loss
```

Source:

```text
src/neuroselect/models.py
src/neuroselect/training.py
src/neuroselect/data.py
```

The SpanishBCBL vocabulary contains 30 classes:

- CTC blank
- 28 official character/symbol classes
- `<space>` mapped to space
- `<special>` mapped to `@`
- `<number>` mapped to `9`

Mapping details:

```text
src/neuroselect/vocab_spanishbcbl.py
docs/DATASET_MAPPING.md
```

## Prepare real data

Run from repository root after placing permitted raw files under
`data/raw/spanishbcbl_s22/`:

```powershell
$env:PYTHONPATH="vendor\brain2qwerty;src"
python scripts\prepare_real_subset.py data\raw\spanishbcbl_s22
```

Pipeline:

```text
raw FIF + MATLAB log
→ official Brain2Qwerty event extraction
→ official sentence/keystroke preprocessing
→ continuous MEG filtering: 0.1–20 Hz
→ resampling: 50 Hz
→ continuous-recording RobustScaler
→ official sentence crop
→ baseline correction + clamp ±5
→ `.npy` signal windows + JSONL/CSV manifests
```

The script preserves official event boundaries and uses
`sentence_typed` as target text. It does not download data automatically.

## Run synthetic sanity check

```powershell
$env:PYTHONPATH="src"
python scripts\sanity_check.py
```

Expected behavior: synthetic CTC loss decreases strongly on a tiny encoded
example. This validates tensor flow and optimization mechanics, not neural
decoding quality.

## Run real-data forensics

The Phase 3 diagnostic runner audits one real-data path end to end:

```powershell
$env:PYTHONPATH="vendor\brain2qwerty;src"
python scripts\real_baseline_forensics.py data\raw\spanishbcbl_s22 --epochs 100
```

Outputs:

```text
results/real_baseline_debug_trial.json
results/mapping_reconstruction_examples.json
results/alignment_audit.json
results/preprocessing_statistics.json
results/ctc_geometry_report.json
results/optimization_diagnostic.json
results/figures/debug/
```

Diagnostics include:

- raw-to-target trace for one trial
- official label reconstruction
- event timestamps and inter-event density
- signal-stage statistics
- CTC `T`, `U`, `T/U`, repeated-label checks
- learning-rate comparison
- clipping comparison
- normalization comparison
- random-target control
- artificial-target control

## Train and evaluate baseline

Training entry point:

```powershell
$env:PYTHONPATH="src"
python -m neuroselect.training data\processed\spanishbcbl_subset\manifest.jsonl `
  --epochs 10 `
  --batch-size 2 `
  --hidden 64 `
  --output results\checkpoint.pt
```

Evaluation entry point:

```powershell
$env:PYTHONPATH="src"
python scripts\evaluate.py `
  data\processed\spanishbcbl_subset\manifest.jsonl `
  results\checkpoint.pt `
  --output results\baseline_metrics.json `
  --examples results\baseline_predictions.txt
```

Do not interpret metrics from this eight-trial, one-subject subset as a
scientific generalization benchmark. Current project policy blocks real
CER/WER reporting until tiny real-data learnability and leakage-safe
evaluation are established.

## Repository layout

```text
src/neuroselect/
  data.py                    Manifest loader, signal dataset, CTC collation
  diagnostics.py             CTC geometry validation
  metrics.py                 CER/WER and edit distance
  models.py                  Compact Conv1D + BiGRU + CTC model
  splitting.py               Duplicate-text-safe split helper
  training.py                Training loop and CLI
  vocab.py                   Generic baseline vocabulary
  vocab_spanishbcbl.py       Verified SpanishBCBL vocabulary

scripts/
  sanity_check.py            Synthetic CTC overfit check
  prepare_real_subset.py     Official event path to local manifests
  real_baseline_forensics.py Phase 3 forensic diagnostics
  evaluate.py                Prediction and metric export

tests/
  test_baseline.py           NeuroSelect regression tests

docs/
  OFFICIAL_PIPELINE.md       Official event-building explanation
  REAL_DATA_PIPELINE.md      Real-data preparation details
  DATASET_MAPPING.md         Label table and reconstruction evidence
  ALIGNMENT_AUDIT.md         Timestamp and event audit
  PREPROCESSING_AUDIT.md     Signal-stage audit
  CTC_GEOMETRY.md            CTC feasibility audit
  LEAKAGE_ANALYSIS.md        Leakage limits and split policy
  COMPUTE_BENCHMARK.md       Runtime observations
```

## Scientific guardrails

- Do not claim Brain2Qwerty v2 data access.
- Do not bypass dataset authentication, embargo, or license restrictions.
- Do not download the full archive for this project phase.
- Do not claim state-of-the-art performance.
- Do not claim novelty before evidence supports it.
- Do not report tiny-subset CER/WER as generalization.
- Do not add Evidence Selector, LLM, calibration, uncertainty, or UI before
  baseline learnability is established.

## Documentation index

- `dataset_audit.md` — access, size, license, and dataset limitations
- `DATASET_CARD.md` — concise dataset summary
- `BASELINE.md` — model scope and current baseline status
- `EXPERIMENT_LOG.md` — phase-by-phase experiment record
- `docs/OFFICIAL_PIPELINE.md` — official source pipeline
- `docs/REAL_DATA_PIPELINE.md` — local extraction path
- `docs/DATASET_MAPPING.md` — verified token mapping
- `docs/ALIGNMENT_AUDIT.md` — event/text alignment findings
- `docs/PREPROCESSING_AUDIT.md` — preprocessing findings
- `docs/CTC_GEOMETRY.md` — sequence feasibility findings
- `docs/LEAKAGE_ANALYSIS.md` — leakage constraints
- `docs/COMPUTE_BENCHMARK.md` — compute observations

## License and attribution

NeuroSelect project files are provided for research use. Public dataset files
remain subject to SpanishBCBL/DECOMEG license and attribution requirements.
The vendored Brain2Qwerty source retains its upstream license and notices in
`vendor/brain2qwerty/LICENSE`.
