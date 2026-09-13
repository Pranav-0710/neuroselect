# NeuroSelect

NeuroSelect is a research codebase for reproducible neural-signal-to-text
experiments. Phase 1 establishes a compact character-level CTC baseline
without an LLM, evidence selector, calibration system, or application UI.

## Current data decision

Brain2Qwerty v2 is not publicly downloadable according to the official
repository. The accessible related release is the SpanishBCBL/DECOMEG dataset
(`bcbl190626/SpanishBCBL`), released under CC BY-NC 4.0 and approximately
262 GB. The baseline therefore consumes a local manifest rather than assuming
the full archive is present.

## Quick start

```powershell
python -m pytest -q
python -m neuroselect.training --help
python scripts/evaluate.py --help
```

The manifest format and data layout are documented in `data/README.md`.
`BASELINE.md`, `dataset_audit.md`, and `DATASET_CARD.md` record the current
scientific and operational status.

## Phase 2 real-data preparation

The verified public S22 subset can be prepared with:

```powershell
$env:PYTHONPATH="vendor\brain2qwerty;src"
python scripts\prepare_real_subset.py data\raw\spanishbcbl_s22
```

This produces the real manifest and compact 50 Hz MEG windows. The official
pipeline and mapping are documented in `docs/`; the current real-data smoke
test passes, but tiny real-data overfitting has not succeeded, so no
scientific CER/WER is reported.
