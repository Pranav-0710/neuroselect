# NeuroSelect Agent Handoff

Generated: 2026-09-22

## Purpose

Transfers current NeuroSelect reproducibility and preprocessing work to a new
agent. Records validated findings, implementation status, constraints,
artifacts, test commands, remaining scope.

## Repository

- Repository: `Pranav-0710/neuroselect`
- Workspace: `Z:\PROJECTS\PROJECTS\Meta project brain`
- OS: Windows
- Current branch/state: worktree holds many untracked research artifacts and
  source files from the diagnostics below. Do not delete or reset them
  without explicit user approval.
- Latest pushed commit: `cf09788`.
- Main branch synced with `origin/main`.

## Non-negotiable project constraints

Do not:

- modify the existing Conv1D+BiGRU+CTC architecture;
- retrain sentence-level CTC unless a future task explicitly authorizes it;
- add an LLM or Evidence Selector;
- download the full/additional dataset;
- modify original raw MEG or MAT files;
- modify the official Brain2Qwerty source;
- silently alter event definitions, timestamps, or official timing;
- change the official dependency environment;
- run broad hyperparameter searches;
- replace the historical NeuroSelect preprocessing;
- add a global preprocessing switch that silently changes existing behavior.

All future experiments stay diagnostic, reproducible, explicitly scoped.

## Core verified facts

### Official Brain2Qwerty v1

- Exact official revision:
  `5f9889621d0df391c5aab37c996683d308e6e926`
- External official checkout:
  `C:\Users\prana\Envs\brain2qwerty-v1-checkout`
- External isolated environment:
  `C:\Users\prana\Envs\neuroselect-brain2qwerty-v1`
- Verified package versions:
  - `neuralset==0.2.2`
  - `neuraltrain==0.2.2`
  - `neuralfetch==0.2.2`
  - `exca==0.5.22`
  - `mne==1.11.0`
  - `torch==2.6.0`
  - `numpy==2.2.6`
  - `scipy==1.14.1`
  - `scikit-learn==1.8.0`
  - `pandas==2.2.3`
  - `pydantic==2.12.5`
  - `lightning==2.5.2`
  - `torchmetrics==1.7.3`
  - `dtw-python==1.7.4`

### Existing raw data

Raw recording:

`data\raw\spanishbcbl_s22\MEG\FIF\22_9788\231214\block1.fif`

Log/MAT data already in repo. Raw and MAT hashes checked in prior work,
unchanged.

Official event extraction succeeded for:

- subject S22
- session 1
- block 1
- 2,918 total events
- 2,119 keystrokes
- 666 words
- 132 sentences

### Fixed parity trial

Trial 2 sentence:

`la tasa excede las velocidades`

First five retained keystrokes:

| Event | Label | Timestamp (s) |
|---:|---|---:|
| 0 | `l` | 383.276 |
| 1 | `a` | 383.451 |
| 2 | `space` | 383.644 |
| 3 | `t` | 383.901 |
| 4 | `a` | 383.985 |

MEG event start for sample mapping: `308.0 s`.

## Historical diagnostics completed

### 5H full event-tensor linear probe

Implemented in:

`scripts\full_event_tensor_probe.py`

Configuration:

- existing 8 processed SpanishBCBL trials only;
- valid event-centered windows `[-0.2, +0.3]` seconds;
- event tensors as `25 x 306`;
- flattened dimension `7,650`;
- fold-local standardization and PCA;
- PCA kept ~95% variance under a practical bound;
- balanced logistic regression;
- seed 33;
- leave-one-trial-out evaluation;
- deterministic feature-column permutation control;
- chronological within-trial diagnostic.

Results:

- valid events: 210
- pooled accuracy: `0.095`
- macro F1: `0.040`
- balanced accuracy: `0.045`
- majority accuracy: `0.152`

Reading: keeping full temporal detail did not recover event-level
information in the historical event representation.

Artifacts:

- `results\full_event_tensor_probe.json`
- `results\figures\debug\full_event_tensor_probe.png`
- `results\figures\debug\full_tensor_fold_results.png`

### Official-v1 compatibility audit

Implemented in:

`scripts\official_v1_compatibility_audit.py`

Established:

- official baseline is per-event over relative `[0.0, +0.2]`;
- historical NeuroSelect baseline applies at sentence level;
- official orientation is `(channels, time)`;
- NeuroSelect model-facing storage is `(time, channels)`;
- official v1 holds subject/channel metadata NeuroSelect did not keep.

Artifacts:

- `results\official_v1_compatibility_audit.json`
- `results\official_v1_parity_event_examples.json`
- `results\official_revision_provenance.json`
- `results\official_dependency_inventory.json`
- `results\exca_compatibility_probe.json`
- `results\brain2qwerty_v1_environment.json`
- `results\brain2qwerty_v1_environment_requirements.txt`
- `results\official_v1_event_extraction_smoke.json`

### Exact event parity work

Initial current NeuroSelect representation differed materially from official
tensors. A temporary official-style baseline alone did not close the gap.

Root cause: historical NeuroSelect sentence-crops from the first retained
event. Early events lose pre-event continuous samples and can be zero-padded.
The official extractor works on the continuous recording and keeps those
samples.

Temporary continuous-scope implementation:

1. selects 306 MEG channels;
2. filters continuously at 0.1-20 Hz;
3. resamples continuously to 50 Hz;
4. applies continuous `RobustScaler`;
5. extracts event windows straight from the continuous recording;
6. subtracts per-event mean over first 10 samples;
7. clamps each event to `[-5, +5]`.

Reached exact element-level parity for all five fixed trial-2 events.

Artifacts:

- `scripts\continuous_scope_event_parity.py`
- `results\continuous_scope_event_parity.json`
- `results\figures\debug\continuous_scope_parity.png`
- `results\official_event_tensor_parity.json`
- `results\official_event_tensor_examples\trial_2_first_5.npz`
- `results\official_event_tensor_examples\trial_2_continuous_scope.npz`
- `results\figures\debug\official_event_parity.png`

Continuous-scope aggregate comparison recorded:

- official vs historical current NeuroSelect RMSE: ~`0.5397634`;
- official vs continuous-scope NeuroSelect RMSE: `0.0`;
- continuous-scope correlation: ~`1.0`;
- early events 0-1: exact parity;
- non-early events 2-4: exact parity.

## Latest implementation: gated official-v1-compatible variant

Latest task: add a separate explicit preprocessing path without changing the
historical path.

Implementation:

`src\neuroselect\official_v1_preprocessing.py`

Public API:

- `NeuroSelectOfficialV1EventPreprocessing.from_raw(...)`
- `NeuroSelectOfficialV1EventPreprocessing.extract_event(...)`
- `preprocess_event_official_v1(...)`

Behavior:

- reads the continuous FIF;
- selects MEG channels;
- filters continuously at 0.1-20 Hz;
- resamples to 50 Hz;
- fits/applies a continuous `RobustScaler`;
- extracts complete 25-sample windows without padding;
- applies per-event baseline over first 10 samples;
- clamps to `[-5, +5]`;
- returns `(time, channels)` tensors, shape `(25, 306)`, `float32`;
- returns provenance metadata: revision, configuration, channel names,
  sample indices, `zero_padded=False`.

Historical path in `scripts\prepare_real_subset.py` not replaced or modified.

Regression tests:

`tests\test_official_v1_preprocessing.py`

Tests cover:

- five fixed trial-2 event identities and timestamps;
- 306 channels and 25 samples;
- finite `float32` tensors;
- channel-order parity;
- strict comparison to the validated official fixture;
- first-event continuous context, no padding;
- historical sentence-scoped pipeline still present.

Validated parity tolerance:

- `rtol=0.0`
- `atol=1e-20`
- max absolute difference seen: ~`1.73e-23`
- RMSE: `0.0`

Documentation:

`docs\preprocessing.md`

Document separates:

- Current Historical Pipeline;
- Official-v1-Compatible Gated Pipeline.

States official-v1 validation covers trial 2 first five events only, not the
whole dataset.

Artifact:

`results\official_v1_preprocessing_variant.json`

## Authoritative current update — 2026-09-22

Sections below supersede stale earlier status statements in this historical
handoff.

Includes configuration, event/sample indices, parity metrics, test results,
scope limits.

## Test status

Baseline command:

```powershell
pytest tests -q
```

Latest result:

`13 passed`

Focused command:

```powershell
pytest tests/test_official_v1_preprocessing.py -q
```

Latest result:

`3 passed`

One expected MNE filename-convention warning because the raw file is named
`block1.fif`; not a test failure.

## Important implementation caveats

1. The API preprocesses the whole continuous recording on every `from_raw`
   call. Correct but expensive. Future optimization must keep exact numerical
   behavior and must not silently change scope.
2. `mne.io.read_raw_fif(...).pick("meg")` selects the 306 MEG channels. Keep
   channel ordering explicit and tested.
3. Event timestamps are absolute recording-timeline timestamps. Sample
   mapping:

   `round((event_time_seconds - 308.0) * 50)`

4. Event windows use:

   `window_start = event_sample - 10`

   `window_stop_exclusive = event_sample + 15`

5. No zero-padding. Out-of-bounds events must fail explicitly.
6. The stored parity fixture is the validated reference for the five fixed
   events only. Not evidence that all trials or the whole dataset are
   validated.
7. Official source has a Windows path parsing quirk (`str(file).split("/")`).
   Prior official extraction used an invocation-only path adapter; official
   source itself not edited.
8. The normal NeuroSelect environment has compatible MNE and sklearn, but it
   is not the external pinned Brain2Qwerty environment. Do not make the
   ordinary test suite depend on EXCA/neuralset unless explicitly requested.

## Current untracked artifacts

Worktree holds the research scripts/results in this document, including:

- `scripts\full_event_tensor_probe.py`
- `scripts\official_v1_compatibility_audit.py`
- `scripts\continuous_scope_event_parity.py`
- all `results\...` diagnostic artifacts;
- `src\neuroselect\official_v1_preprocessing.py`;
- `tests\test_official_v1_preprocessing.py`;
- `docs\preprocessing.md`.

Treat as intentional session work. Do not clean up or reset without user
direction.

## Recommended next scope

Next authorized experiment stays narrow and gated:

1. Use `NeuroSelectOfficialV1EventPreprocessing` explicitly.
2. Apply only to the existing 8 trials / existing processed data scope.
3. Verify event counts, valid windows, shapes, provenance before modeling.
4. Compare official-v1-compatible representation against the historical one.
5. Retrain or evaluate a decoder only if the user explicitly authorizes it in
   a new task.
6. Do not change the Conv1D+BiGRU+CTC model, event definitions, or official
   timing.

One sensible immediate validation: controlled preprocessing comparison across
the existing eight trials, no model training.

## Suggested skills for the incoming agent

- `diagnose` for any new preprocessing mismatch or regression.
- `tdd` when extending the gated preprocessing API or adding parity checks.
- `code-review` for reviewing the final diff before committing.
- `caveman-commit` only if the user later asks for a commit message or commit.
- `handoff` if another transfer is needed.

## Reference files

Use these instead of repeating prior investigations:

- Historical preprocessing: `scripts\prepare_real_subset.py`
- Existing data API: `src\neuroselect\data.py`
- Existing model: `src\neuroselect\models.py`
- New gated API: `src\neuroselect\official_v1_preprocessing.py`
- New tests: `tests\test_official_v1_preprocessing.py`
- Pipeline documentation: `docs\preprocessing.md`
- Continuous parity report: `results\continuous_scope_event_parity.json`
- Final gated-variant report: `results\official_v1_preprocessing_variant.json`
- Official parity fixture: `results\official_event_tensor_examples\trial_2_first_5.npz`
- Official environment report: `results\brain2qwerty_v1_environment.json`

## Current authoritative status

### Eight-trial official-v1 parity

The gated path reproduces official event tensors across all local development
trials 2–9:

- 240/240 exact hash matches;
- max MAE, RMSE, absolute error all 0;
- every event is `(25,306)` with 306 channels;
- historical preprocessing unchanged;
- official source, raw MEG, MAT/log files unchanged.

Primary artifact: `results\eight_trial_official_v1_parity.json`.

### Environment repair and regression tests

The original NeuroSelect environment is the workspace Python 3.13.7 x64
install:

`C:\Users\prana\AppData\Local\Programs\Python\Python313`

Its corrupted MINGW-W64 NumPy 1.26.4 install caused native import crashes.
Only NumPy was replaced, with CPython Windows wheel `numpy==2.2.6`; the
isolated official environment untouched.

Current regression command:

```powershell
$env:PYTHONPATH="src"
python -m pytest tests -q --basetemp=pytest-basetemp
```

Current result: **17 passed**. Explicit basetemp needed because the machine
default pytest temp directory is access-restricted.

Artifacts:

- `results\neuroselect_environment_before_numpy_repair.json`
- `results\neuroselect_environment_after_numpy_repair.json`
- `results\full_regression_test_report.json`

### Official-v1 event-sequence CTC baseline

The separate `official_v1_event_sequence` adapter concatenates official event
tensors:

```text
(25,306) × U → (25U,306)
```

Existing Conv1D+BiGRU+Linear+CTC model unchanged. Config: Adam `lr=0.001`,
gradient clipping `1.0`, 300 epochs, seeds `33/123/777`, train trials 2–7,
evaluation trials 8–9, greedy CTC decoding.

Results:

| Seed | Train CER | Evaluation CER |
|---:|---:|---:|
| 33 | 0.023 | 0.778 |
| 123 | 0.022 | 0.841 |
| 777 | 0.020 | 1.139 |
| Mean ± std | 0.0216 ± 0.0015 | 0.9194 ± 0.1573 |

Reading: **corrected representation improves fitting but not transfer**. Not
a preprocessing-only comparison, since the input representation differs from
the historical continuous-sentence one.

Artifacts:

- `scripts\official_v1_event_sequence_ctc_baseline.py`
- `results\official_v1_event_sequence_ctc_baseline.json`
- `results\figures\debug\official_v1_event_sequence_ctc_baseline.png`
- `results\figures\debug\official_v1_event_sequence_predictions.png`

### Timing feasibility audit

Official timestamps audited without training. Hypothetical rule was
`max(0, round(delta_t*50)-25)`.

| Split | Intervals | Mean Δt | Median Δt | Overlap fraction | Positive-gap fraction |
|---|---:|---:|---:|---:|---:|
| Train 2–7 | 173 | 0.1635 s | 0.1420 s | 100.0% | 0.0% |
| Evaluation 8–9 | 59 | 0.1910 s | 0.1660 s | 96.6% | 3.4% |

All training intervals have zero positive gaps; only two evaluation intervals
have positive gaps. Decision: **B — timing-gap experiment unlikely to be
informative**. Descriptively train/evaluation timing differs, but no
significance testing done. Timestamp gaps are oracle/offline information, not
automatically deployable neural input.

Artifacts:

- `scripts\inter_event_timing_audit.py`
- `results\inter_event_timing_audit.json`
- `results\figures\debug\inter_event_timing_distribution.png`
- `results\figures\debug\inter_event_timing_geometry.png`

### Exact official v1 model feasibility

Exact official v1 source inspected at the pinned revision. It uses:

- per-keystroke `SimpleConvTimeAgg`;
- 2D Fourier channel merger, 270 virtual channels;
- initial projection 512;
- 8 convolutional layers, hidden width 2048, kernel 3;
- GELU, batch normalization, dropout, skip connections, Bahdanau attention;
- one 2048-dimensional vector per keystroke;
- 4-layer, 2-head, 2048-dimensional sentence Transformer with ALiBi;
- per-keystroke linear character head and `CrossEntropyLoss`;
- no LLM in base v1 decoding.

The exact model could not be safely instantiated on the available CPU/laptop;
the parameter probe died before producing a count. No training done and no
reduced model called official v1. Decision: **D — inconclusive**.

Artifacts:

- `scripts\official_v1_style_feasibility_audit.py`
- `results\official_v1_style_baseline.json`
- `results\figures\debug\official_v1_style_baseline.png`
- `results\figures\debug\official_v1_style_predictions.png`

Only proposed next modeling scope is one explicitly labeled reduced
hierarchical approximation: event encoder → event vectors → sentence
Transformer → per-event classifier. Needs explicit authorization and must
document reduced dimensions.

### Current pushed commit history

- `71f5078` — eight-trial parity audit
- `4c12452` — four-event parity forensics
- `e4fb459` — official overlap/index correction
- `908b7c0` — NumPy repair and regression report
- `977da84` — event-sequence baseline script
- `23e8d8b` — event-sequence baseline results
- `2eb967f` — timing feasibility audit
- `8dadbb1` — expanded README
- `cf09788` — official-v1 architecture feasibility audit

All completed logical changes committed and pushed. Keep committing and
pushing verified logical change sets promptly.
