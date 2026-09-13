# Experiment log

## Phase 1 foundation

| Field | Value |
|---|---|
| Experiment ID | `phase1-foundation-20260913` |
| Code revision | Workspace is not currently a Git repository; commit is therefore `not available` |
| Seed | 33 in the reproducibility fixture |
| Dataset split | Synthetic train-only overfit fixture for pipeline validation |
| Subjects | Synthetic |
| Model | Conv1d frontend + bidirectional GRU + linear CTC head |
| Loss | PyTorch CTC loss, blank id 0, `zero_infinity=True` |
| Hardware | Local Windows Python environment; CPU sanity path |
| Public-data training time | Not run |
| Public-data validation/test metric | Not available |
| Checkpoint | Not produced for public data |

## Tiny overfit sanity check

| Field | Value |
|---|---|
| Experiment ID | `phase1-sanity-20260913` |
| Dataset | One synthetic `(time=20, channels=7)` example, target `ab` |
| Seed | 33 |
| Initial CTC loss | 29.33146095275879 |
| Final CTC loss | 0.0019628985319286585 |
| Overfit gate | Passed |
| Result file | `results/sanity_check.json` |

No result has been manually edited. Scientific CER/WER reporting begins only
after a manifest is generated from an accessible signal subset and the
leakage checks are completed.

## Phase 2 — SpanishBCBL real-data integration

- Official repository revision:
  `5f9889621d0df391c5aab37c996683d308e6e926`.
- Downloaded S22 session 1 block 1 MEG and matching behavioral log only.
- Official extraction succeeded; cleaned events are saved as
  `data/raw/spanishbcbl_s22/events_clean.pkl`.
- Eight production sentence windows were extracted at 50 Hz with 306 MEG
  channels.
- Smoke test: finite logits and CTC loss.
- Tiny overfit: loss `20.88 -> 2.93` over 40 CPU epochs on six trials, but
  predictions collapsed to spaces. Additional one-/two-trial investigations
  did not reproduce the sentences.
- Scientific conclusion: real-data CER/WER is not yet reportable; investigate
  alignment, preprocessing, and model learning before a baseline evaluation.

## Phase 3 — Real-data baseline forensics

- Added `scripts/real_baseline_forensics.py` and CTC geometry validation.
- Mapping reconstruction matched event-derived text, manifest text, and CTC
  targets for the first three trials; all eight trials have one retained
  keystroke per target character.
- All eight trials satisfy CTC geometry with `T/U` from `7.44` to `10.19`.
- Found and fixed a preprocessing mismatch: filtering was previously applied
  after cropping short trials, instead of to the continuous recording first.
- After the fix, one-trial loss was `24.05 -> 2.39`, but greedy output
  remained a space-like prediction.
- Normalization comparison, learning-rate comparison, random-target control,
  and artificial-target control did not yet demonstrate reliable sequence
  overfitting.
- Current classification: preprocessing mismatch is verified and fixed;
  remaining root cause is not yet established.
