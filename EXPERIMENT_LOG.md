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

## Phase 4B — CTC and model learnability isolation

- Decoder regression passed for blanks, adjacent repeats, spaces, `abc`, and
  `hello`.
- Direct trainable CTC logits recovered `abc` and `hello` exactly. This
  verifies the current target encoding, tensor dimensions, blank index, and
  CTC loss invocation.
- Exact `Conv1D + BiGRU + Linear + CTC` did not pass the easy `abc` gate:
  prediction `c`, CER `0.667`. It did recover `brain` exactly and produced
  partial outputs for `hello` and `hello world`.
- Noise sweep was inconsistent: clean `hello` CER `0.40`, noise `0.05`
  recovered `hello`, while higher noise levels failed. The 306-channel
  smooth realistic synthetic test produced an empty prediction.
- Shuffled control recovered arbitrary target `xyz` from signal encoding
  `abc`; this is a verified positional-memorization confound, not evidence
  of signal-based decoding.
- Fresh real trials produced partial outputs:
  trial 2 `la ta ees` (CER `0.70`), trial 3 `el motor consrgia` (CER
  `0.3704`). These are not reliable one-trial exact recoveries.
- Phase 4B status: `NOT YET ESTABLISHED`. Do not implement Evidence Selector.
- Full experiment details:
  `docs/CTC_LEARNABILITY.md`.

## Phase 4A — CTC decoder verification

- Tested exact SpanishBCBL `decode_ctc` implementation independently of MEG,
  model, training, and optimization.
- Verified ordinary characters (`abc`), repeated `l` characters (`hello`),
  spaces (`hi there`), blank-heavy paths (`ab`), repeated characters with a
  separating blank (`aa`), and repeated characters without a blank (`a`).
- Decoder behavior matches standard greedy CTC collapse rules.
- Result: `results/ctc_decoder_test.json`.
- Phase 4A status: `CTC DECODER VERIFIED`.

## Phase 4B — Direct CTC logit learnability

- Isolated CTC from MEG, neural signals, `Conv1D`, `BiGRU`, training data, and
  model parameters. Only logits were trainable.
- Used exact SpanishBCBL vocabulary, blank index `0`, `CTCLoss`, and greedy
  decoder. Log probabilities had shape `(T, N, C)`; targets had shape
  `(N, S)`.
- Tested `abc` (`T=30`), `hello` (`T=50`), `hi there` (`T=70`), and
  `la tasa` (`T=80`).
- Recorded initial/minimum/final loss, gradient norm, tensor geometry,
  iterations, decoded output, and CER in
  `results/ctc_direct_logits_test.json`.
- Phase 4B conclusion is determined by artifact results and post-training
  greedy decoding; no model architecture work is included.

## Phase 4C — Exact model synthetic overfit

- Tested the unchanged `Conv1D + BiGRU + Linear` model on a deterministic
  306-channel, `T=300` signal encoding `abc` in channels 0, 1, and 2.
- Recorded model/CTC tensor shapes, loss history, first gradient norm,
  parameter update norm, final greedy decoding, and CER in
  `results/synthetic_easy_overfit.json`.
- Saved the signal visualization at
  `results/figures/debug/synthetic_easy_signal.png`.
- Phase 4C conclusion is based only on exact final decoding and CER.

## Phase 4D — Synthetic noise robustness

- Reused the Phase 4C signal, model, CTC loss, decoder, seed, optimizer,
  learning rate, and 1000-epoch training setup.
- Tested Gaussian noise standard deviations `0.0`, `0.5`, `2.0`, and `5.0`.
  Clean signal standard deviation and realized noise standard deviation are
  recorded per condition.
- Recorded loss, decoded output, CER, and exact-recovery status in
  `results/synthetic_noise_sweep.json`.
- Saved the representative channel comparison at
  `results/figures/debug/synthetic_noise_levels.png`.
- All four tested conditions recovered `abc` exactly with CER `0.0`;
  final loss increased from `0.00162` at no noise to `0.01424` at high noise.
- This phase makes no claim about real MEG or neuroscience.

## Phase 4E — Real MEG one-trial recheck

- Reused the corrected continuous preprocessing manifest and unchanged
  `Conv1D + BiGRU + Linear + CTC` model.
- Trial 2 (`S22`, session 1, `block1`) was trained for 1000 epochs. One
  additional trial was run because Trial 2 showed loss reduction and partial
  decoding, satisfying the meaningful-learning continuation criterion.
- Recorded loss, CER, decoded output, decoded length, mean blank probability,
  blank-argmax fraction, and target length in
  `results/real_trial_recheck.json`.
- Saved comparison figure:
  `results/figures/debug/real_vs_synthetic_training.png`.
- No benchmark, architecture change, real-data generalization claim, or
  downstream NeuroSelect component was introduced.

## Phase 5A — Small real-data baseline

- Used all eight prepared SpanishBCBL trials from the same subject/session/
  block. Trials 2–7 were training; trials 8–9 were evaluation.
- Verified no identical sentence crossed the split.
- Used unchanged model and corrected continuous preprocessing. Configuration:
  seed `33`, Adam, learning rate `0.01`, 300 epochs.
- Per-trial CER/WER, aggregate metrics, blank statistics, and trivial
  references are stored in `results/small_real_baseline.json`.
- This is a small single-subject development-set baseline only, not a
  generalization benchmark.

## Phase 5B — Blank-collapse diagnostic

- Tested fixed train/evaluation split (trials 2–7 / 8–9), seed `33`, unchanged
  preprocessing, vocabulary, model, decoder, and 300 epochs.
- Matrix: baseline Adam `0.01`; Adam `0.001`; Adam `0.0001`; Adam `0.001`
  with max-norm `1.0`; and Adam `0.001` without clipping.
- Each configuration records train/evaluation CER, WER, blank probability,
  blank-argmax fraction, decoded-length ratio, per-trial outputs, and finite
  metric checks in `results/blank_collapse_diagnostic.json`.
- Comparison figure:
  `results/figures/debug/blank_collapse_diagnostic.png`.
- Baseline matrix remained blank-dominated on both splits (train CER `0.755`,
  eval CER `0.944`; train/eval blank-argmax fractions `0.958`/`0.991`).
- Lowering LR to `0.0001` worsened training to all-empty decoding. LR
  `0.001` reduced train/eval CER modestly but remained mostly blank.
- Gradient clipping at `1.0` materially changed behavior: train CER `0.175`
  and blank fraction `0.838`, but eval CER worsened to `1.109` with long
  nonsensical outputs and blank fraction `0.662`. This supports an
  optimization/generalization tradeoff, not a standalone explanation.
- No-clipping `0.001` matched the unclipped `0.001` run under this fixed
  seed. All reported losses and metrics were finite. The matrix supports
  optimization as a major factor and shows train/evaluation divergence under
  clipping, but does not identify a unique root cause.

## Phase 5C — Seed stability of clipped configuration

- Ran exactly seeds `33`, `123`, and `777` with fixed train/evaluation split,
  unchanged preprocessing, vocabulary, decoder, and architecture.
- Configuration: Adam, learning rate `0.001`, gradient clipping max norm
  `1.0`, 300 epochs.
- Recorded per-seed train/evaluation CER, WER, blank-argmax fractions,
  decoded-length ratios, per-trial evaluation outputs, CER gaps, and finite
  checks in `results/seed_stability_diagnostic.json`.
- Saved comparison figure:
  `results/figures/debug/seed_stability_diagnostic.png`.

## Phase 5D — Trial distribution and temporal statistics audit

- Audited all eight existing processed `.npy` trials without retraining or
  changing preprocessing.
- Compared train trials 2–7 with evaluation trials 8–9 across geometry,
  global signal, temporal difference, energy, channel, broad-band spectral,
  and lightweight pooled-distribution distance statistics.
- Evaluation trials were longer on average (`T=287.5` vs `241.0`,
  duration `5.74 s` vs `4.82 s`, `T/U=9.50` vs `8.07`) and had higher mean
  channel standard deviation (`0.629` vs `0.465`). Broad-band relative power
  also differed descriptively, especially 0.1–4 Hz and 4–20 Hz bands.
- These are descriptive results from only two evaluation trials; no
  significance testing or causal explanation was made.
