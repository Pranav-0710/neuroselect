# Experiment log

## Phase 1 foundation

| Field | Value |
|---|---|
| Experiment ID | `phase1-foundation-20260913` |
| Code revision | Workspace not git repo then, so commit `not available` |
| Seed | 33 in reproducibility fixture |
| Dataset split | Synthetic train-only overfit fixture, pipeline validation |
| Subjects | Synthetic |
| Model | Conv1d frontend + bidirectional GRU + linear CTC head |
| Loss | PyTorch CTC loss, blank id 0, `zero_infinity=True` |
| Hardware | Local Windows Python, CPU sanity path |
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

No result hand-edited. CER/WER reporting starts only after manifest built
from accessible signal subset and leakage checks done.

## Phase 2 — SpanishBCBL real-data integration

- Official repo revision:
  `5f9889621d0df391c5aab37c996683d308e6e926`.
- Downloaded S22 session 1 block 1 MEG + matching behavioral log only.
- Official extraction worked; cleaned events saved as
  `data/raw/spanishbcbl_s22/events_clean.pkl`.
- Eight production sentence windows extracted at 50 Hz, 306 MEG channels.
- Smoke test: finite logits and CTC loss.
- Tiny overfit: loss `20.88 -> 2.93` over 40 CPU epochs on six trials, but
  predictions collapsed to spaces. One-/two-trial follow-ups did not
  reproduce sentences.
- Conclusion: real-data CER/WER not reportable yet; check alignment,
  preprocessing, model learning before baseline evaluation.

## Phase 3 — Real-data baseline forensics

- Added `scripts/real_baseline_forensics.py` + CTC geometry validation.
- Mapping reconstruction matched event text, manifest text, CTC targets for
  first three trials; all eight trials have one retained keystroke per
  target character.
- All eight trials satisfy CTC geometry, `T/U` from `7.44` to `10.19`.
- Found + fixed preprocessing mismatch: filtering ran after cropping short
  trials instead of on continuous recording first.
- After fix, one-trial loss `24.05 -> 2.39`, but greedy output stayed
  space-like.
- Normalization comparison, learning-rate comparison, random-target control,
  artificial-target control: no reliable sequence overfitting yet.
- Status: preprocessing mismatch verified + fixed; root cause not
  established.

## Phase 4B — CTC and model learnability isolation

- Decoder regression passed for blanks, adjacent repeats, spaces, `abc`, and
  `hello`.
- Direct trainable CTC logits recovered `abc` and `hello` exactly. Verifies
  target encoding, tensor dimensions, blank index, CTC loss invocation.
- Exact `Conv1D + BiGRU + Linear + CTC` failed easy `abc` gate: prediction
  `c`, CER `0.667`. Recovered `brain` exactly, partial output for `hello`
  and `hello world`.
- Noise sweep inconsistent: clean `hello` CER `0.40`, noise `0.05` recovered
  `hello`, higher noise failed. 306-channel smooth realistic synthetic test
  gave empty prediction.
- Shuffled control recovered arbitrary target `xyz` from signal encoding
  `abc`: verified positional-memorization confound, not signal-based
  decoding.
- Fresh real trials gave partial output: trial 2 `la ta ees` (CER `0.70`),
  trial 3 `el motor consrgia` (CER `0.3704`). Not reliable one-trial exact
  recoveries.
- Phase 4B status: `NOT YET ESTABLISHED`. Do not implement Evidence Selector.
- Details: `docs/CTC_LEARNABILITY.md`.

## Phase 4A — CTC decoder verification

- Tested exact SpanishBCBL `decode_ctc` independent of MEG, model, training,
  optimization.
- Verified ordinary characters (`abc`), repeated `l` characters (`hello`),
  spaces (`hi there`), blank-heavy paths (`ab`), repeats with separating
  blank (`aa`), repeats without blank (`a`).
- Decoder matches standard greedy CTC collapse rules.
- Result: `results/ctc_decoder_test.json`.
- Phase 4A status: `CTC DECODER VERIFIED`.

## Phase 4B — Direct CTC logit learnability

- Isolated CTC from MEG, neural signals, `Conv1D`, `BiGRU`, training data,
  model parameters. Only logits trainable.
- Used exact SpanishBCBL vocabulary, blank index `0`, `CTCLoss`, greedy
  decoder. Log probabilities shape `(T, N, C)`; targets shape `(N, S)`.
- Tested `abc` (`T=30`), `hello` (`T=50`), `hi there` (`T=70`), and
  `la tasa` (`T=80`).
- Recorded initial/minimum/final loss, gradient norm, tensor geometry,
  iterations, decoded output, CER in
  `results/ctc_direct_logits_test.json`.
- Phase 4B conclusion comes from artifact results + post-training greedy
  decoding; no architecture work included.

## Phase 4C — Exact model synthetic overfit

- Tested unchanged `Conv1D + BiGRU + Linear` model on deterministic
  306-channel, `T=300` signal encoding `abc` in channels 0, 1, and 2.
- Recorded model/CTC tensor shapes, loss history, first gradient norm,
  parameter update norm, final greedy decoding, CER in
  `results/synthetic_easy_overfit.json`.
- Signal visualization:
  `results/figures/debug/synthetic_easy_signal.png`.
- Phase 4C conclusion uses only exact final decoding and CER.

## Phase 4D — Synthetic noise robustness

- Reused Phase 4C signal, model, CTC loss, decoder, seed, optimizer,
  learning rate, 1000-epoch setup.
- Tested Gaussian noise standard deviations `0.0`, `0.5`, `2.0`, and `5.0`.
  Clean signal std + realized noise std recorded per condition.
- Recorded loss, decoded output, CER, exact-recovery status in
  `results/synthetic_noise_sweep.json`.
- Channel comparison:
  `results/figures/debug/synthetic_noise_levels.png`.
- All four conditions recovered `abc` exactly with CER `0.0`; final loss rose
  from `0.00162` at no noise to `0.01424` at high noise.
- No claim about real MEG or neuroscience.

## Phase 4E — Real MEG one-trial recheck

- Reused corrected continuous preprocessing manifest + unchanged
  `Conv1D + BiGRU + Linear + CTC` model.
- Trial 2 (`S22`, session 1, `block1`) trained 1000 epochs. One extra trial
  run because trial 2 showed loss reduction + partial decoding, meeting the
  meaningful-learning continuation criterion.
- Recorded loss, CER, decoded output, decoded length, mean blank probability,
  blank-argmax fraction, target length in
  `results/real_trial_recheck.json`.
- Figure:
  `results/figures/debug/real_vs_synthetic_training.png`.
- No benchmark, architecture change, generalization claim, or downstream
  NeuroSelect component added.

## Phase 5A — Small real-data baseline

- Used all eight prepared SpanishBCBL trials, same subject/session/block.
  Trials 2–7 train, trials 8–9 evaluation.
- Verified no identical sentence crossed the split.
- Unchanged model + corrected continuous preprocessing. Config: seed `33`,
  Adam, learning rate `0.01`, 300 epochs.
- Per-trial CER/WER, aggregate metrics, blank statistics, trivial references
  in `results/small_real_baseline.json`.
- Small single-subject development-set baseline only, not a generalization
  benchmark.

## Phase 5B — Blank-collapse diagnostic

- Fixed train/evaluation split (trials 2–7 / 8–9), seed `33`, unchanged
  preprocessing, vocabulary, model, decoder, 300 epochs.
- Matrix: baseline Adam `0.01`; Adam `0.001`; Adam `0.0001`; Adam `0.001`
  with max-norm `1.0`; and Adam `0.001` without clipping.
- Each config records train/evaluation CER, WER, blank probability,
  blank-argmax fraction, decoded-length ratio, per-trial outputs, finite
  metric checks in `results/blank_collapse_diagnostic.json`.
- Figure:
  `results/figures/debug/blank_collapse_diagnostic.png`.
- Baseline matrix stayed blank-dominated on both splits (train CER `0.755`,
  eval CER `0.944`; train/eval blank-argmax fractions `0.958`/`0.991`).
- LR `0.0001` worsened training to all-empty decoding. LR `0.001` cut
  train/eval CER modestly but stayed mostly blank.
- Gradient clipping at `1.0` changed behavior a lot: train CER `0.175` and
  blank fraction `0.838`, but eval CER worsened to `1.109` with long
  nonsense output and blank fraction `0.662`. Supports an
  optimization/generalization tradeoff, not a standalone explanation.
- No-clipping `0.001` matched the unclipped `0.001` run under this fixed
  seed. All losses and metrics finite. Matrix supports optimization as major
  factor and shows train/evaluation divergence under clipping, but no unique
  root cause.

## Phase 5C — Seed stability of clipped configuration

- Ran exactly seeds `33`, `123`, and `777` with fixed train/evaluation split,
  unchanged preprocessing, vocabulary, decoder, architecture.
- Config: Adam, learning rate `0.001`, gradient clipping max norm `1.0`,
  300 epochs.
- Recorded per-seed train/evaluation CER, WER, blank-argmax fractions,
  decoded-length ratios, per-trial evaluation outputs, CER gaps, finite
  checks in `results/seed_stability_diagnostic.json`.
- Figure:
  `results/figures/debug/seed_stability_diagnostic.png`.

## Phase 5D — Trial distribution and temporal statistics audit

- Audited all eight processed `.npy` trials, no retraining, no preprocessing
  change.
- Compared train trials 2–7 with evaluation trials 8–9 on geometry, global
  signal, temporal difference, energy, channel, broad-band spectral, and
  lightweight pooled-distribution distance statistics.
- Evaluation trials longer on average (`T=287.5` vs `241.0`, duration
  `5.74 s` vs `4.82 s`, `T/U=9.50` vs `8.07`) with higher mean channel
  standard deviation (`0.629` vs `0.465`). Broad-band relative power differed
  descriptively, mostly 0.1–4 Hz and 4–20 Hz bands.
- Descriptive only, two evaluation trials; no significance test, no causal
  explanation.

## Phase 5E — Leave-one-trial-out generalization

- Eight folds, each trial held out once. Every fold used seed `33`, Adam,
  learning rate `0.001`, gradient clipping max norm `1.0`, and 300 epochs
  with unchanged preprocessing, model, vocabulary, decoder.
- Mean held-out CER `0.917`; median `0.864`; mean WER `1.200`.
- Only trial 7 got held-out CER below `0.75`. Trials 2–6, 8, and 9 were
  poor-transfer cases under the descriptive threshold.
- Trial 8 CER `0.824`, trial 9 CER `0.852`: not uniquely worse than most
  other held-out trials. Trial 3 worst at `1.185`.
- 5D geometry + distribution statistics attached in
  `results/leave_one_trial_out.json`. No significance tests, no causal
  claims.

## Phase 5F — Event-aligned character-information audit

- Reused `events_clean.pkl` + existing processed trial tensors. Extracted
  exactly 25 samples per retained keystroke from `event_time-0.2s` through
  `event_time+0.3s` at 50 Hz, boundary padding only where needed.
- Audited 240 events across 19 observed character classes. Five classes had
  under three examples (`b`, `g`, `u`, `v`, `x`), limiting class-wise
  reading.
- Group-aware leave-one-trial-out linear classification sat only slightly
  above majority accuracy with the best feature set (mean/std features):
  accuracy `0.142`, macro-F1 `0.080`, balanced accuracy `0.114`; majority
  accuracy was `0.129`.
- Within-trial chronological classification weak (accuracy `0.051`, balanced
  accuracy `0.056`), not above the deterministic shuffled label control
  (accuracy `0.053`, balanced accuracy `0.044`).
- Artifact + figures:
  `results/event_information_audit.json`,
  `results/figures/debug/event_label_distribution.png`,
  `results/figures/debug/event_window_examples.png`,
  `results/figures/debug/event_classification_results.png`.
- Reading: weak or no detectable event-level character information under
  these simple features and this small dataset; no sentence-level CTC
  retraining performed.

## Phase 5G — Event-centered versus temporal-control windows

- Compared paired event-centered windows `[-0.2s,+0.3s]` with control windows
  `[-1.2s,-0.7s]`, both 25 samples at 50 Hz, identical events, labels,
  features, classifier, folds, chronological within-trial split.
- Boundary filtering left 168 paired events from 240; 72 events excluded from
  both conditions. Both used exactly the same paired events.
- Event-centered cross-trial performance below control: accuracy `0.089` vs
  `0.118`, macro-F1 `0.041` vs `0.054`, balanced accuracy `0.076` vs
  `0.085`.
- Event-centered within-trial also below control: accuracy `0.076` vs
  `0.097`, macro-F1 `0.034` vs `0.054`, balanced accuracy `0.088` vs
  `0.108`.
- Differences mixed across held-out trials. No clear event-timing advantage;
  does not establish absence of MEG character information.

## Phase 5V — Target-permutation negative control

- Reused the committed 5U official-v1 event-sequence CTC code unchanged.
  Training signals untouched; training targets deranged once with shuffle
  seed `2026` (signal trial → target trial:
  `2→3, 3→6, 4→2, 5→7, 6→4, 7→5`). Evaluation trials 8–9 stayed correctly
  paired. Same permutation for model seeds `33`, `123`, `777`.
- Control memorized mismatched pairs nearly as well as real labels: train CER
  `0.043 ± 0.007` vs `0.022`.
- Held-out CER, control vs real: seed 33 `1.356` vs `0.778`; seed 123
  `0.842` vs `0.841`; seed 777 `1.047` vs `1.139`. Mean `1.082 ± 0.212` vs
  `0.919 ± 0.157` (difference `+0.162`, from seed 33 alone).
- Where real and control seeds hit the same blank regime, CERs matched: eval
  CER tracks blank regime, not label correctness.
- Interpretation: **B — real-label and no-signal performance are similar**.
  Two evaluation trials only; no significance claim; not proof that MEG
  character information is absent.
- Artifacts: `results/no_signal_target_permutation.json`,
  `results/figures/debug/no_signal_target_permutation.png`,
  `results/figures/debug/no_signal_control_outputs.png`.

## Phase 5W — S22 data-scale inventory (metadata only)

- Listed the SpanishBCBL Hugging Face repository at revision
  `88f9096c6ce3a3fb17cc7b8e3131ff7f96da5684`, no downloads.
- S22 has two recordings: `22_9788/231214` (session 1) and `22_9788/231222`
  (session 2), each with two typing blocks plus a tapping localizer.
- Local `block1.fif` + MAT log match the remote LFS SHA-256 exactly.
- Three typing blocks missing locally: 4.11 GB of FIF + MAT.
- Artifacts: `results/s22_data_scale_inventory.json`,
  `results/figures/debug/s22_data_scale_inventory.png`.

## Phase 5X — Targeted S22 acquisition and sentence-overlap audit

- Downloaded exactly three typing FIFs and three MAT logs at the pinned
  revision; all six SHA-256 and size checks passed
  (`4,108,645,358` bytes). No tapping downloaded; existing
  `block1.fif`/MAT untouched.
- Official extraction over all four blocks gave `12,952` events: `9,650`
  keystrokes, `3,042` words, `256` production sentences, trials `2–65` per
  block.

  | Block | Events | Keystrokes | Words | Sentences |
  |---|---:|---:|---:|---:|
  | session 1 / block1 (list1) | 2,743 | 2,040 | 638 | 64 |
  | session 1 / block2 (list2) | 3,700 | 2,755 | 880 | 64 |
  | session 2 / block1 (list2) | 3,741 | 2,792 | 884 | 64 |
  | session 2 / block2 (list1) | 2,768 | 2,063 | 640 | 64 |

- Block-1 rows reproduce the historical `events_clean.pkl` exactly
  (2,743/2,743 row match); only pandas-version string formatting of
  `sentence_UID`/`button_unique_id` differs. The handoff
  `2918/2119/666/132` figures were raw pre-cleaning counts.
- Sentence-list overlap now **verified, not inferred**: list1 blocks
  (s1b1, s2b2) share 64/64 sentences, list2 blocks (s1b2, s2b1) share 64/64,
  all cross-list pairs share 0. S22 has `128` unique sentences, each typed
  twice, once per session.
- Consequence: cross-session splits are not sentence-disjoint by default.
- No decoder, classifier or training run. Artifacts:
  `results/s22_acquisition.json`,
  `results/s22_official_event_extraction.json`,
  `results/figures/debug/s22_sentence_overlap.png`,
  `data/raw/spanishbcbl_s22/events_clean_all_blocks.pkl` (untracked data).

## Phase 5Y — Split-ladder design and leakage audit (metadata only)

- Wrote pandas-independent sentence manifest
  (`data/manifests/s22_sentence_block_metadata.jsonl`, 256 records).
- Grouping uses the **exact presented stimulus text**. Typed text cannot
  group sentences across sessions: 256 records hold only 195 unique typed
  strings (134 once, 61 twice) because typing errors differ per session,
  while presented text gives exactly 128 texts × 2.
- List membership derived from exact sentence-text set equality between
  blocks, then cross-checked against log filenames; both agree.
- Word events include RSVP presentation words, so produced words
  (`is_percep == False`, 1,518) counted separately from perceptual words
  (1,524).

  | Block | List | Sentences | Keystrokes | Typed words | Mean target chars |
  |---|---|---:|---:|---:|---:|
  | session1/block1 | list1 | 64 | 2,040 | 318 | 31.9 |
  | session1/block2 | list2 | 64 | 2,755 | 438 | 43.0 |
  | session2/block1 | list2 | 64 | 2,792 | 442 | 43.6 |
  | session2/block2 | list1 | 64 | 2,063 | 320 | 32.2 |

- Splits A–E sentence-disjoint (text, group, typed-string and UID overlap all
  zero). F and G explicitly labelled CROSS-SESSION WITH SENTENCE OVERLAP
  (64/64 texts shared; 35 and 26 typed strings identical).
- `trial_id` is block-scoped (2–65 per block), so raw trial numbers coincide
  across blocks by construction; the block-scoped
  `(session, block, trial)` overlap is zero for every split.
- Overall status: PASS. No training, no preprocessing change, no downloads.
- Artifacts: `results/s22_split_ladder_audit.json`,
  `results/figures/debug/s22_split_ladder.png`,
  `data/manifests/split_[A-G]_*.json`.

## Phase 5Z — Expanded-data baseline (all four S22 blocks)

- Scale: 240 → `9,650` keystrokes (40.2×), 8 → `256` sentences (`128`
  unique texts, each typed once per session), 1 → 4 blocks, 1 → 2 sessions.
  Keystrokes per sentence `37.7 ± 7.9` (13–64); 26 of 29 classes occur, space
  `13.0%`.
- **Official-v1 output placement completed.** 469 keystrokes land exactly on a
  half-sample; for `34` (`0.35%`) fp rounding makes `TimedArray.overlap` return
  24 of 25 samples. The official `MegExtractor` adds that into a zero array of
  the requested duration (`neuralset.base.TimedArray.__iadd__`), leaving one
  zero sample at position 0 (17) or 24 (17). The gated module raised instead;
  it now places the overlap the same way. Gated overlap arithmetic was already
  bit-exact with `_overlap_slice`, so no previously handled event changed
  (pytest trial-2 reference hashes unchanged).
- Parity chain, all exact (max abs difference `0.0`):
  boundary + control events vs official `MegExtractor`, 66/66;
  stored slab (workspace env) vs gated module in pinned env, 869/869 across
  all four blocks. The workspace env differs from the pinned one (scipy 1.15.2,
  scikit-learn 1.5.2, pandas 3.0.1, torch 2.13.0) yet reproduces it exactly.
- Tensors: `(9650, 25, 306)` float32 memmap + JSONL index; splits are row
  views, never copies. All finite; every sentence's labels reconstruct its
  typed text exactly (`<space>`→` `, `<special>`→`@`).
- **Event probe** (StandardScaler → PCA 95% → balanced logistic regression,
  fit on training side only). Train accuracy `1.000` on every split
  (~900–1,330 PCA components on 2–4k events: memorization). Held-out balanced
  accuracy vs chance `0.0385`:

  | Split | Bal. acc | × chance | Macro F1 | Pred. classes |
  |---|---:|---:|---:|---:|
  | C (primary) | 0.0378 | 0.98 | 0.010 | 3 |
  | A | 0.0399 | 1.04 | 0.013 | 4 |
  | B | 0.0346 | 0.90 | 0.017 | 5 |
  | D | 0.0637 | 1.66 | 0.051 | 21 |
  | E | 0.0349 | 0.91 | 0.021 | 11 |

  Raw accuracy sits below the majority baseline everywhere; expected with
  `class_weight="balanced"`, so balanced accuracy vs chance is the reference.
- **CTC**, unchanged ConvCTC, Adam 1e-3, clip 1.0, 300 epochs, seeds
  33/123/777. Batches group **identical-length** sentences only: no padding,
  forward passes identical to one-at-a-time; gradients averaged per group
  (22 updates/epoch on C). Only protocol difference from the 8-trial runs.
- Primary metric is aligned-character F1. CER is length-sensitive: across all
  12 trained runs, Pearson r(CER, decoded/target length) = `+0.962`,
  r(insertion rate, length) = `+0.984`, r(deletion rate, length) = `−0.975`,
  but r(character F1, length) = `+0.138`. CER spans 0.778–1.097 across runs;
  F1 spans 0.297–0.319.

  | Condition | Train CER | Held-out CER | Char F1 (pooled) | F1 per seed | Len ratio |
  |---|---:|---:|---:|---|---:|
  | C real (primary) | 0.022 | 0.877 ± 0.033 | **0.313** | .318 .302 .319 | 1.008 |
  | C control | 0.020 | 0.892 ± 0.157 | **0.305** | .313 .299 .304 | 0.942 |
  | D real | 0.008 | 0.863 ± 0.092 | **0.309** | .307 .306 .315 | 0.914 |
  | E real | 0.007 | 0.922 ± 0.112 | **0.298** | .298 .297 .300 | 0.993 |
  | Text anchor (C) | — | — | **0.345** | — | 0.953 (pooled) |

  CER and length ratio are means of per-sentence values (same CER definition
  as 5U's `0.9194`); char F1 is pooled over all held-out characters.
- Every run memorizes its training set (train F1 0.986–0.996), including the
  control, which reproduces deliberately mismatched targets verbatim.
- **Control (split C):** real above control in 3/3 seed pairs (same init and
  batch order; only targets differ) by `+0.003` to `+0.016` F1 — no larger
  than within-condition seed spread (0.017, 0.014). CER direction flips by
  seed (control lower for 33 and 123, higher for 777): the 2-seed CER reading
  did not survive the third seed. Pooled insertion rates equal (0.18 vs 0.18).
- Control derangement is over sentence UIDs; in split C each sentence occurs
  twice, so 3/128 pairs (2.3%; ~1 expected) map a signal to its own sentence
  typed in the other session (2 text-identical). That dilutes the control
  toward the real condition, so it could only hide an advantage. Stimulus-level
  derangement would exclude it.
- **Data scale (Step 7, char F1):** 8-trial real `0.270`, 8-trial control
  `0.274`; expanded C real `0.313`, C control `0.305`. F1 rose with data in
  **both** conditions by similar amounts, so the gain is not attributable to
  valid signal–text correspondence. 8-trial F1 rests on 2 held-out sentences
  per seed.
- Cross-session vs same-session cannot be contrasted for CTC: D and E are both
  cross-session, and same-session A/B were not run for CTC. The probe's A/B
  show no same-session advantage.
- Conclusion: **under this protocol, the correctly paired neural condition
  does not yet demonstrate an advantage over the no-valid-signal-assignment
  control on length-robust character F1.** A length-matched training-sentence
  textual anchor achieved higher character F1 than the neural decoder on this
  development set. One subject, four blocks, three seeds, one control
  partition; not evidence that neural information is absent.
- Integrity: all 8 raw FIF/MAT files re-hashed and match the acquisition
  record; official source, historical preprocessing (`scripts/prepare_real_subset.py`)
  and tests unchanged; no download, LLM, Evidence Selector, architecture,
  decoder or vocabulary change. pytest `17 passed`.
- Runtime: 12 runs, ~0.6–3.2 h each at concurrency 5 (12 at once exceeded RAM
  and paged); queue is resumable.
- Artifacts: `results/s22_expanded_baseline.json` (phase summary),
  `results/s22_expanded_event_tensors.json`,
  `results/official_v1_boundary_event_parity.json`,
  `results/s22_event_tensor_env_parity.json`,
  `results/s22_expanded_event_probe.json`,
  `results/s22_expanded_ctc_baseline.json`,
  `results/s22_expanded_no_signal_control.json`,
  `results/runs/s22_expanded_ctc/*.json`,
  `results/figures/debug/s22_expanded_{baseline_summary,event_probe,ctc}.png`,
  `results/figures/debug/s22_no_signal_control.png`.

## Phase 6 — Official Brain2Qwerty v1 keystroke-level baseline

- **Outcome D: official baseline cannot be executed reliably on this
  hardware.** Nothing was trained.
- **Formulation (audited from source).** Per-keystroke classification, not
  CTC. Each `(306, 25)` keystroke window → per-subject channel merger
  (306 → 270 virtual channels, Fourier-embedded 2D sensor layout) → 1×1 conv
  270→512 → per-subject `SubjectLayers` 512→2048 → 8 × Conv1d 2048→2048 (k=3,
  dilations 1,2,4 repeating, BatchNorm, GELU, dropout 0.5, LayerScale-0.1 skips)
  → Bahdanau attention over time → one 2048-d vector. Keystrokes grouped by
  sentence *within a batch* → x-transformers Encoder (4 layers, 2 heads, dim
  2048) → `Linear(2048, 29)`, cross-entropy per keystroke, argmax decoding.
  **One character per keystroke: the decoder is given the segmentation, so v1
  CER is not comparable with CTC CER.**
- Official schedule: AdamW lr 5e-5, wd 1e-4, OneCycleLR, no clipping, 300
  epochs, batch 64 keystrokes, early stopping on val_CER (patience 30); the
  unshuffled sentence-order sampler splits sentences across batch boundaries;
  `trainer.test` uses final weights; the official `CER` metric is per batch,
  not per sentence; the merger's `usage_penalty` is never added to the loss.
- Corrections to the `c13a6e8` audit: the transformer uses **ScaleNorm** and
  **rotary + ALiBi** (class defaults, since the config sets only ALiBi, depth
  and heads). `subject_layers_config={}` parses to an *enabled* SubjectLayers.
- **Parameters: 623,548,457**, of which 320,716,800 are 200-slot per-subject
  tables (merger heads `(200, 270, 2048)`, subject layers `(200, 512, 2048)`);
  S22 reaches one slot (304.4M reachable).
- **The pinned env was incomplete for v1 training**: it lacked the official
  lock's `x-transformers==2.4.9`, `einops`, `einx`, `loguru`, `frozendict`,
  `Levenshtein`, `RapidFuzz`. Installed `--no-deps` from hashed wheels into a
  separate overlay (`~/Envs/phase6-official-overlay`), plus Windows shim
  `win32-setctime`; pinned env verified unchanged (`pip freeze` identical).
- Windows deviation: the official study's exca event cache names files with the
  timeline as JSON, which Windows rejects; events are rebuilt uncached and
  verified identical to the stored 5X extraction (9,650/9,650 keystrokes).
- **Smoke test PASS** (forward only, 172 keystrokes, one sentence per block):
  slab-backed batches equal the official `SegmentDataset` batches exactly in
  all four inputs (`neuro (B,306,25) f32`, `feature (B,1) i64`,
  `subject_id = 0`, `channel_positions (306,2)` in [0,1], none masked,
  identical across blocks). Finite logits and loss (untrained CE 3.53–3.61 vs
  ln 29 = 3.37); decoding runs.
- **Feasibility (measured):** training holds 4 fp32 copies of every parameter
  = 9.29 GiB static, ~12.0 GiB for the process, against 3.1 GiB of available
  commit with the model loaded (15.6 GiB RAM). Forward 48–66 ms/keystroke →
  ~14 min/epoch on C; official 300 epochs = 71/35/48 h (C/D/E); 100-epoch
  diagnostic = 24/12/16 h (12 runs ≈ 9 days); 2 h fits 8/16/12 epochs. No GPU-only
  operations. Reduced epochs do not reduce memory, and architecture changes
  were not allowed, so no lightweight diagnostic was run.
- Not executed: Part 4 baseline, Part 5 control, Part 6 comparison, Part 7
  signal ablation (ablating an untrained network would only show that random
  weights respond to input). `phase6_signal_ablation.png` deliberately not
  created.
- **Text anchor** (length-matched training sentence, no model): character F1
  C `0.345`, D `0.356`, E `0.345`; reproduces 5Z.
- **Leakage audit.** Exact text, typed text, UID and stimulus-group overlap all
  zero for C/D/E. **The official paraphrase rule fails:** 32 of 64 list2
  sentences are a list1 sentence plus one modifier (TF-IDF cosine up to
  `0.875`, e.g. `la investigacion recoge los conocimientos` →
  `la investigacion real recoge los conocimientos`). The official splitter
  (cosine > 0.5) would keep these together. Splits A–E are exact-text disjoint,
  **not paraphrase-disjoint**. This can only inflate held-out scores, so the 5Z
  null stands, but it confounds any future positive result on these splits.
- Integrity: official source, historical preprocessing, official-v1
  preprocessing and tests unchanged; earlier results unmodified; all 8 raw
  files re-hashed and match; event pickles untouched; no dataset download; no
  LLM, no Evidence Selector. pytest `17 passed`.
- Artifacts: `results/phase6_master_report.json`,
  `results/phase6_official_v1_model_audit.json`,
  `results/phase6_official_v1_smoke_test.json`,
  `results/phase6_official_v1_baseline.json`,
  `results/phase6_official_v1_no_signal_control.json`,
  `results/phase6_signal_dependence.json`, `results/phase6_text_anchor.json`,
  `results/phase6_leakage_audit.json`,
  `results/figures/debug/phase6_{real_vs_control,feasibility,outputs}.png`.

## Phase 7 — Official paraphrase-disjoint splits and exact v1 GPU harness

- **Classification A: clean official split reproduced and GPU harness ready**
  ("ready" = built and dry-run verified end to end on CPU with no optimisation
  step; the GPU preflight was **not executed**: no CUDA device). Nothing was
  trained. No decoding conclusion is drawn.
- **Official splitter reproduced exactly.** The configured official
  `Brain2QwertyV1Splitter` (seed 1, threshold 0.5, 80/10/10 in keystrokes) was
  run unmodified in the official order (study → SpanishBCBLPreprocessing →
  splitter). Its clusters were captured by temporarily swapping the `random`
  reference seen by the official module for a forwarding recorder; a verbatim
  re-implementation reproduces both the captured clusters and the assignment.
  Normalisation: none (raw presented text); TF-IDF: sklearn defaults; clusters:
  connected components of cosine > 0.5; `random.shuffle` with seed 1; first-fit
  allocation, overflow to test. Scope: the published run splits all
  participants; here the universe is S22's 128 texts.
- **Clusters:** 96 = 64 singletons + 32 pairs; every pair joins a list1 and a
  list2 text (cosine 0.51–0.875); no within-list pairs, no transitive merges.
- **Official 80/10/10 (S22):** train 101 groups / 202 records / 7,708
  keystrokes / 75 clusters; val 13 / 26 / 959 / 10; test 14 / 28 / 983 / 11.
  Both occurrences of every group sit in one partition. Zero cross-partition
  overlap in UIDs, groups, clusters, presented and typed text; max cross-partition
  cosine 0.30. Residual: TF-IDF does not catch inflectional variants (train
  `las estadisticas robustas siguen la distribucion` vs test
  `la estadistica sigue la distribucion`, cosine 0.29), clean by the official rule.
- **D-clean / E-clean:** the literal definitions are **infeasible** without
  breaking clusters (32 clusters span train and test in each). Resolution:
  clusters with a training-side member stay on the training side; their
  session-2 occurrences are excluded from test. D-clean: train 57 / val 7 /
  test 32 sentences (1,821 / 219 / 1,363 keystrokes); E-clean: 58 / 6 / 32
  (2,458 / 297 / 1,010). Validation is carved from training clusters (official
  seed and first-fit at 80:10; a research choice). Both cluster-disjoint.
- **Harness** (`scripts/phase7_official_v1_train.py`): official
  `experiment_config`, `Data.build`, `_build_modules`, `BrainModule`,
  `materialize_lazy_params`, `_trainer_setup`, `fit` then `test` on final
  weights; only the splitter is replaced by a manifest transform. Logs git
  commit, packages vs the official lock, parameter count, config, split and
  dataset hashes, runtime, GPU and CUDA. Refuses to train without CUDA.
  Six dry runs (3 splits × real/control) pass: 8 raw hashes verified, loaders
  equal manifests, 623,548,457 parameters, 300 epochs, EarlyStopping val_CER
  patience 30, no clipping, `Experiment.run` source hash matches the audit.
  Run matrix: 18 runs + 9 signal-dependence evaluations.
- **Control:** stimulus-level derangement of training targets (seed 2026,
  different group and different cluster, length-ordered windows of 8; same map
  for every model seed); one label per keystroke forces truncation or
  space-separated repetition of the donor. Loader check: MEG, subject ids,
  channel positions, segments identical; val/test targets identical; ~87–90 %
  of training labels changed.
- **Signal dependence** script ready (real runs only; final weights; zero MEG,
  fixed temporal and channel derangements, seed 2026); dry run on untrained
  weights exercised the code path only.
- **Text anchor** (clean test partitions): F1 official 0.338, D-clean 0.359,
  E-clean 0.349.
- Tests: `28 passed` (17 existing + 11 new in `tests/test_phase7_splits.py`).
  Integrity: raw FIF/MAT hashes match; historical and official-v1 preprocessing,
  official source, existing tests and earlier results unchanged; pinned env
  unchanged; no download; no LLM, no Evidence Selector; no local training.
- Artifacts: `results/phase7_{official_split_audit,clean_split_manifests,
  gpu_preflight,training_harness,control_harness,text_anchor,master_report,
  official_clusters}.json`; `data/manifests/official_v1_clean_{train,val,test}.json`,
  `s22_{D,E}_clean.json`, `s22_official_v1_dataset.json`;
  `results/runs/phase7/dry_run/`;
  `results/figures/debug/phase7_{split_clusters,split_composition,gpu_preflight}.png`.
