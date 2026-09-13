# Phase 1 baseline

## Scope

The baseline maps a variable-length neural signal window to a character
sequence with a compact convolutional frontend, bidirectional GRU, and CTC
loss. It intentionally excludes the LLM, evidence selector, uncertainty
system, calibration system, and application UI.

## Input and output

Input tensors are `(batch, time, channels)`. The manifest preserves trial ID,
subject, session, and split. The model emits `(batch, time, vocabulary)` logits.
The generic vocabulary remains blank plus lowercase English letters and
space (28 classes). SpanishBCBL uses the verified 30-class CTC vocabulary
in `src/neuroselect/vocab_spanishbcbl.py`.

## Training

`src/neuroselect/training.py` uses AdamW, gradient clipping, deterministic
seeding, padded variable-length batches, and `torch.nn.CTCLoss`. The training
entry point is intentionally small enough for CPU tests and one limited GPU.
Mixed precision and gradient accumulation are left for the measured full-data
run rather than being enabled before the tensor pipeline is validated.

## Evaluation

`scripts/evaluate.py` writes per-example reference, prediction, raw edit
distance, CER, and WER to JSON and writes ranked best/median/worst examples.
The final public-data CER/WER is not reported because the tiny real-data
overfit gate has not yet succeeded.

## Sanity status

The synthetic end-to-end fixture in `scripts/sanity_check.py` verifies that
signals load, CTC dimensions are valid, and loss can decrease on a tiny
training subset. On 2026-09-13 it reduced CTC loss from `29.3315` to
`0.0020` on one synthetic example (`seed=33`), satisfying the overfit gate.
This is a pipeline check, not a scientific dataset result. Run:

```powershell
$env:PYTHONPATH="src"
python scripts/sanity_check.py
```

## Phase 2 real-data status

The SpanishBCBL subset is connected through the official event-building
pipeline and dataset-specific vocabulary. A one-trial forward pass and CTC
loss are finite. Tiny real-data training reduces loss, but does not
reconstruct the training sentences; real CER/WER is therefore not reported.

## Limitations

The preparation script parses the selected `.fif` file through the official
Brain2Qwerty v1 event path, but it does not represent the full archive.
Exact generalization metrics require more recording units and a successful
real-data learning/validation experiment.
