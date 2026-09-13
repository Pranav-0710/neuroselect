# Phase 4B — CTC Learnability

## Scope

This phase isolates the existing `Conv1D + BiGRU + Linear + CTC` stack from
real MEG. No Evidence Selector, LLM, calibration, uncertainty system, or
architecture redesign was used. Experiments use seed `33`, SpanishBCBL
vocabulary, blank id `0`, and `torch.nn.CTCLoss(blank=0, zero_infinity=True)`.

## Results

| Experiment | Result | Conclusion |
|---|---|---|
| Pure greedy decoder | `abc`, `hello`, spaces, repeats, and blanks collapse correctly | **VERIFIED** decoder behavior |
| Trainable CTC logits | `abc` and `hello` recovered exactly | **VERIFIED** CTC tensor dimensions, target encoding, and blank index work |
| Exact model, easy `abc` | `T=180`, `U=3`, `T/U=60`; final prediction `c`, CER `0.667` | **OBSERVED** exact model did not pass this long-segment gate |
| Sequence scaling | `T/U=8`: `abc` passed; `brain` passed; `hello` and `hello world` near-pass; 30-character Spanish target failed | **OBSERVED** optimization is length-sensitive |
| Noise sweep | `T/U=8`: clean CER `0.40`, noise `0.05` passed, higher noise failed | **OBSERVED** current synthetic code is not consistently noise-robust |
| Realistic dimensions | `T=240`, `U=3`, `T/U=80`; prediction empty, CER `1.0` | **OBSERVED** realistic dimensionality did not pass |
| Shuffled target | Signal encoded `abc`, target `xyz`, yet `xyz` was recovered | **OBSERVED** control is confounded by fixed temporal position; it does not prove signal dependence |
| Real MEG recheck | Trial 2 CER `0.70`; trial 3 CER `0.37`; outputs were partial sentences | **OBSERVED** no reliable one-trial exact recovery |

## Blank-collapse analysis

`results/ctc_blank_collapse_diagnostic.json` and
`results/figures/debug/ctc_blank_collapse.png` record blank probability,
blank argmax fraction, decoded length, and nonblank probability over training.
The synthetic and real traces do not establish a single shared mechanism:
the synthetic run can collapse to blank before recovering a partial token,
while real trials produce partial language-like strings.

## Root-cause classification

1. **A — CTC decoder bug:** ruled out (**VERIFIED**).
2. **B — CTC loss/target encoding bug:** not supported; direct logits pass
   (**VERIFIED**).
3. **C — Conv1D/BiGRU implementation problem:** not established. The exact
   model recovers `abc` at `T/U=8` and `brain` (**UNKNOWN**).
4. **D — Optimization problem:** plausible (**INFERRED**) because outcomes
   vary by target and noise despite a valid CTC stack.
5. **E — Sequence-length/model-capacity limitation:** plausible
   (**INFERRED**) because `T/U=8` can pass while long-segment and
   realistic-dimension controls fail.
6. **F — Artificial-control design problem:** verified for the shuffled
   control (**VERIFIED**); a single fixed sequence allows positional
   memorization even when signal and target disagree.
7. **G — Real-data learning difficulty:** not selectable yet (**UNKNOWN**);
   synthetic gates did not all pass.
8. **H — NOT YET ESTABLISHED:** current overall classification
   (**VERIFIED** as the reporting status).

## Required conclusion

**NOT YET ESTABLISHED.** The decoder and direct CTC objective are validated,
and exact-model recovery is possible for some short synthetic cases, but the
required long-segment/realistic gate and shuffled control are not decisive.
Do not proceed to NeuroSelect Evidence Selector experiments.
