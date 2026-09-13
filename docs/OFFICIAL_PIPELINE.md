# Official SpanishBCBL/DECOMEG pipeline

The vendored Brain2Qwerty repository is pinned to commit
`5f9889621d0df391c5aab37c996683d308e6e926`. The public source is
`https://github.com/facebookresearch/brain2qwerty`; the downloaded dataset
files are from `bcbl190626/SpanishBCBL` and are licensed CC BY-NC 4.0.

For this Phase 2 subset, `Study(name="Pinet2024Meg").run()` discovers the
matching `MEG/FIF/22_9788/231214/block1.fif` recording and
`MEG/logs/S22-session1_block1_list1.mat` behavioral log. The official code
loads the MATLAB log, cleans sentence text and button events, reads `STI101`
MEG triggers with MNE, aligns behavioral and trigger events, and emits
keystroke, word, sentence, and MEG events. The alignment assertion is kept
unchanged.

`SpanishBCBLPreprocessing._run(events)` applies the official v1 event
postprocessing: practice trials are removed, sentence events are rebuilt,
`sentence_typed` is populated, and `<space>`, `<special>`, and `<number>` are
mapped to space, `@`, and `9`. The public `run()` wrapper rejects the current
pandas DataFrame under the installed `exca` version, so the compatibility
bridge calls the official `_run` implementation without changing its logic.

The official MEG extractor configuration is 0.1–20 Hz band-pass, 50 Hz
resampling, RobustScaler, and clamp 5. NeuroSelect applies the same filter,
resampling, robust scaling, and clamp to sentence windows. Sentence windows
use the official production sentence `start` and `stop` events.
