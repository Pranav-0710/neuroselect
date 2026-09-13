# Real-data pipeline

The reproducible preparation command is:

```powershell
$env:PYTHONPATH="vendor\brain2qwerty;src"
python scripts\prepare_real_subset.py data\raw\spanishbcbl_s22
```

It runs official event extraction and preprocessing, selects production
sentences, applies the official 0.1–20 Hz filter and 50 Hz resampling to the
continuous MEG recording, applies continuous-recording RobustScaler,
extracts official sentence intervals, baseline-corrects, clamps to ±5, and
saves `(time, channels)` float32 `.npy` arrays.
The resulting manifest is `data/processed/spanishbcbl_subset/manifest.jsonl`
and the CSV audit manifest is
`data/manifests/spanishbcbl_meg_subset.csv`.

The extracted subset contains eight complete production trials from subject
S22, session 1, block 1, with 306 MEG channels. A one-trial forward pass and
finite CTC loss succeed. The tiny real-data overfit attempt reduced loss but
did not reproduce the target strings, so no real CER/WER baseline is claimed.
