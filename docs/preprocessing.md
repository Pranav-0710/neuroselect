# NeuroSelect preprocessing variants

## Current historical pipeline

The historical NeuroSelect path remains implemented in
`scripts/prepare_real_subset.py`. It continuously filters, resamples, and
RobustScales the recording, then sentence-crops each trial, applies baseline
correction to the sentence crop, clamps the sentence tensor, and stores
`(time, channels)` arrays.

This path is retained unchanged for reproducibility of previous experiments.

## Official-v1-compatible pipeline

`src/neuroselect/official_v1_preprocessing.py` provides the explicit
`NeuroSelectOfficialV1EventPreprocessing` variant and
`preprocess_event_official_v1` helper. Callers must request it explicitly; it
does not replace or switch the historical path.

The variant:

1. selects the 306 MEG channels;
2. filters continuously at 0.1–20 Hz;
3. resamples continuously to 50 Hz;
4. applies a continuous `RobustScaler`;
5. extracts each `[-0.2, +0.3]` second event window without padding;
6. subtracts each channel's mean over relative `[0.0, +0.2]` seconds;
7. clamps each event tensor to `[-5, +5]`.

Outputs use `(time, channels)` shape `(25, 306)` and include provenance
metadata identifying `preprocessing_variant="official_v1_compatible"` and
official revision `5f9889621d0df391c5aab37c996683d308e6e926`.

The gated implementation mirrors the official `TimedArray.overlap` semantics:
fractional event-window bounds are resolved on the 50 Hz grid, baseline slices
are derived from the snapped extended overlap, and the requested event window
is cropped before clamping. It was validated with element-for-element equality
on all 240 events in the existing eight-trial development subset (trials 2–9).
This does not establish validation for the entire dataset.
