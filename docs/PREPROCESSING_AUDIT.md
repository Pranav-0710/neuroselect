# Phase 3 preprocessing audit

**VERIFIED FROM OFFICIAL CODE:** `MegExtractor` preprocesses the complete
continuous MEG recording before event windows are materialized. Its order is
channel selection, filtering, resampling, robust scaling, optional baseline
correction on each segment, and clamping.

The previous NeuroSelect preparation script filtered each 4–8 second cropped
trial independently. MNE reported FIR filter lengths longer than the cropped
signals. This was a concrete preprocessing mismatch and has been fixed:
continuous MEG is now filtered and resampled first, robust-scaled globally,
then sentence windows are cropped and baseline-corrected/clamped.

**OBSERVED:** after the correction, processed trials remain finite and retain
substantial variation. Per-trial clipped fractions range from 0% to 0.342%;
near-zero channel fraction is approximately 3.3% (the MEG channel selection
contains channels with negligible variance). No NaN or Inf values were found.

**UNKNOWN:** the exact official cached extractor values for channel dropping,
projectors, and baseline treatment have not been reproduced through the
`exca` wrapper because the installed wrapper rejects a pandas DataFrame.
