# Phase 3 alignment audit

**Status: VERIFIED FROM OFFICIAL CODE/DATA**

The official pipeline reads behavioral keypress metadata from the MATLAB log
and MEG trigger events from `STI101`. MNE returns trigger sample indices;
Brain2Qwerty converts them to seconds by dividing by the original
`raw.info["sfreq"]` of 1000 Hz. The aligned event frame is sorted by event
construction time and retains `Keystroke` events with their `start`, `stop`,
`button`, and `sentence_UID` fields.

For the eight prepared production trials, each retained keystroke maps to one
character after `<space>`, `<special>`, and `<number>` conversion. Every
target character has exactly one retained keystroke. All trials have valid
CTC geometry: `T=260–346`, `U=27–34`, and `T/U=7.44–10.19`.

**VERIFIED FROM OFFICIAL CODE/DATA:** no additional FIF/MAT offset was
applied by NeuroSelect. Sentence boundaries are rebuilt by the official
preprocessing from the minimum keystroke start to maximum keystroke stop.

**OBSERVED:** trials 8 and 9 have official stimulus text differing from the
typed event-derived text because the participant typed a different character
sequence. The manifest correctly uses official `sentence_typed`, not the
stimulus sentence.
