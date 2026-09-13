# Phase 3 CTC geometry

`validate_ctc_geometry(input_length, target)` reports `T`, `U`, `T/U`,
adjacent repeated target positions, the minimum required CTC timesteps, and
structural feasibility.

For the eight selected trials, every trial is feasible. There are no adjacent
repeated target symbols in the selected targets. The processed sequence length
is 260–346 timesteps for targets of length 27–34, so temporal downsampling is
not the cause of invalid CTC inputs.

**VERIFIED FROM DATA:** each trial has `T >= U` and satisfies the stricter
`T >= U + repeated_adjacent_count` condition.
