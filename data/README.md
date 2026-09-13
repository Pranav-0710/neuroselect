# Data layout

The baseline accepts a JSON Lines manifest. Each line describes one sentence
trial:

```json
{"id":"trial-001","signal_path":"signals/trial-001.npy","text":"hola mundo","subject":"S1","session":"session-1","split":"train"}
```

Signals may be `.npy` arrays shaped `(time, channels)` or `.npz` files with a
`signal` array. The loader validates finite values and preserves subject,
session, and sentence identifiers.

The official SpanishBCBL release is raw MEG/EEG plus behavioral logs. It is not
converted into this manifest automatically because the public archive is
approximately 262 GB and requires the Brain2Qwerty `neuralset` event-building
pipeline. Use the official repository pipeline to create sentence-level
windows, then export the manifest without changing labels or splits.

