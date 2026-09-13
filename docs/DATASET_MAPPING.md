# SpanishBCBL mapping

The official v1 `BUTTON_MAPPING` contains 29 non-blank classes in the order
used by `CHAR_INDEX`. NeuroSelect adds CTC blank ID 0 and preserves the
official order in `src/neuroselect/vocab_spanishbcbl.py`.

The verified CTC mapping is:

`s o t e n c i a <space> d l r b @ z v f m u h p g q w x y j k 9`

The official preprocessing normalizes accented characters and maps special
keyboard events to `@` and numeric events to `9`. The resulting
`sentence_typed` strings in the downloaded block are lowercase ASCII plus
spaces; no Unicode character handling was invented.

Round-trip and special-token tests are in `tests/test_baseline.py`.
