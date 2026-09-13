# SpanishBCBL mapping

The official v1 `BUTTON_MAPPING` contains 29 non-blank classes in the order
used by `CHAR_INDEX`. NeuroSelect adds CTC blank ID 0 and preserves the
official order in `src/neuroselect/vocab_spanishbcbl.py`.

The verified CTC mapping is:

`s o t e n c i a <space> d l r b @ z v f m u h p g q w x y j k 9`

| Official label | Official class | NeuroSelect index | Decoded symbol |
|---|---:|---:|---|
| `s` | 0 | 1 | `s` |
| `o` | 1 | 2 | `o` |
| `t` | 2 | 3 | `t` |
| `e` | 3 | 4 | `e` |
| `n` | 4 | 5 | `n` |
| `c` | 5 | 6 | `c` |
| `i` | 6 | 7 | `i` |
| `a` | 7 | 8 | `a` |
| `<space>` | 8 | 9 | space |
| `d` | 9 | 10 | `d` |
| `l` | 10 | 11 | `l` |
| `r` | 11 | 12 | `r` |
| `b` | 12 | 13 | `b` |
| `<special>` | 13 | 14 | `@` |
| `z` | 14 | 15 | `z` |
| `v` | 15 | 16 | `v` |
| `f` | 16 | 17 | `f` |
| `m` | 17 | 18 | `m` |
| `u` | 18 | 19 | `u` |
| `h` | 19 | 20 | `h` |
| `p` | 20 | 21 | `p` |
| `g` | 21 | 22 | `g` |
| `q` | 22 | 23 | `q` |
| `w` | 23 | 24 | `w` |
| `x` | 24 | 25 | `x` |
| `y` | 25 | 26 | `y` |
| `j` | 26 | 27 | `j` |
| `k` | 27 | 28 | `k` |
| `<number>` | 28 | 29 | `9` |
| CTC blank | n/a | 0 | blank |

The official source definition is
`vendor/brain2qwerty/brain2qwerty_v1/utils.py`; the NeuroSelect offset by one
is required because CTC blank is inserted at index 0.

The official preprocessing normalizes accented characters and maps special
keyboard events to `@` and numeric events to `9`. The resulting
`sentence_typed` strings in the downloaded block are lowercase ASCII plus
spaces; no Unicode character handling was invented.

Round-trip and special-token tests are in `tests/test_baseline.py`.

Phase 3 independently reconstructed the first three complete selected
sentences from cleaned keystroke events. Raw button labels, mapped characters,
official sentence text, manifest text, and CTC target text matched exactly.
Across all eight selected trials, every target character had one retained
keystroke. Trials 8 and 9 intentionally expose participant typing errors:
their event-derived `sentence_typed` differs from the displayed stimulus
sentence, and the manifest follows the official typed sequence.
