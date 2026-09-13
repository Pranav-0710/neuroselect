"""Character vocabulary and CTC helpers."""

from __future__ import annotations

VOCAB = ("<blank>",) + tuple("abcdefghijklmnopqrstuvwxyz") + (" ",)
BLANK_ID = 0
CHAR_TO_ID = {char: idx for idx, char in enumerate(VOCAB) if idx}


def encode_text(text: str) -> list[int]:
    """Encode lowercase English text; reject unsupported characters explicitly."""
    normalized = text.lower()
    unsupported = sorted(set(normalized) - set(CHAR_TO_ID))
    if unsupported:
        raise ValueError(f"Unsupported characters: {unsupported!r}")
    return [CHAR_TO_ID[char] for char in normalized]


def decode_ctc(ids: list[int] | tuple[int, ...]) -> str:
    """Greedy CTC decode with blank removal and repeat collapse."""
    output: list[str] = []
    previous = BLANK_ID
    for token in ids:
        token = int(token)
        if token != BLANK_ID and token != previous:
            if not 0 < token < len(VOCAB):
                raise ValueError(f"Invalid vocabulary id: {token}")
            output.append(VOCAB[token])
        previous = token
    return "".join(output)

