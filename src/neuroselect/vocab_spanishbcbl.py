"""Verified SpanishBCBL character vocabulary.

The token order follows Brain2Qwerty v1 ``BUTTON_MAPPING``/``CHAR_INDEX``.
SpanishBCBL preprocessing normalizes accents and maps unsupported keyboard
events to ``@`` or ``9`` before this CTC vocabulary is applied.
"""

from __future__ import annotations

BLANK_ID = 0
TOKEN_TO_CHAR = {
    1: "s",
    2: "o",
    3: "t",
    4: "e",
    5: "n",
    6: "c",
    7: "i",
    8: "a",
    9: " ",
    10: "d",
    11: "l",
    12: "r",
    13: "b",
    14: "@",
    15: "z",
    16: "v",
    17: "f",
    18: "m",
    19: "u",
    20: "h",
    21: "p",
    22: "g",
    23: "q",
    24: "w",
    25: "x",
    26: "y",
    27: "j",
    28: "k",
    29: "9",
}
VOCAB = ("<blank>",) + tuple(TOKEN_TO_CHAR.values())
CHAR_TO_ID = {char: token for token, char in TOKEN_TO_CHAR.items()}


def encode_text(text: str) -> list[int]:
    unsupported = sorted(set(text) - set(CHAR_TO_ID))
    if unsupported:
        raise ValueError(f"Unsupported SpanishBCBL characters: {unsupported!r}")
    return [CHAR_TO_ID[char] for char in text]


def decode_ctc(ids: list[int] | tuple[int, ...]) -> str:
    output: list[str] = []
    previous = BLANK_ID
    for token in ids:
        token = int(token)
        if token != BLANK_ID and token != previous:
            if token not in TOKEN_TO_CHAR:
                raise ValueError(f"Invalid SpanishBCBL vocabulary id: {token}")
            output.append(TOKEN_TO_CHAR[token])
        previous = token
    return "".join(output)

