"""Small, deterministic diagnostics shared by real-data forensic scripts."""

from __future__ import annotations

from collections.abc import Sequence


def validate_ctc_geometry(input_length: int, target: Sequence[int]) -> dict:
    target = [int(value) for value in target]
    repeated = [
        index
        for index, (left, right) in enumerate(zip(target, target[1:]))
        if left == right
    ]
    minimum_timesteps = len(target) + len(repeated)
    return {
        "T": int(input_length),
        "U": len(target),
        "T_over_U": float(input_length / len(target)) if target else None,
        "repeated_adjacent_indices": repeated,
        "has_repeated_adjacent_target": bool(repeated),
        "minimum_ctc_timesteps": minimum_timesteps,
        "structurally_feasible": input_length >= minimum_timesteps,
    }
