"""Leakage-safe sentence-level splitting."""

from __future__ import annotations

import hashlib
from collections import defaultdict


def split_by_unique_text(
    records: list[dict],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 1,
) -> list[dict]:
    """Assign every identical text to one deterministic split."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1")
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record["text"]).strip().lower()].append(record)
    ordered = sorted(
        groups.items(),
        key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest(),
    )
    targets = [ratios[0], ratios[0] + ratios[1]]
    total = len(records)
    counts = [0, 0, 0]
    output: list[dict] = []
    for text, group in ordered:
        score = (counts[0] + len(group)) / max(total, 1)
        split_index = 0 if score <= targets[0] else 1
        if split_index == 1 and (counts[0] + counts[1] + len(group)) / max(
            total, 1
        ) > targets[1]:
            split_index = 2
        split = ("train", "val", "test")[split_index]
        counts[split_index] += len(group)
        for record in group:
            item = dict(record)
            item["split"] = split
            output.append(item)
    return output

