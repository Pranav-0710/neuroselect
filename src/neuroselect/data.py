"""Manifest-backed signal dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .vocab import encode_text
from .vocab_spanishbcbl import encode_text as encode_spanishbcbl


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        required = {"id", "signal_path", "text", "subject", "session", "split"}
        missing = required - record.keys()
        if missing:
            raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
        records.append(record)
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def load_signal(path: str | Path) -> np.ndarray:
    file_path = Path(path)
    if file_path.suffix == ".npy":
        array = np.load(file_path)
    elif file_path.suffix == ".npz":
        with np.load(file_path) as archive:
            if "signal" not in archive:
                raise ValueError(f"{file_path} must contain an array named 'signal'")
            array = archive["signal"]
    else:
        raise ValueError(f"Unsupported signal file: {file_path}")
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{file_path} must have shape (time, channels), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{file_path} contains NaN or infinite values")
    return array


class NeuralTextDataset(Dataset):
    """Loads variable-length `(time, channels)` arrays and CTC targets."""

    def __init__(self, manifest: str | Path, split: str | None = None):
        records = read_manifest(manifest)
        self.records = [r for r in records if split is None or r["split"] == split]
        if not self.records:
            raise ValueError(f"No records for split={split!r}")
        self.root = Path(manifest).parent

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        signal = torch.from_numpy(load_signal(self.root / record["signal_path"]))
        encoder = (
            encode_spanishbcbl
            if record.get("vocab", "english") == "spanishbcbl"
            else encode_text
        )
        target = torch.tensor(encoder(record["text"]), dtype=torch.long)
        return {"signal": signal, "target": target, "meta": record}


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    channels = {item["signal"].shape[1] for item in batch}
    if len(channels) != 1:
        raise ValueError(f"All signals must have the same channel count, got {channels}")
    max_time = max(item["signal"].shape[0] for item in batch)
    signals = torch.zeros(len(batch), max_time, channels.pop())
    input_lengths = []
    targets = []
    target_lengths = []
    for index, item in enumerate(batch):
        time = item["signal"].shape[0]
        signals[index, :time] = item["signal"]
        input_lengths.append(time)
        targets.append(item["target"])
        target_lengths.append(len(item["target"]))
    return {
        "signals": signals,
        "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
        "targets": torch.cat(targets),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "meta": [item["meta"] for item in batch],
    }
