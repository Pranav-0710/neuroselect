"""Run the mandatory tiny-subset CTC overfit check without external data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from neuroselect.data import NeuralTextDataset, collate_batch
from neuroselect.models import ConvCTC
from neuroselect.training import run_epoch, seed_everything
from neuroselect.vocab import VOCAB


def main() -> None:
    seed_everything(33)
    root = Path("results/sanity_fixture")
    root.mkdir(parents=True, exist_ok=True)
    records = []
    texts = ["ab"]
    for index, text in enumerate(texts):
        signal = np.zeros((20, 7), dtype=np.float32)
        for step, char in enumerate(text):
            signal[step * 7 : (step + 1) * 7, ord(char) - ord("a")] = 3.0
        path = root / f"{index}.npy"
        np.save(path, signal)
        records.append(
            {
                "id": str(index),
                "signal_path": path.name,
                "text": text,
                "subject": "synthetic",
                "session": "synthetic",
                "split": "train",
            }
        )
    manifest = root / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    dataset = NeuralTextDataset(manifest, "train")
    loader = DataLoader(dataset, 4, shuffle=False, collate_fn=collate_batch)
    model = ConvCTC(7, len(VOCAB), hidden=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(150):
        losses.append(run_epoch(model, loader, optimizer, torch.device("cpu")))
    result = {
        "seed": 33,
        "examples": len(dataset),
        "initial_ctc_loss": losses[0],
        "final_ctc_loss": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "overfit_success": losses[-1] < 0.2,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/sanity_check.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
