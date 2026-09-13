"""Small CTC training entry point."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .data import NeuralTextDataset, collate_batch
from .models import ConvCTC
from .vocab import BLANK_ID, VOCAB, decode_ctc
from .vocab_spanishbcbl import VOCAB as SPANISH_VOCAB


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, optimizer, device):
    training = optimizer is not None
    model.train(training)
    loss_fn = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    total = 0.0
    count = 0
    for batch in loader:
        signals = batch["signals"].to(device)
        logits = model(signals)
        log_probs = logits.log_softmax(-1).transpose(0, 1)
        input_lengths = batch["input_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        loss = loss_fn(log_probs, targets, input_lengths, target_lengths)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        total += float(loss.detach()) * signals.shape[0]
        count += signals.shape[0]
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--output", default="results/checkpoint.pt")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = NeuralTextDataset(args.manifest, "train")
    val = NeuralTextDataset(args.manifest, "val")
    loader = DataLoader(train, args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val, args.batch_size, shuffle=False, collate_fn=collate_batch)
    channels = train[0]["signal"].shape[1]
    vocabulary_size = len(SPANISH_VOCAB) if train.records[0].get("vocab") == "spanishbcbl" else len(VOCAB)
    model = ConvCTC(channels, vocabulary_size, args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, loader, optimizer, device)
        val_loss = run_epoch(model, val_loader, None, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(json.dumps(history[-1]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "channels": channels, "hidden": args.hidden}, output)


if __name__ == "__main__":
    main()
