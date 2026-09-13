"""Phase 4B: isolate decoder, CTC, model, and synthetic learnability."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from neuroselect.data import NeuralTextDataset, collate_batch
from neuroselect.metrics import cer
from neuroselect.models import ConvCTC
from neuroselect.training import seed_everything
from neuroselect.vocab_spanishbcbl import VOCAB, decode_ctc, encode_text


BLANK = 0


def decode_ids(ids: list[int]) -> str:
    return decode_ctc(ids)


def ctc_loss(logits: torch.Tensor, target: list[int]) -> torch.Tensor:
    target_tensor = torch.tensor(target, dtype=torch.long)
    log_probs = logits.log_softmax(-1).unsqueeze(1)
    return torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)(
        log_probs,
        target_tensor,
        torch.tensor([logits.shape[0]]),
        torch.tensor([len(target)]),
    )


def direct_logits(target: list[int], epochs: int = 500) -> dict:
    torch.manual_seed(33)
    logits = torch.nn.Parameter(torch.zeros(40, len(VOCAB)))
    optimizer = torch.optim.Adam([logits], lr=0.1)
    losses = []
    for _ in range(epochs):
        loss = ctc_loss(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    ids = logits.detach().argmax(-1).tolist()
    prediction = decode_ids(ids)
    return {
        "target": decode_ids(target),
        "initial_loss": losses[0],
        "minimum_loss": min(losses),
        "final_loss": losses[-1],
        "epochs": epochs,
        "prediction": prediction,
        "cer": cer(decode_ids(target), prediction),
        "decoded_indices": ids,
    }


def make_signal(
    text: str,
    channels: int = 306,
    repeat: int = 8,
    gap: int | None = None,
    noise: float = 0.0,
    smooth: bool = False,
    seed: int = 33,
) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
    target = encode_text(text)
    rng = np.random.default_rng(seed)
    gap = repeat if gap is None else gap
    total = len(target) * (repeat + gap)
    signal = rng.normal(0.0, noise, (total, channels)).astype(np.float32)
    regions = []
    for index, token in enumerate(target):
        start = index * (repeat + gap)
        stop = start + repeat
        channel = (token - 1) % min(channels, 30)
        signal[start:stop, channel] += 4.0
        regions.append((start, stop, VOCAB[token]))
    if smooth:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        for channel in range(min(channels, 30)):
            signal[:, channel] = np.convolve(signal[:, channel], kernel, mode="same")
    return signal, regions


def train_model(signal: np.ndarray, text: str, epochs: int = 300) -> dict:
    target = encode_text(text)
    item = {
        "signal": torch.from_numpy(signal),
        "target": torch.tensor(target, dtype=torch.long),
        "meta": {"id": "synthetic", "text": text},
    }
    loader = DataLoader([item], 1, shuffle=False, collate_fn=collate_batch)
    model = ConvCTC(signal.shape[1], len(VOCAB), hidden=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    blank_stats = []
    start_time = time.perf_counter()
    for _ in range(epochs):
        model.train()
        batch = next(iter(loader))
        logits = model(batch["signals"])[0]
        loss = ctc_loss(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            probs = logits.softmax(-1)
            ids = probs.argmax(-1).tolist()
            blank_stats.append({
                "mean_blank_probability": float(probs[:, BLANK].mean()),
                "max_nonblank_probability": float(probs[:, 1:].max()),
                "blank_argmax_fraction": float(np.mean(np.asarray(ids) == BLANK)),
                "decoded_length": len(decode_ids(ids)),
            })
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        logits = model(item["signal"].unsqueeze(0))[0]
        ids = logits.argmax(-1).tolist()
    prediction = decode_ids(ids)
    return {
        "target": text,
        "T": int(signal.shape[0]),
        "U": len(target),
        "T_over_U": float(signal.shape[0] / len(target)),
        "initial_loss": losses[0],
        "minimum_loss": min(losses),
        "final_loss": losses[-1],
        "epochs": epochs,
        "prediction": prediction,
        "cer": cer(text, prediction),
        "decoded_indices": ids,
        "seconds": time.perf_counter() - start_time,
        "blank_history": blank_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-manifest", default="data/processed/spanishbcbl_subset/manifest.jsonl")
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    seed_everything(33)
    results = Path("results")
    debug = results / "figures" / "debug"
    debug.mkdir(parents=True, exist_ok=True)

    decoder_cases = {}
    for text in ("abc", "hello", "a  bb"):
        ids = encode_text(text)
        path = []
        for token in ids:
            path.extend((BLANK, token, token))
        decoder_cases[text] = {"path": path, "decoded": decode_ids(path), "passed": decode_ids(path) == text}
    (results / "ctc_decoder_test.json").write_text(json.dumps(decoder_cases, indent=2), encoding="utf-8")

    direct = {text: direct_logits(encode_text(text)) for text in ("abc", "hello")}
    (results / "ctc_direct_logits_test.json").write_text(json.dumps(direct, indent=2), encoding="utf-8")

    easy_signal, regions = make_signal("abc", repeat=30)
    easy = train_model(easy_signal, "abc", args.epochs)
    (results / "synthetic_easy_overfit.json").write_text(json.dumps(easy, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(12, 4))
    easy_channels = [(token - 1) % 30 for token in encode_text("abc")]
    ax.plot(easy_signal[:, easy_channels])
    ax.legend([f"channel {channel}" for channel in easy_channels])
    for start, stop, label in regions:
        ax.axvspan(start, stop, alpha=0.15)
        ax.text((start + stop) / 2, 4.2, label, ha="center")
    ax.set_title("Easy synthetic target encoding: abc")
    ax.set_xlabel("timestep")
    fig.tight_layout()
    fig.savefig(debug / "synthetic_easy_signal.png", dpi=120)
    plt.close(fig)

    lengths = {}
    for text in ("abc", "hello", "brain", "hello world", "la tasa excede las velocidades"):
        signal, _ = make_signal(text, repeat=4, gap=4)
        lengths[text] = train_model(signal, text, args.epochs)
    (results / "synthetic_sequence_length.json").write_text(json.dumps(lengths, indent=2), encoding="utf-8")

    noise = {}
    for level in (0.0, 0.05, 0.2, 0.5):
        signal, _ = make_signal("hello", repeat=4, gap=4, noise=level)
        noise[str(level)] = train_model(signal, "hello", args.epochs)
    (results / "synthetic_noise_sweep.json").write_text(json.dumps(noise, indent=2), encoding="utf-8")

    realistic_signal, _ = make_signal("abc", repeat=40, gap=40, noise=0.15, smooth=True)
    realistic = train_model(realistic_signal, "abc", args.epochs)
    (results / "synthetic_realistic_dimensions.json").write_text(json.dumps(realistic, indent=2), encoding="utf-8")

    shuffled_signal, _ = make_signal("abc", repeat=40, gap=40, noise=0.15, smooth=True)
    shuffled = train_model(shuffled_signal, "xyz", args.epochs)
    (results / "synthetic_shuffle_control.json").write_text(json.dumps(shuffled, indent=2), encoding="utf-8")

    real = {}
    dataset = NeuralTextDataset(args.real_manifest, None)
    for index in (0, 1):
        item = dataset[index]
        real[str(item["meta"]["id"])] = train_model(item["signal"].numpy(), item["meta"]["text"], args.epochs)
    (results / "real_meg_recheck.json").write_text(json.dumps(real, indent=2), encoding="utf-8")

    collapse = {
        "synthetic_easy": easy["blank_history"],
        "real_trial_2": next(iter(real.values()))["blank_history"],
    }
    (results / "ctc_blank_collapse_diagnostic.json").write_text(json.dumps(collapse, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, history in collapse.items():
        ax.plot([row["blank_argmax_fraction"] for row in history], label=name)
    ax.set_title("CTC blank argmax fraction")
    ax.set_xlabel("epoch")
    ax.set_ylabel("fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(debug / "ctc_blank_collapse.png", dpi=120)
    plt.close(fig)
    print(json.dumps({
        "decoder_passed": all(row["passed"] for row in decoder_cases.values()),
        "direct_logits": {key: value["prediction"] for key, value in direct.items()},
        "easy_prediction": easy["prediction"],
        "easy_cer": easy["cer"],
        "real_predictions": {key: value["prediction"] for key, value in real.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
