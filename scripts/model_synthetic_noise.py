"""Phase 4D: noise robustness of the exact synthetic model setup."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from neuroselect.metrics import cer
from neuroselect.models import ConvCTC
from neuroselect.vocab_spanishbcbl import BLANK_ID, VOCAB, decode_ctc, encode_text


def clean_signal() -> np.ndarray:
    signal = np.zeros((300, 306), dtype=np.float32)
    signal[0:80, 0] = 5.0
    signal[100:180, 1] = 5.0
    signal[200:280, 2] = 5.0
    return signal


def run_condition(name: str, noise_std: float, seed: int = 33) -> tuple[dict, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    clean = clean_signal()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, clean.shape).astype(np.float32)
    signal = clean + noise
    target_text = "abc"
    target_ids = encode_text(target_text)
    inputs = torch.from_numpy(signal).unsqueeze(0)
    targets = torch.tensor([target_ids], dtype=torch.long)
    input_lengths = torch.tensor([300])
    target_lengths = torch.tensor([3])
    model = ConvCTC(306, len(VOCAB))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    history = []

    for _ in range(1000):
        logits = model(inputs)
        loss = loss_fn(
            logits.log_softmax(-1).transpose(0, 1),
            targets,
            input_lengths,
            target_lengths,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))

    model.eval()
    with torch.no_grad():
        logits = model(inputs)[0]
        decoded_ids = logits.argmax(-1).tolist()
    decoded = decode_ctc(decoded_ids)
    clean_std = float(clean.std())
    actual_noise_std = float(noise.std())
    return {
        "condition": name,
        "seed": seed,
        "noise_std_requested": noise_std,
        "signal_std": clean_std,
        "noise_std_actual": actual_noise_std,
        "T": 300,
        "channels": 306,
        "target": target_text,
        "target_length": 3,
        "learning_rate": 1e-2,
        "optimizer": "Adam",
        "epochs": 1000,
        "initial_loss": history[0],
        "minimum_loss": min(history),
        "final_loss": history[-1],
        "decoded": decoded,
        "cer": cer(target_text, decoded),
        "exact_recovery": decoded == target_text,
        "loss_history": history,
    }, signal


def main() -> None:
    conditions = {
        "no_noise": 0.0,
        "low_noise": 0.5,
        "medium_noise": 2.0,
        "high_noise": 5.0,
    }
    results = {}
    signals = {}
    for name, noise_std in conditions.items():
        results[name], signals[name] = run_condition(name, noise_std)

    result_path = Path("results/synthetic_noise_sweep.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(12, 5))
    for name in conditions:
        axis.plot(signals[name][:, 0], label=name)
    axis.set_title("Synthetic noise levels: representative encoded channel 0")
    axis.set_xlabel("timestep")
    axis.set_ylabel("activation")
    axis.legend()
    figure.tight_layout()
    figure_path = Path("results/figures/debug/synthetic_noise_levels.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=120)
    plt.close(figure)

    print(json.dumps({
        name: {
            "decoded": result["decoded"],
            "cer": result["cer"],
            "signal_std": result["signal_std"],
            "noise_std_actual": result["noise_std_actual"],
        }
        for name, result in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
