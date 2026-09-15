"""Phase 4E: one-trial real SpanishBCBL overfit recheck."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from neuroselect.data import NeuralTextDataset
from neuroselect.metrics import cer
from neuroselect.models import ConvCTC
from neuroselect.vocab_spanishbcbl import BLANK_ID, VOCAB, decode_ctc


def run_trial(item: dict, seed: int = 33, epochs: int = 1000) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    signal = item["signal"].float()
    target = item["target"].long()
    text = item["meta"]["text"]
    inputs = signal.unsqueeze(0)
    targets = target.unsqueeze(0)
    input_lengths = torch.tensor([signal.shape[0]], dtype=torch.long)
    target_lengths = torch.tensor([target.numel()], dtype=torch.long)
    model = ConvCTC(signal.shape[1], len(VOCAB))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    losses = []
    blank_history = []

    for _ in range(epochs):
        model.train()
        logits = model(inputs)
        log_probs = logits.log_softmax(-1).transpose(0, 1)
        loss = loss_fn(log_probs, targets, input_lengths, target_lengths)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            probabilities = logits[0].softmax(-1)
            argmax_ids = probabilities.argmax(-1)
            blank_history.append({
                "mean_blank_probability": float(probabilities[:, BLANK_ID].mean()),
                "max_nonblank_probability": float(probabilities[:, 1:].max()),
                "blank_argmax_fraction": float((argmax_ids == BLANK_ID).float().mean()),
                "decoded_length": len(decode_ctc(argmax_ids.tolist())),
            })
        losses.append(float(loss.detach()))

    model.eval()
    with torch.no_grad():
        final_logits = model(inputs)[0]
        final_probabilities = final_logits.softmax(-1)
        final_ids = final_probabilities.argmax(-1).tolist()
    decoded = decode_ctc(final_ids)
    return {
        "id": item["meta"]["id"],
        "target": text,
        "T": int(signal.shape[0]),
        "U": int(target.numel()),
        "T_over_U": float(signal.shape[0] / target.numel()),
        "channels": int(signal.shape[1]),
        "seed": seed,
        "learning_rate": 1e-2,
        "optimizer": "Adam",
        "epochs": epochs,
        "initial_loss": losses[0],
        "minimum_loss": min(losses),
        "final_loss": losses[-1],
        "decoded": decoded,
        "cer": cer(text, decoded),
        "decoded_length": len(decoded),
        "target_length": len(text),
        "mean_blank_probability_final": float(final_probabilities[:, BLANK_ID].mean()),
        "blank_argmax_fraction_final": float(
            (final_probabilities.argmax(-1) == BLANK_ID).float().mean()
        ),
        "max_nonblank_probability_final": float(final_probabilities[:, 1:].max()),
        "blank_history": blank_history,
        "loss_history": losses,
    }


def main() -> None:
    manifest = "data/processed/spanishbcbl_subset/manifest.jsonl"
    dataset = NeuralTextDataset(manifest, None)
    trial_ids = {"2.0_S22_1_block1", "3.0_S22_1_block1"}
    selected = [
        dataset[index]
        for index in range(len(dataset))
        if dataset[index]["meta"]["id"] in trial_ids
    ]
    if len(selected) != 2:
        raise RuntimeError(f"Expected trials 2 and 3 in {manifest}")

    results = {item["meta"]["id"]: run_trial(item) for item in selected}
    output = Path("results/real_trial_recheck.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    synthetic = json.loads(Path("results/synthetic_easy_overfit.json").read_text())
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    axes[0].plot(synthetic["loss_history"], label="synthetic abc")
    for trial_id, result in results.items():
        axes[0].plot(result["loss_history"], label=trial_id)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("CTC loss")
    axes[0].legend()
    axes[1].plot(
        synthetic["loss_history"],
        label="synthetic abc",
    )
    for trial_id, result in results.items():
        axes[1].plot(
            [row["blank_argmax_fraction"] for row in result["blank_history"]],
            label=f"{trial_id} blank argmax",
        )
    axes[1].set_title("Synthetic loss vs real blank-argmax traces")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("value")
    axes[1].legend()
    figure.tight_layout()
    figure_path = Path("results/figures/debug/real_vs_synthetic_training.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=120)
    plt.close(figure)

    print(json.dumps({
        trial_id: {
            "T": result["T"],
            "U": result["U"],
            "initial_loss": result["initial_loss"],
            "minimum_loss": result["minimum_loss"],
            "final_loss": result["final_loss"],
            "decoded": result["decoded"],
            "cer": result["cer"],
            "mean_blank_probability_final": result["mean_blank_probability_final"],
            "blank_argmax_fraction_final": result["blank_argmax_fraction_final"],
        }
        for trial_id, result in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
