"""Phase 5B: controlled blank-collapse diagnostic matrix."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from neuroselect.data import NeuralTextDataset, collate_batch
from neuroselect.metrics import cer, wer
from neuroselect.models import ConvCTC
from neuroselect.vocab_spanishbcbl import BLANK_ID, VOCAB, decode_ctc


TRAIN_IDS = [f"{index}.0_S22_1_block1" for index in range(2, 8)]
EVAL_IDS = [f"{index}.0_S22_1_block1" for index in range(8, 10)]


def train_one(items: list[dict], lr: float, clip_norm: float | None) -> tuple[ConvCTC, float]:
    model = ConvCTC(306, len(VOCAB))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    batch = collate_batch(items)
    final_loss = float("nan")
    for _ in range(300):
        model.train()
        logits = model(batch["signals"])
        loss = loss_fn(
            logits.log_softmax(-1).transpose(0, 1),
            batch["targets"],
            batch["input_lengths"],
            batch["target_lengths"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        final_loss = float(loss.detach())
    if not np.isfinite(final_loss):
        raise RuntimeError("Non-finite training loss")
    return model, final_loss


def evaluate_items(model: ConvCTC, items: list[dict]) -> dict:
    rows = []
    for item in items:
        model.eval()
        signal = item["signal"].unsqueeze(0)
        with torch.no_grad():
            logits = model(signal)[0]
            probabilities = logits.softmax(-1)
            ids = probabilities.argmax(-1)
        target = item["meta"]["text"]
        decoded = decode_ctc(ids.tolist())
        rows.append({
            "trial_id": item["meta"]["id"],
            "target": target,
            "decoded": decoded,
            "cer": cer(target, decoded),
            "wer": wer(target, decoded),
            "target_length": len(target),
            "decoded_length": len(decoded),
            "decoded_length_ratio": len(decoded) / max(len(target), 1),
            "mean_blank_probability": float(probabilities[:, BLANK_ID].mean()),
            "blank_argmax_fraction": float((ids == BLANK_ID).float().mean()),
        })
    return {
        "per_trial": rows,
        "mean_cer": float(np.mean([row["cer"] for row in rows])),
        "mean_wer": float(np.mean([row["wer"] for row in rows])),
        "mean_blank_probability": float(np.mean([row["mean_blank_probability"] for row in rows])),
        "blank_argmax_fraction": float(np.mean([row["blank_argmax_fraction"] for row in rows])),
        "mean_decoded_length_ratio": float(np.mean([row["decoded_length_ratio"] for row in rows])),
    }


def main() -> None:
    seed = 33
    torch.set_num_threads(1)
    dataset = NeuralTextDataset("data/processed/spanishbcbl_subset/manifest.jsonl", None)
    by_id = {dataset[index]["meta"]["id"]: dataset[index] for index in range(len(dataset))}
    train_items = [by_id[trial_id] for trial_id in TRAIN_IDS]
    eval_items = [by_id[trial_id] for trial_id in EVAL_IDS]
    configs = {
        "A_baseline": {"learning_rate": 0.01, "clip_norm": None},
        "B_lr_0.001": {"learning_rate": 0.001, "clip_norm": None},
        "C_lr_0.0001": {"learning_rate": 0.0001, "clip_norm": None},
        "D_lr_0.001_clip_1": {"learning_rate": 0.001, "clip_norm": 1.0},
        "E_lr_0.001_no_clip": {"learning_rate": 0.001, "clip_norm": None},
    }
    results = {
        "metadata": {
            "manifest": "data/processed/spanishbcbl_subset/manifest.jsonl",
            "train_trial_ids": TRAIN_IDS,
            "evaluation_trial_ids": EVAL_IDS,
            "seed": seed,
            "model": "Conv1D + BiGRU + Linear + CTC",
            "vocabulary": "SpanishBCBL verified 30-class vocabulary",
            "preprocessing": "existing corrected continuous preprocessing manifest",
            "epochs": 300,
            "same_split_all_configs": True,
        },
        "configurations": {},
    }
    for name, config in configs.items():
        torch.manual_seed(seed)
        np.random.seed(seed)
        model, final_loss = train_one(
            train_items, config["learning_rate"], config["clip_norm"]
        )
        train_metrics = evaluate_items(model, train_items)
        eval_metrics = evaluate_items(model, eval_items)
        results["configurations"][name] = {
            **config,
            "optimizer": "Adam",
            "final_training_ctc_loss": final_loss,
            "training": train_metrics,
            "evaluation": eval_metrics,
            "all_metrics_finite": all(
                np.isfinite(value)
                for value in [
                    final_loss,
                    train_metrics["mean_cer"],
                    train_metrics["mean_wer"],
                    eval_metrics["mean_cer"],
                    eval_metrics["mean_wer"],
                    train_metrics["mean_blank_probability"],
                    eval_metrics["mean_blank_probability"],
                ]
            ),
        }

    output = Path("results/blank_collapse_diagnostic.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    names = list(configs)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(names))
    width = 0.35
    axes[0].bar(x - width / 2, [results["configurations"][n]["training"]["mean_cer"] for n in names], width, label="train")
    axes[0].bar(x + width / 2, [results["configurations"][n]["evaluation"]["mean_cer"] for n in names], width, label="eval")
    axes[0].set_title("Mean CER")
    axes[1].bar(x - width / 2, [results["configurations"][n]["training"]["blank_argmax_fraction"] for n in names], width, label="train")
    axes[1].bar(x + width / 2, [results["configurations"][n]["evaluation"]["blank_argmax_fraction"] for n in names], width, label="eval")
    axes[1].set_title("Blank argmax fraction")
    axes[2].bar(x - width / 2, [results["configurations"][n]["training"]["mean_decoded_length_ratio"] for n in names], width, label="train")
    axes[2].bar(x + width / 2, [results["configurations"][n]["evaluation"]["mean_decoded_length_ratio"] for n in names], width, label="eval")
    axes[2].set_title("Decoded/target length")
    for axis in axes:
        axis.set_xticks(x, names, rotation=35, ha="right")
        axis.legend()
    figure.tight_layout()
    figure_path = Path("results/figures/debug/blank_collapse_diagnostic.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=120)
    plt.close(figure)
    print(json.dumps({
        name: {
            "train_cer": row["training"]["mean_cer"],
            "eval_cer": row["evaluation"]["mean_cer"],
            "train_blank": row["training"]["blank_argmax_fraction"],
            "eval_blank": row["evaluation"]["blank_argmax_fraction"],
        }
        for name, row in results["configurations"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
