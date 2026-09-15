"""Phase 5C: seed stability for the clipped real-data configuration."""

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
SEEDS = (33, 123, 777)


def evaluate_items(model: ConvCTC, items: list[dict]) -> dict:
    rows = []
    for item in items:
        model.eval()
        with torch.no_grad():
            logits = model(item["signal"].unsqueeze(0))[0]
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
            "decoded_length": len(decoded),
            "target_length": len(target),
            "decoded_length_ratio": len(decoded) / max(len(target), 1),
            "blank_argmax_fraction": float((ids == BLANK_ID).float().mean()),
        })
    return {
        "per_trial": rows,
        "mean_cer": float(np.mean([row["cer"] for row in rows])),
        "mean_wer": float(np.mean([row["wer"] for row in rows])),
        "mean_blank_argmax_fraction": float(np.mean([row["blank_argmax_fraction"] for row in rows])),
        "mean_decoded_length_ratio": float(np.mean([row["decoded_length_ratio"] for row in rows])),
    }


def train_seed(train_items: list[dict], seed: int) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    model = ConvCTC(306, len(VOCAB))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    batch = collate_batch(train_items)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
    train = evaluate_items(model, train_items)
    return {
        "seed": seed,
        "final_training_ctc_loss": final_loss,
        "training": train,
    }, model


def main() -> None:
    manifest = "data/processed/spanishbcbl_subset/manifest.jsonl"
    dataset = NeuralTextDataset(manifest, None)
    by_id = {dataset[index]["meta"]["id"]: dataset[index] for index in range(len(dataset))}
    train_items = [by_id[trial_id] for trial_id in TRAIN_IDS]
    eval_items = [by_id[trial_id] for trial_id in EVAL_IDS]
    results = {
        "metadata": {
            "manifest": manifest,
            "train_trial_ids": TRAIN_IDS,
            "evaluation_trial_ids": EVAL_IDS,
            "preprocessing": "existing corrected continuous preprocessing manifest",
            "vocabulary": "SpanishBCBL verified 30-class vocabulary",
            "model": "Conv1D + BiGRU + Linear + CTC",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "gradient_clip_max_norm": 1.0,
            "epochs": 300,
            "seeds_requested": list(SEEDS),
        },
        "seeds": {},
    }
    for seed in SEEDS:
        seed_result, model = train_seed(train_items, seed)
        evaluation = evaluate_items(model, eval_items)
        seed_result["evaluation"] = evaluation
        seed_result["train_eval_cer_gap"] = (
            evaluation["mean_cer"] - seed_result["training"]["mean_cer"]
        )
        seed_result["all_metrics_finite"] = all(
            np.isfinite(value)
            for value in [
                seed_result["final_training_ctc_loss"],
                seed_result["training"]["mean_cer"],
                seed_result["training"]["mean_wer"],
                evaluation["mean_cer"],
                evaluation["mean_wer"],
                seed_result["train_eval_cer_gap"],
            ]
        )
        results["seeds"][str(seed)] = seed_result

    train_cers = [row["training"]["mean_cer"] for row in results["seeds"].values()]
    eval_cers = [row["evaluation"]["mean_cer"] for row in results["seeds"].values()]
    results["summary"] = {
        "train_cer_mean": float(np.mean(train_cers)),
        "train_cer_std": float(np.std(train_cers)),
        "evaluation_cer_mean": float(np.mean(eval_cers)),
        "evaluation_cer_std": float(np.std(eval_cers)),
        "all_metrics_finite": all(row["all_metrics_finite"] for row in results["seeds"].values()),
    }
    output = Path("results/seed_stability_diagnostic.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    names = [str(seed) for seed in SEEDS]
    x = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(x - 0.18, train_cers, width=0.36, label="train CER")
    axis.bar(x + 0.18, eval_cers, width=0.36, label="evaluation CER")
    axis.set_xticks(x, names)
    axis.set_xlabel("seed")
    axis.set_ylabel("mean CER")
    axis.set_title("Seed stability: train vs evaluation CER")
    axis.legend()
    figure.tight_layout()
    figure_path = Path("results/figures/debug/seed_stability_diagnostic.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=120)
    plt.close(figure)
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
