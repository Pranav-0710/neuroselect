"""Phase 5E: leave-one-trial-out generalization diagnostic."""

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


TRIAL_IDS = [f"{index}.0_S22_1_block1" for index in range(2, 10)]


def evaluate(model: ConvCTC, items: list[dict]) -> dict:
    rows = []
    for item in items:
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
            "target_length": len(target),
            "decoded_length": len(decoded),
            "decoded_length_ratio": len(decoded) / max(len(target), 1),
            "blank_argmax_fraction": float((ids == BLANK_ID).float().mean()),
            "mean_blank_probability": float(probabilities[:, BLANK_ID].mean()),
        })
    return rows


def train_fold(train_items: list[dict]) -> tuple[ConvCTC, float]:
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
    return model, final_loss


def main() -> None:
    torch.set_num_threads(1)
    manifest = "data/processed/spanishbcbl_subset/manifest.jsonl"
    dataset = NeuralTextDataset(manifest, None)
    by_id = {dataset[index]["meta"]["id"]: dataset[index] for index in range(len(dataset))}
    audit = json.loads(Path("results/trial_distribution_audit.json").read_text())
    statistics = {row["trial_id"]: row for row in audit["per_trial"]}
    folds = []
    for held_out_id in TRIAL_IDS:
        torch.manual_seed(33)
        np.random.seed(33)
        train_ids = [trial_id for trial_id in TRIAL_IDS if trial_id != held_out_id]
        train_items = [by_id[trial_id] for trial_id in train_ids]
        held_out_item = by_id[held_out_id]
        model, final_loss = train_fold(train_items)
        training_rows = evaluate(model, train_items)
        held_out_row = evaluate(model, [held_out_item])[0]
        mean_train_cer = float(np.mean([row["cer"] for row in training_rows]))
        mean_train_wer = float(np.mean([row["wer"] for row in training_rows]))
        mean_train_blank = float(np.mean([row["blank_argmax_fraction"] for row in training_rows]))
        folds.append({
            "held_out_trial_id": held_out_id,
            "train_trial_ids": train_ids,
            "held_out": held_out_row,
            "final_training_ctc_loss": final_loss,
            "mean_training_cer": mean_train_cer,
            "mean_training_wer": mean_train_wer,
            "mean_training_blank_argmax_fraction": mean_train_blank,
            "cer_gap": held_out_row["cer"] - mean_train_cer,
            "all_metrics_finite": all(
                np.isfinite(value)
                for value in [final_loss, held_out_row["cer"], held_out_row["wer"], mean_train_cer, mean_train_wer]
            ),
            "trial_statistics": statistics[held_out_id],
        })

    held_out_cers = [fold["held_out"]["cer"] for fold in folds]
    held_out_wers = [fold["held_out"]["wer"] for fold in folds]
    result = {
        "configuration": {
            "manifest": manifest,
            "seed": 33,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "gradient_clip_max_norm": 1.0,
            "epochs": 300,
            "model": "Conv1D + BiGRU + Linear + CTC",
            "preprocessing": "existing corrected continuous preprocessing manifest",
            "vocabulary": "SpanishBCBL verified 30-class vocabulary",
            "decoder": "existing SpanishBCBL greedy CTC decoder",
        },
        "folds": folds,
        "summary": {
            "mean_held_out_cer": float(np.mean(held_out_cers)),
            "median_held_out_cer": float(np.median(held_out_cers)),
            "mean_held_out_wer": float(np.mean(held_out_wers)),
            "median_held_out_wer": float(np.median(held_out_wers)),
            "best_trial": folds[int(np.argmin(held_out_cers))]["held_out_trial_id"],
            "worst_trial": folds[int(np.argmax(held_out_cers))]["held_out_trial_id"],
            "better_transfer_trials_cer_lt_0_75": [
                fold["held_out_trial_id"] for fold in folds if fold["held_out"]["cer"] < 0.75
            ],
            "poor_transfer_trials_cer_ge_0_75": [
                fold["held_out_trial_id"] for fold in folds if fold["held_out"]["cer"] >= 0.75
            ],
            "all_metrics_finite": all(fold["all_metrics_finite"] for fold in folds),
        },
    }
    output = Path("results/leave_one_trial_out.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    x = np.arange(2, 10)
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(x, held_out_cers)
    axis.set_xlabel("held-out trial")
    axis.set_ylabel("CER")
    axis.set_title("Leave-one-trial-out held-out CER")
    figure.tight_layout()
    figure.savefig("results/figures/debug/leave_one_trial_out_cer.png", dpi=120)
    plt.close(figure)

    ratios = [fold["trial_statistics"]["statistics"]["T_over_target"] for fold in folds]
    distances = [fold["trial_statistics"]["distance_to_pooled_train"] for fold in folds]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(ratios, held_out_cers)
    for fold, ratio, score in zip(folds, ratios, held_out_cers):
        axis.annotate(fold["held_out_trial_id"].split(".", 1)[0], (ratio, score))
    axis.set_xlabel("T / target length")
    axis.set_ylabel("held-out CER")
    figure.tight_layout()
    figure.savefig("results/figures/debug/leave_one_trial_out_geometry_vs_cer.png", dpi=120)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(distances, held_out_cers)
    for fold, distance, score in zip(folds, distances, held_out_cers):
        axis.annotate(fold["held_out_trial_id"].split(".", 1)[0], (distance, score))
    axis.set_xlabel("distance to pooled training distribution")
    axis.set_ylabel("held-out CER")
    figure.tight_layout()
    figure.savefig("results/figures/debug/leave_one_trial_out_distribution_vs_cer.png", dpi=120)
    plt.close(figure)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
