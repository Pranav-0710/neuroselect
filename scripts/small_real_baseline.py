"""Phase 5A: small single-subject SpanishBCBL development baseline."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from neuroselect.data import NeuralTextDataset, collate_batch
from neuroselect.metrics import cer, wer
from neuroselect.models import ConvCTC
from neuroselect.vocab_spanishbcbl import BLANK_ID, VOCAB, decode_ctc, encode_text


def train_model(items: list[dict], seed: int = 33, epochs: int = 300) -> ConvCTC:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    model = ConvCTC(306, len(VOCAB))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    model.train()
    for _ in range(epochs):
        batch = collate_batch(items)
        logits = model(batch["signals"])
        loss = loss_fn(
            logits.log_softmax(-1).transpose(0, 1),
            batch["targets"],
            batch["input_lengths"],
            batch["target_lengths"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def evaluate(model: ConvCTC, item: dict) -> dict:
    model.eval()
    signal = item["signal"].unsqueeze(0)
    with torch.no_grad():
        logits = model(signal)[0]
        probabilities = logits.softmax(-1)
        argmax_ids = probabilities.argmax(-1).tolist()
    target = item["meta"]["text"]
    decoded = decode_ctc(argmax_ids)
    return {
        "trial_id": item["meta"]["id"],
        "target": target,
        "decoded": decoded,
        "cer": cer(target, decoded),
        "wer": wer(target, decoded),
        "target_length": len(target),
        "decoded_length": len(decoded),
        "mean_blank_probability": float(probabilities[:, BLANK_ID].mean()),
        "blank_argmax_fraction": float(
            (probabilities.argmax(-1) == BLANK_ID).float().mean()
        ),
    }


def main() -> None:
    manifest = "data/processed/spanishbcbl_subset/manifest.jsonl"
    dataset = NeuralTextDataset(manifest, None)
    items = [dataset[index] for index in range(len(dataset))]
    train_ids = [f"{trial}.0_S22_1_block1" for trial in range(2, 8)]
    eval_ids = [f"{trial}.0_S22_1_block1" for trial in range(8, 10)]
    by_id = {item["meta"]["id"]: item for item in items}
    train_items = [by_id[trial_id] for trial_id in train_ids]
    eval_items = [by_id[trial_id] for trial_id in eval_ids]
    if len({item["meta"]["text"] for item in train_items} & {item["meta"]["text"] for item in eval_items}):
        raise RuntimeError("Identical sentence appears across train/evaluation")

    model = train_model(train_items)
    per_trial = [evaluate(model, item) for item in eval_items]
    mean_cer = float(np.mean([row["cer"] for row in per_trial]))
    median_cer = float(np.median([row["cer"] for row in per_trial]))
    mean_wer = float(np.mean([row["wer"] for row in per_trial]))
    median_wer = float(np.median([row["wer"] for row in per_trial]))

    all_train_text = "".join(item["meta"]["text"] for item in train_items)
    most_common = Counter(all_train_text.replace(" ", "")).most_common(1)[0][0]
    trivial = {
        "empty_prediction": {
            "per_trial_cer": [cer(row["target"], "") for row in per_trial],
            "per_trial_wer": [wer(row["target"], "") for row in per_trial],
            "mean_cer": float(np.mean([cer(row["target"], "") for row in per_trial])),
            "mean_wer": float(np.mean([wer(row["target"], "") for row in per_trial])),
        },
        "most_common_character": {
            "character": most_common,
            "per_trial_cer": [
                cer(row["target"], most_common * len(row["target"])) for row in per_trial
            ],
            "per_trial_wer": [
                wer(row["target"], most_common * len(row["target"])) for row in per_trial
            ],
        },
    }
    trivial["most_common_character"]["mean_cer"] = float(
        np.mean(trivial["most_common_character"]["per_trial_cer"])
    )
    trivial["most_common_character"]["mean_wer"] = float(
        np.mean(trivial["most_common_character"]["per_trial_wer"])
    )
    result = {
        "dataset": "SpanishBCBL/DECOMEG",
        "baseline_type": "small single-subject development-set baseline",
        "manifest": manifest,
        "preprocessing_version": "continuous filter/resample/RobustScaler then sentence crop, baseline correction, clamp",
        "vocabulary_version": "SpanishBCBL verified 30-class vocabulary",
        "seed": 33,
        "train_trial_ids": train_ids,
        "evaluation_trial_ids": eval_ids,
        "duplicate_sentences_across_split": False,
        "model": "Conv1D + BiGRU + Linear + CTC",
        "channels": 306,
        "learning_rate": 1e-2,
        "optimizer": "Adam",
        "epochs": 300,
        "per_trial": per_trial,
        "aggregate": {
            "mean_cer": mean_cer,
            "median_cer": median_cer,
            "mean_wer": mean_wer,
            "median_wer": median_wer,
        },
        "trivial_baselines": trivial,
    }
    output = Path("results/small_real_baseline.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar([row["trial_id"] for row in per_trial], [row["cer"] for row in per_trial])
    axis.set_title("Small real-data baseline evaluation CER")
    axis.set_ylabel("CER")
    figure.tight_layout()
    figure_path = Path("results/figures/debug/small_real_baseline_predictions.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=120)
    plt.close(figure)
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
