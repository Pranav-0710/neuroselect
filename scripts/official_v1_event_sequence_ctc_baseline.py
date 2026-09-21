"""Controlled CTC baseline on concatenated official-v1 event tensors."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from neuroselect.metrics import cer, wer
from neuroselect.models import ConvCTC
from neuroselect.official_v1_preprocessing import NeuroSelectOfficialV1EventPreprocessing
from neuroselect.vocab_spanishbcbl import BLANK_ID, VOCAB, decode_ctc, encode_text

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/spanishbcbl_s22/MEG/FIF/22_9788/231214/block1.fif"
PARITY_PATH = ROOT / "results/eight_trial_official_v1_parity.json"
OUT_PATH = ROOT / "results/official_v1_event_sequence_ctc_baseline.json"
FIGURE_PATH = ROOT / "results/figures/debug/official_v1_event_sequence_ctc_baseline.png"
PREDICTION_FIGURE_PATH = ROOT / "results/figures/debug/official_v1_event_sequence_predictions.png"
SEEDS = (33, 123, 777)
TRAIN_TRIALS = tuple(range(2, 8))
EVAL_TRIALS = (8, 9)
EPOCHS = 300
LEARNING_RATE = 0.001
CLIP_NORM = 1.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def build_items(processor: NeuroSelectOfficialV1EventPreprocessing) -> list[dict]:
    artifact = json.loads(PARITY_PATH.read_text(encoding="utf-8"))
    items = []
    for trial in artifact["trial_identity_and_event_audit"]:
        trial_number = int(trial["trial_number"])
        events = sorted(trial["events"], key=lambda row: int(row["event_index"]))
        event_tensors = [processor.extract_event(float(row["timestamp_seconds"])).tensor for row in events]
        if any(tensor.shape != (25, 306) for tensor in event_tensors):
            raise RuntimeError(f"Event shape audit failed for trial {trial_number}")
        sequence = np.concatenate(event_tensors, axis=0)
        target = str(trial["sentence_typed"])
        expected_t = 25 * len(events)
        if sequence.shape != (expected_t, 306):
            raise RuntimeError(f"Sequence shape audit failed for trial {trial_number}: {sequence.shape}")
        items.append({
            "trial_number": trial_number,
            "trial_id": trial["trial_id"],
            "target": target,
            "events": len(events),
            "sequence": sequence,
            "target_ids": torch.tensor(encode_text(target), dtype=torch.long),
            "official_event_order": True,
            "no_gaps": True,
            "zero_padded": False,
        })
    return items


def predict(model: ConvCTC, item: dict) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(item["sequence"])[None].float())[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        probabilities = log_probs.exp()
        argmax_ids = logits.argmax(dim=-1).cpu().tolist()
    decoded = decode_ctc(argmax_ids)
    blank_fraction = float(np.mean(np.asarray(argmax_ids) == BLANK_ID))
    most_fraction = 0.0
    if decoded:
        counts = np.unique(np.asarray(list(decoded)), return_counts=True)[1]
        most_fraction = float(counts.max() / len(decoded))
    return {
        "trial_number": item["trial_number"],
        "trial_id": item["trial_id"],
        "target": item["target"],
        "decoded": decoded,
        "cer": cer(item["target"], decoded),
        "wer": wer(item["target"], decoded),
        "target_length": len(item["target"]),
        "decoded_length": len(decoded),
        "decoded_target_ratio": len(decoded) / max(len(item["target"]), 1),
        "blank_argmax_fraction": blank_fraction,
        "mean_blank_probability": float(probabilities[:, BLANK_ID].mean().item()),
        "unique_decoded_characters": len(set(decoded)),
        "most_frequent_decoded_character_fraction": most_fraction,
    }


def aggregate(rows: list[dict]) -> dict:
    return {
        "mean_cer": float(np.mean([row["cer"] for row in rows])),
        "mean_wer": float(np.mean([row["wer"] for row in rows])),
        "mean_blank_argmax_fraction": float(np.mean([row["blank_argmax_fraction"] for row in rows])),
        "mean_decoded_target_ratio": float(np.mean([row["decoded_target_ratio"] for row in rows])),
    }


def train_seed(items: list[dict], seed: int) -> dict:
    seed_everything(seed)
    train = [item for item in items if item["trial_number"] in TRAIN_TRIALS]
    evaluation = [item for item in items if item["trial_number"] in EVAL_TRIALS]
    model = ConvCTC(306, len(VOCAB))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    losses = []
    for _epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        for item in train:
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.from_numpy(item["sequence"])[None].float())
            log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
            input_lengths = torch.tensor([logits.shape[1]], dtype=torch.long)
            target_lengths = torch.tensor([len(item["target_ids"])], dtype=torch.long)
            loss = loss_fn(log_probs, item["target_ids"], input_lengths, target_lengths)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite CTC loss at seed {seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
    train_predictions = [predict(model, item) for item in train]
    eval_predictions = [predict(model, item) for item in evaluation]
    return {
        "seed": seed,
        "initial_ctc_loss": losses[0],
        "final_ctc_loss": losses[-1],
        "minimum_ctc_loss": min(losses),
        "loss_history": losses,
        "train_predictions": train_predictions,
        "evaluation_predictions": eval_predictions,
        "train_aggregate": aggregate(train_predictions),
        "evaluation_aggregate": aggregate(eval_predictions),
        "checkpoint": None,
    }


def main() -> None:
    processor = NeuroSelectOfficialV1EventPreprocessing.from_raw(RAW_PATH, recording_start_seconds=308.0)
    items = build_items(processor)
    shape_audit = []
    for item in items:
        values = item["sequence"]
        shape_audit.append({
            "trial_number": item["trial_number"],
            "trial_id": item["trial_id"],
            "target": item["target"],
            "U": item["events"],
            "T": int(values.shape[0]),
            "channels": int(values.shape[1]),
            "T_over_U": float(values.shape[0] / item["events"]),
            "expected_T": 25 * item["events"],
            "nan_count": int(np.isnan(values).sum()),
            "inf_count": int(np.isinf(values).sum()),
            "sequence_sha256": sha256(values),
        })
    seed_results = [train_seed(items, seed) for seed in SEEDS]
    train_cers = [result["train_aggregate"]["mean_cer"] for result in seed_results]
    eval_cers = [result["evaluation_aggregate"]["mean_cer"] for result in seed_results]
    figure_dir = FIGURE_PATH.parent
    figure_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(SEEDS))
    width = 0.35
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(x - width / 2, train_cers, width, label="train CER")
    axis.bar(x + width / 2, eval_cers, width, label="eval CER")
    axis.set_xticks(x, [str(seed) for seed in SEEDS])
    axis.set_xlabel("seed")
    axis.set_ylabel("CER")
    axis.set_title("Official-v1 event-sequence CTC baseline")
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)
    prediction_figure, axes = plt.subplots(len(SEEDS), 1, figsize=(12, 7), squeeze=False)
    for axis, result in zip(axes[:, 0], seed_results):
        rows = result["evaluation_predictions"]
        axis.axis("off")
        axis.set_title(f"seed {result['seed']}")
        axis.text(0.01, 0.75, "\n".join(f"trial {row['trial_number']}: {row['decoded']!r}  target={row['target']!r}" for row in rows), fontsize=9, va="top")
    prediction_figure.tight_layout()
    prediction_figure.savefig(PREDICTION_FIGURE_PATH, dpi=160)
    plt.close(prediction_figure)
    artifact = {
        "experiment": "official_v1_event_sequence_ctc_baseline",
        "official_revision": "5f9889621d0df391c5aab37c996683d308e6e926",
        "representation": "official_v1_event_sequence",
        "representation_definition": "ordered event tensors (25,306) concatenated along time with no gaps or padding",
        "data": {"trials": [row["trial_number"] for row in shape_audit], "train_trials": list(TRAIN_TRIALS), "evaluation_trials": list(EVAL_TRIALS)},
        "shape_audit": shape_audit,
        "model": "Conv1D + BiGRU + Linear + CTC (existing ConvCTC implementation)",
        "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "gradient_clip_max_norm": CLIP_NORM, "epochs": EPOCHS},
        "decoder": "existing greedy CTC decoder",
        "seeds": list(SEEDS),
        "seed_results": seed_results,
        "cross_seed": {"train_cer_mean": float(np.mean(train_cers)), "train_cer_std": float(np.std(train_cers)), "evaluation_cer_mean": float(np.mean(eval_cers)), "evaluation_cer_std": float(np.std(eval_cers))},
        "historical_continuous_sentence_baseline": {"train_cer": [0.175, 0.209, 0.122], "evaluation_cer": [1.109, 0.753, 0.764], "train_mean": 0.169, "evaluation_mean": 0.875, "comparison_caveat": "Input representation differs; this is not a preprocessing-only comparison."},
        "figures": [str(FIGURE_PATH), str(PREDICTION_FIGURE_PATH)],
        "verification": {"official_v1_used": True, "historical_preprocessing_used": False, "no_gaps": True, "zero_padding": False, "no_llm": True, "no_evidence_selector": True, "no_data_download": True},
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(OUT_PATH), "cross_seed": artifact["cross_seed"]}, indent=2))


if __name__ == "__main__":
    main()
