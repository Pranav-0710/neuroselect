"""Steps 5 and 6: event-sequence CTC on the expanded S22 data.

One process trains one (split, seed, mode) run of the unchanged
Conv1D + BiGRU + Linear + CTC model on `official_v1_event_sequence` inputs and
writes a single JSON result. `scripts/s22_expanded_ctc_aggregate.py` combines
the runs into the phase artifacts.

Sequences are grouped into batches of *identical* length, so no zero padding is
ever introduced and every forward pass is numerically identical to the
one-sequence-at-a-time protocol used by the 8-trial baseline. The only protocol
difference is that gradients are averaged over a same-length group before the
optimizer step; see `--batch 1` to reproduce the 8-trial protocol exactly.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from neuroselect.metrics import cer, wer
from neuroselect.models import ConvCTC
from neuroselect.vocab_spanishbcbl import BLANK_ID, VOCAB, decode_ctc, encode_text

ROOT = Path(__file__).resolve().parents[1]
TENSOR_PATH = ROOT / "data/processed/s22_official_v1/events.npy"
INDEX_PATH = ROOT / "data/processed/s22_official_v1/index.jsonl"
MANIFEST_DIR = ROOT / "data/manifests"
RUN_DIR = ROOT / "results/runs/s22_expanded_ctc"

SPLIT_FILES = {
    "A": "split_A_same_session.json",
    "B": "split_B_same_session.json",
    "C": "split_C_sentence_disjoint.json",
    "D": "split_D_cross_session_sentence_disjoint.json",
    "E": "split_E_cross_session_sentence_disjoint.json",
    "F": "split_F_cross_session_sentence_overlap.json",
    "G": "split_G_cross_session_sentence_overlap.json",
}
EPOCHS = 300
LEARNING_RATE = 0.001
CLIP_NORM = 1.0
PERMUTATION_SEED = 2026
EVENT_SAMPLES = 25


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_sentences() -> dict[str, dict]:
    """Group the event index into per-sentence row lists (no tensor copies)."""
    sentences: dict[str, dict] = {}
    for line in INDEX_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        record = sentences.setdefault(row["sentence_UID"], {
            "sentence_UID": row["sentence_UID"],
            "session": row["session"],
            "block": row["block"],
            "list_id": row["list_id"],
            "unique_sentence_group_id": row["unique_sentence_group_id"],
            "trial_id": row["trial_id"],
            "rows": [],
            "labels": [],
        })
        record["rows"].append((row["event_position"], row["row"]))
        record["labels"].append((row["event_position"], row["label"]))
    for record in sentences.values():
        record["rows"] = [row for _, row in sorted(record["rows"])]
        record["target"] = "".join(label for _, label in sorted(record["labels"]))
        record["events"] = len(record["rows"])
        del record["labels"]
    return sentences


def derangement(keys: list[str], seed: int) -> tuple[dict[str, str], int]:
    """Deterministic permutation with no fixed point (same rule as task 5V)."""
    rng = np.random.default_rng(seed)
    draws = 0
    while True:
        draws += 1
        order = [keys[index] for index in rng.permutation(len(keys))]
        if all(source != target for source, target in zip(keys, order)):
            return dict(zip(keys, order)), draws


def sequence_of(slab: np.ndarray, record: dict) -> np.ndarray:
    return slab[record["rows"]].reshape(record["events"] * EVENT_SAMPLES, -1)


def repetition_statistics(text: str) -> dict:
    if not text:
        return {"longest_run": 0, "mean_run_length": 0.0, "adjacent_repeat_fraction": 0.0,
                "unique_characters": 0, "most_frequent_character_fraction": 0.0}
    runs, current = [], 1
    for previous, character in zip(text, text[1:]):
        if character == previous:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)
    counts = np.unique(np.asarray(list(text)), return_counts=True)[1]
    return {
        "longest_run": int(max(runs)),
        "mean_run_length": float(np.mean(runs)),
        "adjacent_repeat_fraction": float(sum(1 for a, b in zip(text, text[1:]) if a == b) / max(len(text) - 1, 1)),
        "unique_characters": int(len(counts)),
        "most_frequent_character_fraction": float(counts.max() / len(text)),
    }


def predict(model: ConvCTC, slab: np.ndarray, record: dict, target: str) -> dict:
    model.eval()
    with torch.no_grad():
        sequence = torch.from_numpy(np.ascontiguousarray(sequence_of(slab, record))).float()
        logits = model(sequence[None])[0]
        probabilities = torch.log_softmax(logits, dim=-1).exp()
        argmax_ids = logits.argmax(dim=-1).cpu().tolist()
    decoded = decode_ctc(argmax_ids)
    return {
        "sentence_UID": record["sentence_UID"],
        "session": record["session"],
        "block": f"session{record['session']}/{record['block']}",
        "list_id": record["list_id"],
        "trial_id": record["trial_id"],
        "events": record["events"],
        "T": record["events"] * EVENT_SAMPLES,
        "target": target,
        "decoded": decoded,
        "cer": cer(target, decoded),
        "wer": wer(target, decoded),
        "target_length": len(target),
        "decoded_length": len(decoded),
        "decoded_target_ratio": len(decoded) / max(len(target), 1),
        "blank_argmax_fraction": float(np.mean(np.asarray(argmax_ids) == BLANK_ID)),
        "mean_blank_probability": float(probabilities[:, BLANK_ID].mean().item()),
        **repetition_statistics(decoded),
    }


def aggregate(rows: list[dict]) -> dict:
    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))
    by_block: dict[str, list[dict]] = {}
    for row in rows:
        by_block.setdefault(row["block"], []).append(row)
    return {
        "sentences": len(rows),
        "mean_cer": mean("cer"),
        "mean_wer": mean("wer"),
        "median_cer": float(np.median([row["cer"] for row in rows])),
        "mean_blank_argmax_fraction": mean("blank_argmax_fraction"),
        "mean_blank_probability": mean("mean_blank_probability"),
        "mean_decoded_target_ratio": mean("decoded_target_ratio"),
        "mean_decoded_length": mean("decoded_length"),
        "mean_target_length": mean("target_length"),
        "mean_longest_run": mean("longest_run"),
        "mean_adjacent_repeat_fraction": mean("adjacent_repeat_fraction"),
        "mean_most_frequent_character_fraction": mean("most_frequent_character_fraction"),
        "mean_unique_decoded_characters": mean("unique_characters"),
        "empty_decodes": int(sum(1 for row in rows if not row["decoded"])),
        "per_block": {
            block: {
                "sentences": len(block_rows),
                "mean_cer": float(np.mean([row["cer"] for row in block_rows])),
                "mean_wer": float(np.mean([row["wer"] for row in block_rows])),
                "mean_blank_argmax_fraction": float(np.mean([row["blank_argmax_fraction"] for row in block_rows])),
                "mean_decoded_target_ratio": float(np.mean([row["decoded_target_ratio"] for row in block_rows])),
            }
            for block, block_rows in sorted(by_block.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=sorted(SPLIT_FILES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("real", "control"), default="real")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=0,
                        help="0 = one batch per identical-length group; 1 = one sequence per step")
    parser.add_argument("--threads", type=int, default=1)
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)

    manifest = json.loads((MANIFEST_DIR / SPLIT_FILES[arguments.split]).read_text(encoding="utf-8"))
    sentences = load_sentences()
    slab = np.load(TENSOR_PATH, mmap_mode="r")

    train_records = [sentences[uid] for uid in manifest["train_sentence_UIDs"]]
    test_records = [sentences[uid] for uid in manifest["test_sentence_UIDs"]]
    targets = {record["sentence_UID"]: record["target"] for record in train_records}

    permutation_report = None
    if arguments.mode == "control":
        keys = [record["sentence_UID"] for record in train_records]
        mapping, draws = derangement(keys, PERMUTATION_SEED)
        targets = {source: sentences[target]["target"] for source, target in mapping.items()}
        identical = sum(1 for source, target in mapping.items() if sentences[source]["target"] == sentences[target]["target"])
        permutation_report = {
            "shuffle_seed": PERMUTATION_SEED,
            "draws_until_derangement": draws,
            "training_sentences_permuted": len(mapping),
            "fixed_points": sum(1 for source, target in mapping.items() if source == target),
            "targets_identical_by_text_despite_permutation": identical,
            "signals_untouched": True,
            "sequence_lengths_unchanged": True,
            "evaluation_targets_correctly_paired": True,
            "same_permutation_for_all_model_seeds": True,
            "mapping_examples": [
                {"sentence_UID": source, "original_target": sentences[source]["target"],
                 "assigned_target": sentences[target]["target"]}
                for source, target in list(mapping.items())[:5]
            ],
        }

    seed_everything(arguments.seed)
    model = ConvCTC(306, len(VOCAB))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)

    encoded = {
        record["sentence_UID"]: torch.tensor(encode_text(targets[record["sentence_UID"]]), dtype=torch.long)
        for record in train_records
    }
    if arguments.batch == 1:
        groups = [[record] for record in train_records]
    else:
        buckets: dict[int, list[dict]] = {}
        for record in train_records:
            buckets.setdefault(record["events"], []).append(record)
        groups = [buckets[length] for length in sorted(buckets)]
        if arguments.batch > 1:
            groups = [group[i:i + arguments.batch] for group in groups for i in range(0, len(group), arguments.batch)]

    # Every group is fixed for the whole run, so stack each one once. This holds
    # exactly the same bytes as caching per sentence would, but removes a fresh
    # (B, T, 306) allocation on every one of the 300 x len(groups) steps.
    batches = []
    for group in groups:
        signals = torch.stack([
            torch.from_numpy(np.ascontiguousarray(sequence_of(slab, record))).float()
            for record in group
        ])
        batches.append((
            signals,
            torch.cat([encoded[record["sentence_UID"]] for record in group]),
            torch.tensor([record["events"] * EVENT_SAMPLES for record in group], dtype=torch.long),
            torch.tensor([len(encoded[record["sentence_UID"]]) for record in group], dtype=torch.long),
        ))

    rng = np.random.default_rng(arguments.seed)
    losses = []
    started = time.time()
    for epoch in range(arguments.epochs):
        model.train()
        epoch_losses = []
        for index in rng.permutation(len(batches)):
            signals, target_ids, input_lengths, target_lengths = batches[index]
            optimizer.zero_grad(set_to_none=True)
            logits = model(signals)
            log_probabilities = torch.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = loss_function(log_probabilities, target_ids, input_lengths, target_lengths)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite CTC loss at seed {arguments.seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_losses)))
        if (epoch + 1) % 25 == 0:
            print(f"epoch {epoch + 1}/{arguments.epochs} loss={losses[-1]:.4f} "
                  f"elapsed={time.time() - started:.0f}s", flush=True)
    elapsed = time.time() - started
    del batches

    train_predictions = [predict(model, slab, record, targets[record["sentence_UID"]]) for record in train_records]
    evaluation_predictions = [predict(model, slab, record, record["target"]) for record in test_records]

    result = {
        "split": arguments.split,
        "split_name": manifest["split_name"],
        "classification": manifest["classification"],
        "mode": arguments.mode,
        "seed": arguments.seed,
        "model": "Conv1D + BiGRU + Linear + CTC (unchanged ConvCTC)",
        "representation": "official_v1_event_sequence",
        "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE,
                      "gradient_clip_max_norm": CLIP_NORM, "epochs": arguments.epochs},
        "batching": {
            "rule": "one batch per identical-length group" if arguments.batch == 0 else f"max {arguments.batch} per batch",
            "zero_padding": False,
            "groups_per_epoch": len(groups),
            "group_sizes": sorted(len(group) for group in groups),
            "updates_per_epoch": len(groups),
            "total_updates": len(groups) * arguments.epochs,
        },
        "data": {
            "train_sentences": len(train_records),
            "test_sentences": len(test_records),
            "train_events": sum(record["events"] for record in train_records),
            "test_events": sum(record["events"] for record in test_records),
            "train_blocks": sorted({f"session{r['session']}/{r['block']}" for r in train_records}),
            "test_blocks": sorted({f"session{r['session']}/{r['block']}" for r in test_records}),
        },
        "target_permutation": permutation_report,
        "initial_ctc_loss": losses[0],
        "final_ctc_loss": losses[-1],
        "minimum_ctc_loss": min(losses),
        "loss_history": losses,
        "train_aggregate": aggregate(train_predictions),
        "evaluation_aggregate": aggregate(evaluation_predictions),
        "train_predictions": train_predictions,
        "evaluation_predictions": evaluation_predictions,
        "training_seconds": elapsed,
        "verification": {
            "official_v1_preprocessing": True,
            "historical_preprocessing_used": False,
            "architecture_changed": False,
            "decoder_changed": False,
            "vocabulary_changed": False,
            "zero_padding_used": False,
            "llm_used": False,
            "evidence_selector_used": False,
        },
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUN_DIR / f"{arguments.split}_{arguments.mode}_seed{arguments.seed}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "run": out_path.name,
        "train_cer": result["train_aggregate"]["mean_cer"],
        "eval_cer": result["evaluation_aggregate"]["mean_cer"],
        "eval_blank_fraction": result["evaluation_aggregate"]["mean_blank_argmax_fraction"],
        "final_loss": result["final_ctc_loss"],
        "seconds": round(elapsed, 1),
    }))


if __name__ == "__main__":
    main()
