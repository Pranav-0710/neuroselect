"""5X: official event extraction and sentence-overlap audit across all S22 blocks.

Runs the official Pinet2024Meg study over the four S22 typing blocks, reports
real event/sentence counts per block, and measures sentence overlap between
blocks and sessions. No decoder, classifier or training is involved.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import studies  # noqa: F401  - registers Pinet2024Meg
from brain2qwerty_v1.transforms import SpanishBCBLPreprocessing
from neuralset.events import Study

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/raw/spanishbcbl_s22"
EVENTS_PATH = DATA_ROOT / "events_clean_all_blocks.pkl"
OUT_PATH = ROOT / "results/s22_official_event_extraction.json"
FIGURE_PATH = ROOT / "results/figures/debug/s22_sentence_overlap.png"
HISTORICAL_EVENTS = DATA_ROOT / "events_clean.pkl"


def normalize(text: object) -> str:
    return str(text).strip().lower()


def main() -> None:
    historical_before = HISTORICAL_EVENTS.stat().st_mtime_ns if HISTORICAL_EVENTS.exists() else None

    study = Study(name="Pinet2024Meg", path=DATA_ROOT)
    events = SpanishBCBLPreprocessing()._run(study.run())
    events.to_pickle(EVENTS_PATH)

    keys = ["subject", "session", "task"] if "task" in events.columns else ["subject", "session"]
    blocks = {}
    for key, frame in events.groupby(keys, dropna=False):
        label = "/".join(str(part) for part in (key if isinstance(key, tuple) else (key,)))
        sentences = frame[frame["type"] == "Sentence"]
        production = sentences[sentences["is_percep"] == False] if "is_percep" in sentences.columns else sentences  # noqa: E712
        texts = [normalize(t) for t in production["text"].dropna()] if "text" in production.columns else []
        typed = [normalize(t) for t in production["sentence_typed"].dropna()] if "sentence_typed" in production.columns else []
        blocks[label] = {
            "events_total": int(len(frame)),
            "keystrokes": int((frame["type"] == "Keystroke").sum()),
            "words": int((frame["type"] == "Word").sum()),
            "sentences_all": int(len(sentences)),
            "sentences_production": int(len(production)),
            "unique_presented_sentences": len(set(texts)),
            "unique_typed_sentences": len(set(typed)),
            "presented_sentences": sorted(set(texts)),
            "typed_sentences": sorted(set(typed)),
            "trial_ids": sorted(int(t) for t in production["trial_id"].dropna().unique()) if "trial_id" in production.columns else [],
        }

    overlap = []
    labels = sorted(blocks)
    for left, right in itertools.combinations(labels, 2):
        a = set(blocks[left]["presented_sentences"])
        b = set(blocks[right]["presented_sentences"])
        overlap.append({
            "block_a": left,
            "block_b": right,
            "unique_a": len(a),
            "unique_b": len(b),
            "shared_presented_sentences": len(a & b),
            "jaccard": len(a & b) / len(a | b) if a | b else 0.0,
            "examples_shared": sorted(a & b)[:3],
        })

    all_texts = [text for block in blocks.values() for text in block["presented_sentences"]]
    artifact = {
        "task": "5X official event extraction and sentence overlap (no decoder)",
        "data_root": str(DATA_ROOT.relative_to(ROOT)),
        "events_file": str(EVENTS_PATH.relative_to(ROOT)),
        "historical_events_file_untouched": historical_before == (HISTORICAL_EVENTS.stat().st_mtime_ns if HISTORICAL_EVENTS.exists() else None),
        "event_columns": list(events.columns),
        "total_events": int(len(events)),
        "blocks": blocks,
        "pairwise_sentence_overlap": overlap,
        "corpus": {
            "sentence_instances": len(all_texts),
            "unique_sentences_across_all_blocks": len(set(all_texts)),
        },
        "decoder_or_classifier_run": False,
        "training_run": False,
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    figure, (left_axis, right_axis) = plt.subplots(1, 2, figsize=(13, 5))
    left_axis.bar(labels, [blocks[label]["keystrokes"] for label in labels], color="#3a6ea5")
    for index, label in enumerate(labels):
        left_axis.text(index, blocks[label]["keystrokes"], str(blocks[label]["keystrokes"]), ha="center", va="bottom", fontsize=8)
    left_axis.set_ylabel("keystroke events")
    left_axis.set_title("Official keystrokes per block")
    left_axis.tick_params(axis="x", labelrotation=30)

    matrix = pd.DataFrame(0, index=labels, columns=labels)
    for label in labels:
        matrix.loc[label, label] = blocks[label]["unique_presented_sentences"]
    for row in overlap:
        matrix.loc[row["block_a"], row["block_b"]] = row["shared_presented_sentences"]
        matrix.loc[row["block_b"], row["block_a"]] = row["shared_presented_sentences"]
    image = right_axis.imshow(matrix.values, cmap="Blues")
    right_axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
    right_axis.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            right_axis.text(j, i, int(matrix.values[i, j]), ha="center", va="center", fontsize=9,
                            color="white" if matrix.values[i, j] > matrix.values.max() / 2 else "black")
    right_axis.set_title("Shared presented sentences (diagonal = unique count)")
    figure.colorbar(image, ax=right_axis, shrink=0.8)
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)

    print(json.dumps({"blocks": {k: {m: v[m] for m in ("events_total", "keystrokes", "words", "sentences_production", "unique_presented_sentences")} for k, v in blocks.items()},
                      "overlap": overlap, "corpus": artifact["corpus"]}, indent=2))


if __name__ == "__main__":
    main()
