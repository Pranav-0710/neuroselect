"""Step 3: official-v1 event tensors for every S22 keystroke in all four blocks.

Runs the validated gated official-v1 preprocessing over each of the four typing
recordings and stores one `(25, 306)` tensor per keystroke in a single
memory-mapped float32 slab plus a JSONL index. No historical preprocessing, no
model, no training.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from neuroselect.official_v1_preprocessing import (
    CHANNEL_COUNT,
    NeuroSelectOfficialV1EventPreprocessing,
    OFFICIAL_REVISION,
)
from neuroselect.vocab_spanishbcbl import CHAR_TO_ID

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/raw/spanishbcbl_s22"
EVENTS_PATH = DATA_ROOT / "events_clean_all_blocks.pkl"
METADATA_PATH = ROOT / "data/manifests/s22_sentence_block_metadata.jsonl"
OUT_DIR = ROOT / "data/processed/s22_official_v1"
TENSOR_PATH = OUT_DIR / "events.npy"
INDEX_PATH = OUT_DIR / "index.jsonl"
OUT_PATH = ROOT / "results/s22_expanded_event_tensors.json"

EVENT_SAMPLES = 25
# session, official task name, FIF path relative to DATA_ROOT.
BLOCKS = (
    ("1", "block1", "MEG/FIF/22_9788/231214/block1.fif"),
    ("1", "block2", "MEG/FIF/22_9788/231214/block2.fif"),
    ("2", "block1", "MEG/FIF/22_9788/231222/Block1.fif"),
    ("2", "block2", "MEG/FIF/22_9788/231222/Block2.fif"),
)
# Official button tokens that are not literal characters.
BUTTON_TO_CHAR = {"<space>": " ", "<special>": "@"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def button_char(button: object) -> str:
    text = str(button)
    return BUTTON_TO_CHAR.get(text, text)


def reuse_existing() -> tuple[list[dict], list[dict], bool]:
    """Rebuild the reporting inputs from the stored slab, index and artifact."""
    previous = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    index_rows = [
        json.loads(line) for line in INDEX_PATH.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    identical = previous["verification"].get(
        "channel_names_identical_across_blocks",
        previous["verification"].get("checks", {}).get("channel_names_identical_across_blocks", True),
    )
    return index_rows, previous["blocks"], bool(identical)


def main() -> None:
    if "--verify-only" in sys.argv:
        index_rows, block_reports, channels_identical = reuse_existing()
        cursor = len(index_rows)
        total_events = cursor
        report(index_rows, block_reports, channels_identical, cursor, total_events)
        return

    events = pd.read_pickle(EVENTS_PATH)
    events["session"] = events["session"].astype(str)
    events["task"] = events["task"].astype(str)
    keystrokes = events[events["type"] == "Keystroke"]
    sentences = events[(events["type"] == "Sentence") & (events["is_percep"] == False)]  # noqa: E712
    meg_rows = events[events["type"] == "Meg"]
    metadata = {
        (record["session"], record["block"], record["trial_id"]): record
        for record in (json.loads(line) for line in METADATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip())
    }

    recording_start = {
        (str(row["session"]), str(row["task"])): float(row["start"]) for _, row in meg_rows.iterrows()
    }
    sentence_rows = {
        (str(row["session"]), str(row["task"]), int(row["trial_id"])): row
        for _, row in sentences.iterrows()
    }

    total_events = int(len(keystrokes))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slab = np.lib.format.open_memmap(
        TENSOR_PATH, mode="w+", dtype=np.float32, shape=(total_events, EVENT_SAMPLES, CHANNEL_COUNT)
    )

    index_rows: list[dict] = []
    block_reports: list[dict] = []
    channel_name_sets: dict[str, tuple[str, ...]] = {}
    cursor = 0

    for session, task, relative_fif in BLOCKS:
        fif_path = DATA_ROOT / relative_fif
        label = f"session{session}/{task}"
        start_seconds = recording_start[(session, task)]
        processor = NeuroSelectOfficialV1EventPreprocessing.from_raw(
            fif_path, recording_start_seconds=start_seconds
        )
        channel_name_sets[label] = processor.channel_names
        block_keystrokes = keystrokes[(keystrokes["session"] == session) & (keystrokes["task"] == task)]
        trials = sorted(int(t) for t in block_keystrokes["trial_id"].unique())
        block_start = cursor
        sentence_reports = []

        for trial in trials:
            trial_rows = block_keystrokes[block_keystrokes["trial_id"] == trial].sort_values("start")
            sentence = sentence_rows[(session, task, trial)]
            typed = str(sentence["sentence_typed"])
            record = metadata[(session, task, trial)]
            reconstructed = "".join(button_char(b) for b in trial_rows["button"])
            if reconstructed != typed:
                raise RuntimeError(f"Label reconstruction failed for {label} trial {trial}")
            if len(trial_rows) != record["number_of_keystrokes"]:
                raise RuntimeError(f"Keystroke count disagrees with the manifest for {label} trial {trial}")
            for position, (_, event_row) in enumerate(trial_rows.iterrows()):
                timestamp = float(event_row["start"])
                tensor = processor.extract_event(timestamp).tensor
                if tensor.shape != (EVENT_SAMPLES, CHANNEL_COUNT):
                    raise RuntimeError(f"Bad event shape {tensor.shape} at {label} trial {trial} position {position}")
                if not np.isfinite(tensor).all():
                    raise RuntimeError(f"Non-finite event tensor at {label} trial {trial} position {position}")
                character = button_char(event_row["button"])
                slab[cursor] = tensor
                index_rows.append({
                    "row": cursor,
                    "subject": "S22",
                    "session": session,
                    "block": task,
                    "list_id": record["list_id"],
                    "sentence_UID": record["sentence_UID"],
                    "unique_sentence_group_id": record["unique_sentence_group_id"],
                    "trial_id": trial,
                    "event_position": position,
                    "timestamp_seconds": timestamp,
                    "button": str(event_row["button"]),
                    "label": character,
                    "label_id": CHAR_TO_ID[character],
                })
                cursor += 1
            sentence_reports.append({
                "sentence_UID": record["sentence_UID"],
                "trial_id": trial,
                "events": int(len(trial_rows)),
                "target_length": len(typed),
                "target_matches_keystrokes": True,
            })

        block_slab = np.asarray(slab[block_start:cursor])
        block_reports.append({
            "block": label,
            "session": session,
            "task": task,
            "fif": relative_fif,
            "fif_sha256": sha256_file(fif_path),
            "recording_start_seconds": start_seconds,
            "continuous_samples_50hz": int(processor.signal.shape[0]),
            "continuous_duration_seconds": float(processor.signal.shape[0] / 50.0),
            "sentences": len(trials),
            "events": cursor - block_start,
            "row_range": [block_start, cursor],
            "all_finite": bool(np.isfinite(block_slab).all()),
            "value_min": float(block_slab.min()),
            "value_max": float(block_slab.max()),
            "value_mean": float(block_slab.mean()),
            "value_std": float(block_slab.std()),
            "clamp_saturated_fraction": float(np.mean((np.abs(block_slab) >= 5.0))),
            "tensor_sha256": sha256_array(block_slab),
            "sentence_audit": sentence_reports,
        })
        del processor, block_slab
        gc.collect()

    slab.flush()
    INDEX_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in index_rows), encoding="utf-8"
    )

    report(index_rows, block_reports, len(set(channel_name_sets.values())) == 1, cursor, total_events)


def report(
    index_rows: list[dict],
    block_reports: list[dict],
    channels_identical: bool,
    cursor: int,
    total_events: int,
) -> None:
    reread = np.load(TENSOR_PATH, mmap_mode="r")
    labels = [row["label"] for row in index_rows]

    # A zero-filled sample is an entire 306-channel time slice of exact zeros.
    # Baseline-corrected MEG is never exactly zero on all 306 channels at once,
    # so this detects the official right-edge fill directly from the stored data.
    zero_slices = np.asarray((np.asarray(reread) == 0).all(axis=2))
    zero_per_event = zero_slices.sum(axis=1)
    zero_filled_events = int((zero_per_event > 0).sum())
    zero_fill = {
        "definition": "a time slice whose 306 channels are all exactly zero",
        "cause": (
            "the official MegExtractor adds a one-sample-short overlap into a zero array of the "
            "requested duration when the window start lands exactly halfway between two 50 Hz samples"
        ),
        "zero_filled_events": zero_filled_events,
        "zero_filled_event_fraction": zero_filled_events / max(cursor, 1),
        "zero_filled_samples_total": int(zero_per_event.sum()),
        "max_zero_samples_in_one_event": int(zero_per_event.max()),
        "position_histogram": {
            str(position): int(count)
            for position, count in enumerate(zero_slices.sum(axis=0)) if count
        },
        "per_block": {},
    }
    for block in block_reports:
        start, stop = block["row_range"]
        block_zero = zero_per_event[start:stop]
        block["zero_filled_events"] = int((block_zero > 0).sum())
        block["zero_filled_event_fraction"] = float((block_zero > 0).mean())
        zero_fill["per_block"][block["block"]] = {
            "events": block["events"],
            "zero_filled_events": block["zero_filled_events"],
        }

    checks = {
        "event_count_matches": cursor == 9650 == total_events,
        "shape_correct": tuple(reread.shape) == (9650, EVENT_SAMPLES, CHANNEL_COUNT),
        "dtype_is_float32": reread.dtype == np.float32,
        "all_finite": bool(np.isfinite(np.asarray(reread)).all()),
        "channel_names_identical_across_blocks": channels_identical,
        "all_labels_in_vocabulary": all(character in CHAR_TO_ID for character in labels),
        "all_256_sentences_covered": len({row["sentence_UID"] for row in index_rows}) == 256,
        "label_target_consistency_checked_per_sentence": True,
        "at_most_one_zero_filled_sample_per_event": int(zero_per_event.max()) <= 1,
        "official_v1_preprocessing_used": True,
    }
    verification = {
        "expected_events": 9650,
        "written_events": cursor,
        "shape": list(reread.shape),
        "dtype": str(reread.dtype),
        "channels_per_event": CHANNEL_COUNT,
        "samples_per_event": EVENT_SAMPLES,
        "sentences_covered": len({row["sentence_UID"] for row in index_rows}),
        "sentences_expected": 256,
        "historical_preprocessing_used": False,
        "no_neuroselect_padding": True,
        "no_neuroselect_padding_note": (
            "NeuroSelect adds no padding of its own. The only zeros are the official extractor's "
            "own right-edge fill, counted in zero_fill and verified against the official extractor "
            "in results/official_v1_boundary_event_parity.json."
        ),
        "checks": checks,
        "zero_fill": zero_fill,
    }
    verification["status"] = "PASS" if all(checks.values()) else "FAIL"

    artifact = {
        "task": "Step 3 official-v1 event tensors for all four S22 blocks",
        "official_revision": OFFICIAL_REVISION,
        "preprocessing_variant": "official_v1_compatible",
        "preprocessing_module": "src/neuroselect/official_v1_preprocessing.py",
        "preprocessing_order": [
            "select 306 MEG channels", "continuous 0.1-20 Hz filter", "continuous resample to 50 Hz",
            "continuous RobustScaler", "official overlap/sample-index semantics",
            "event window [-0.2, +0.3] s", "official per-event baseline", "official clamp [-5, +5]",
        ],
        "storage": {
            "tensor_file": str(TENSOR_PATH.relative_to(ROOT)).replace("\\", "/"),
            "index_file": str(INDEX_PATH.relative_to(ROOT)).replace("\\", "/"),
            "layout": "(event, time, channel) float32, one row per keystroke, memory-mapped",
            "bytes": int(TENSOR_PATH.stat().st_size),
            "duplication": "single slab; splits are row-index views, never copies",
        },
        "blocks": block_reports,
        "totals": {
            "events": cursor,
            "sentences": len({row["sentence_UID"] for row in index_rows}),
            "blocks": len(block_reports),
            "sessions": 2,
        },
        "label_distribution": {
            character: int(sum(1 for value in labels if value == character))
            for character in sorted(set(labels))
        },
        "verification": verification,
        "training_run": False,
        "model_used": False,
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "totals": artifact["totals"],
        "verification": verification,
        "blocks": [{k: block[k] for k in ("block", "events", "sentences", "value_min", "value_max", "tensor_sha256")} for block in block_reports],
    }, indent=2))


if __name__ == "__main__":
    main()
