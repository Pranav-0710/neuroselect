"""Cross-environment parity for the stored S22 event tensors.

The tensor slab was written from the workspace environment, whose scipy,
scikit-learn, pandas and torch differ from the pinned Brain2Qwerty v1
environment. `official_v1_boundary_event_parity.py` already shows that the gated
implementation equals the official MegExtractor *inside* the pinned
environment. This closes the remaining link: run the gated implementation in
the pinned environment and require exact equality with the stored slab rows.

Continuous filtering, resampling and RobustScaler run once per block, so any
environment difference would surface across essentially every event in that
block. An evenly spaced sample per block, plus every zero-filled boundary
event, is therefore decisive.

Run in the pinned environment:
    PYTHONPATH=src <official-env>/python scripts/s22_event_tensor_env_parity.py
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np

from neuroselect.official_v1_preprocessing import NeuroSelectOfficialV1EventPreprocessing

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/raw/spanishbcbl_s22"
TENSOR_PATH = ROOT / "data/processed/s22_official_v1/events.npy"
INDEX_PATH = ROOT / "data/processed/s22_official_v1/index.jsonl"
TENSOR_ARTIFACT = ROOT / "results/s22_expanded_event_tensors.json"
OUT_PATH = ROOT / "results/s22_event_tensor_env_parity.json"
SAMPLE_PER_BLOCK = 200
PACKAGES = ("numpy", "scipy", "scikit-learn", "pandas", "torch", "mne")


def sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main() -> None:
    slab = np.load(TENSOR_PATH, mmap_mode="r")
    index = [json.loads(line) for line in INDEX_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    blocks = json.loads(TENSOR_ARTIFACT.read_text(encoding="utf-8"))["blocks"]

    reports = []
    for block in blocks:
        start, stop = block["row_range"]
        rows = index[start:stop]
        stored = np.asarray(slab[start:stop])
        boundary = {i for i in range(len(rows)) if (stored[i] == 0).all(axis=1).any()}
        step = max(len(rows) // SAMPLE_PER_BLOCK, 1)
        chosen = sorted(set(range(0, len(rows), step)) | boundary)

        processor = NeuroSelectOfficialV1EventPreprocessing.from_raw(
            DATA_ROOT / block["fif"], recording_start_seconds=block["recording_start_seconds"]
        )
        exact = 0
        max_difference = 0.0
        mismatches = []
        for position in chosen:
            row = rows[position]
            fresh = processor.extract_event(float(row["timestamp_seconds"])).tensor
            difference = float(np.abs(fresh.astype(np.float64) - stored[position].astype(np.float64)).max())
            max_difference = max(max_difference, difference)
            if sha256(fresh) == sha256(stored[position]):
                exact += 1
            elif len(mismatches) < 5:
                mismatches.append({"row": row["row"], "max_abs_difference": difference})
        reports.append({
            "block": block["block"],
            "events_in_block": len(rows),
            "events_compared": len(chosen),
            "boundary_events_included": len(boundary),
            "exact_hash_matches": exact,
            "maximum_absolute_difference": max_difference,
            "mismatch_examples": mismatches,
        })
        print(json.dumps(reports[-1]), flush=True)
        del processor

    compared = sum(report["events_compared"] for report in reports)
    matched = sum(report["exact_hash_matches"] for report in reports)
    artifact = {
        "task": "cross-environment parity of the stored S22 event tensors",
        "question": (
            "Does the slab written from the workspace environment equal what the gated official-v1 "
            "implementation produces inside the pinned Brain2Qwerty v1 environment?"
        ),
        "comparison_environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        },
        "slab_written_by": "workspace environment (see results/s22_expanded_baseline.json reproducibility)",
        "sampling": f"{SAMPLE_PER_BLOCK} evenly spaced events per block plus every zero-filled boundary event",
        "blocks": reports,
        "events_compared": compared,
        "exact_hash_matches": matched,
        "maximum_absolute_difference": max(report["maximum_absolute_difference"] for report in reports),
        "status": "PASS" if compared and matched == compared else "FAIL",
        "chain_of_evidence": [
            "slab (workspace env) == gated implementation (pinned env): this artifact",
            "gated implementation (pinned env) == official MegExtractor (pinned env): "
            "results/official_v1_boundary_event_parity.json and results/eight_trial_official_v1_parity.json",
        ],
        "training_run": False,
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({k: artifact[k] for k in ("events_compared", "exact_hash_matches",
                                               "maximum_absolute_difference", "status")}, indent=2))


if __name__ == "__main__":
    main()
