"""Parity audit for the official right-edge output placement.

A keypress whose window start falls exactly halfway between two 50 Hz samples
makes `TimedArray.overlap` return a 24-sample window instead of 25. The official
`MegExtractor` still emits 25 samples, because it adds that overlap into a zero
array of the requested duration. This audit runs the real official extractor and
the gated implementation over every such event in all four S22 blocks, plus a
control sample of ordinary events, and checks them for exact equality.

Runs in the pinned Brain2Qwerty v1 environment. No decoder, no training.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from exca import MapInfra
from neuralset.events.etypes import Meg
from neuralset.extractors import MegExtractor

from neuroselect.official_v1_preprocessing import NeuroSelectOfficialV1EventPreprocessing

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/raw/spanishbcbl_s22"
EVENTS_PATH = DATA_ROOT / "events_clean_all_blocks.pkl"
OUT_PATH = ROOT / "results/official_v1_boundary_event_parity.json"
FS = 50.0
FS_RAW = 1000.0
WINDOW_START, WINDOW_END = -0.2, 0.3
CLAMP = 5.0
CONTROL_EVENTS_PER_BLOCK = 8

BLOCKS = (
    ("1", "block1", "MEG/FIF/22_9788/231214/block1.fif"),
    ("1", "block2", "MEG/FIF/22_9788/231214/block2.fif"),
    ("2", "block1", "MEG/FIF/22_9788/231222/Block1.fif"),
    ("2", "block2", "MEG/FIF/22_9788/231222/Block2.fif"),
)


def sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def compare(official_ct: np.ndarray, gated_tc: np.ndarray) -> dict:
    official = np.asarray(official_ct).T.astype(np.float64)
    gated = np.asarray(gated_tc).astype(np.float64)
    difference = official - gated
    return {
        "shape_equal_after_transpose": bool(official.shape == gated.shape),
        "maximum_absolute_difference": float(np.abs(difference).max()),
        "rmse": float(np.sqrt((difference**2).mean())),
        "exact_hash_equal_after_transpose": bool(sha256(np.asarray(official_ct)) == sha256(gated_tc.T)),
        "both_finite": bool(np.isfinite(official).all() and np.isfinite(gated).all()),
    }


def main() -> None:
    events = pd.read_pickle(EVENTS_PATH)
    events["session"] = events["session"].astype(str)
    events["task"] = events["task"].astype(str)
    keystrokes = events[events["type"] == "Keystroke"]
    meg_rows = events[events["type"] == "Meg"]
    meg_info = {
        (str(row["session"]), str(row["task"])): {
            "start": float(row["start"]),
            "duration": float(row["duration"]),
            "timeline": str(row["timeline"]),
        }
        for _, row in meg_rows.iterrows()
    }

    cache = Path.home() / "Envs" / "official-boundary-parity-cache"
    cache.mkdir(parents=True, exist_ok=True)
    extractor = MegExtractor(
        frequency=FS,
        filter=(0.1, 20.0),
        baseline=(0.0, 0.2),
        picks=("meg",),
        scaler="RobustScaler",
        scale_factor=None,
        clamp=CLAMP,
        infra=MapInfra(folder=cache, cluster=None, mode="cached"),
        allow_maxshield=True,
    )

    block_reports = []
    all_boundary = []
    all_control = []
    for session, task, relative_fif in BLOCKS:
        label = f"session{session}/{task}"
        info = meg_info[(session, task)]
        processor = NeuroSelectOfficialV1EventPreprocessing.from_raw(
            DATA_ROOT / relative_fif, recording_start_seconds=info["start"]
        )
        block_keystrokes = keystrokes[
            (keystrokes["session"] == session) & (keystrokes["task"] == task)
        ].sort_values("start", kind="stable")
        timestamps = [float(value) for value in block_keystrokes["start"]]

        boundary_timestamps = []
        gated_cache: dict[float, np.ndarray] = {}
        for timestamp in timestamps:
            result = processor.extract_event(timestamp)
            if result.metadata["zero_filled_samples"]:
                boundary_timestamps.append(timestamp)
                gated_cache[timestamp] = result.tensor

        step = max(len(timestamps) // CONTROL_EVENTS_PER_BLOCK, 1)
        control_timestamps = [
            timestamp for timestamp in timestamps[::step][:CONTROL_EVENTS_PER_BLOCK]
            if timestamp not in gated_cache
        ]
        for timestamp in control_timestamps:
            gated_cache[timestamp] = processor.extract_event(timestamp).tensor

        meg = Meg(
            start=info["start"],
            duration=info["duration"],
            frequency=FS_RAW,
            filepath=str(DATA_ROOT / relative_fif),
            subject="S22",
            timeline=info["timeline"],
        )

        def official_tensor(timestamp: float) -> np.ndarray:
            return (
                extractor(meg, start=timestamp + WINDOW_START, duration=WINDOW_END - WINDOW_START)
                .detach().cpu().numpy().astype(np.float32)
            )

        boundary_rows = []
        for timestamp in boundary_timestamps:
            gated = gated_cache[timestamp]
            official = official_tensor(timestamp)
            comparison = compare(official, gated)
            zero_columns = [int(i) for i in np.flatnonzero((np.asarray(official) == 0).all(axis=0))]
            boundary_rows.append({
                "block": label,
                "timestamp_seconds": timestamp,
                "official_zero_sample_positions": zero_columns,
                "gated_zero_sample_positions": [
                    int(i) for i in np.flatnonzero((gated == 0).all(axis=1))
                ],
                **comparison,
            })

        control_rows = []
        for timestamp in control_timestamps:
            control_rows.append({
                "block": label,
                "timestamp_seconds": timestamp,
                **compare(official_tensor(timestamp), gated_cache[timestamp]),
            })

        all_boundary.extend(boundary_rows)
        all_control.extend(control_rows)
        block_reports.append({
            "block": label,
            "fif": relative_fif,
            "keystrokes": len(timestamps),
            "boundary_events": len(boundary_timestamps),
            "boundary_fraction": len(boundary_timestamps) / len(timestamps),
            "control_events": len(control_rows),
            "boundary_exact_matches": sum(1 for row in boundary_rows if row["exact_hash_equal_after_transpose"]),
            "control_exact_matches": sum(1 for row in control_rows if row["exact_hash_equal_after_transpose"]),
            "boundary_max_absolute_difference": max(
                (row["maximum_absolute_difference"] for row in boundary_rows), default=0.0
            ),
            "control_max_absolute_difference": max(
                (row["maximum_absolute_difference"] for row in control_rows), default=0.0
            ),
        })
        del processor

    total_boundary = len(all_boundary)
    total_control = len(all_control)
    verification = {
        "all_boundary_events_match_official_exactly": all(
            row["exact_hash_equal_after_transpose"] for row in all_boundary
        ),
        "all_control_events_match_official_exactly": all(
            row["exact_hash_equal_after_transpose"] for row in all_control
        ),
        "boundary_maximum_absolute_difference": max(
            (row["maximum_absolute_difference"] for row in all_boundary), default=0.0
        ),
        "control_maximum_absolute_difference": max(
            (row["maximum_absolute_difference"] for row in all_control), default=0.0
        ),
        "boundary_events_compared": total_boundary,
        "control_events_compared": total_control,
        "gated_zero_positions_match_official": all(
            row["official_zero_sample_positions"] == row["gated_zero_sample_positions"]
            for row in all_boundary
        ),
    }
    verification["status"] = "PASS" if (
        verification["all_boundary_events_match_official_exactly"]
        and verification["all_control_events_match_official_exactly"]
        and verification["gated_zero_positions_match_official"]
    ) else "FAIL"

    artifact = {
        "task": "official-v1 right-edge output-placement parity",
        "why": (
            "TimedArray.overlap returns 24 of 25 samples when the window start lands exactly "
            "halfway between two 50 Hz samples. The official MegExtractor adds that overlap "
            "into a zero TimedArray of the requested duration, so its output is 25 samples with "
            "one zero. The gated implementation previously raised instead of reproducing this."
        ),
        "official_revision": "5f9889621d0df391c5aab37c996683d308e6e926",
        "extractor": {
            "class": "neuralset.extractors.MegExtractor",
            "frequency": FS, "filter": [0.1, 20.0], "baseline": [0.0, 0.2],
            "picks": ["meg"], "scaler": "RobustScaler", "clamp": CLAMP,
        },
        "blocks": block_reports,
        "boundary_events": all_boundary,
        "control_events": all_control,
        "verification": verification,
        "training_run": False,
        "decoder_or_classifier_run": False,
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({"blocks": block_reports, "verification": verification}, indent=2))


if __name__ == "__main__":
    main()
