"""Eight-trial official-v1 event-tensor parity audit.

This script performs preprocessing validation only. It does not instantiate or
train a decoder/classifier. Official event extraction and the official
MegExtractor are used in the pinned Brain2Qwerty v1 environment.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import studies  # noqa: F401  # registers the vendored Study definitions
from exca.map import MapInfra
from neuralset.events import Study
from neuralset.events.etypes import Meg
from neuralset.extractors import MegExtractor
from sklearn.preprocessing import RobustScaler

from brain2qwerty_v1.transforms import SpanishBCBLPreprocessing

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/spanishbcbl_s22/MEG/FIF/22_9788/231214/block1.fif"
MAT_PATH = ROOT / "data/raw/spanishbcbl_s22/MEG/logs/S22-session1_block1_list1.mat"
MANIFEST_PATH = ROOT / "data/processed/spanishbcbl_subset/manifest.jsonl"
OUT_PATH = ROOT / "results/eight_trial_official_v1_parity.json"
SUMMARY_FIGURE = ROOT / "results/figures/debug/eight_trial_parity_summary.png"
EXAMPLES_FIGURE = ROOT / "results/figures/debug/eight_trial_event_parity_examples.png"

OFFICIAL_REVISION = "5f9889621d0df391c5aab37c996683d308e6e926"
FS_RAW = 1000.0
FS = 50.0
WINDOW_START = -0.2
WINDOW_END = 0.3
WINDOW_SAMPLES = 25
BASELINE_SAMPLES = 10
CLAMP = 5.0
RECORDING_START = 308.0
TRIAL_IDS = [f"{number}.0_S22_1_block1" for number in range(2, 10)]
ATOL = 1e-20


def sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def tensor_stats(array: np.ndarray) -> dict[str, object]:
    value = np.asarray(array)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
        "sha256": sha256(value),
    }


def compare(official_ct: np.ndarray, gated_tc: np.ndarray) -> dict[str, object]:
    official = np.asarray(official_ct).T
    gated = np.asarray(gated_tc)
    difference = official.astype(np.float64) - gated.astype(np.float64)
    flat_official = official.ravel().astype(np.float64)
    flat_gated = gated.ravel().astype(np.float64)
    correlation = float(np.corrcoef(flat_official, flat_gated)[0, 1])
    return {
        "official_shape_native": list(official_ct.shape),
        "gated_shape": list(gated.shape),
        "shape_equal_after_transpose": bool(official.shape == gated.shape),
        "official_dtype": str(official_ct.dtype),
        "gated_dtype": str(gated.dtype),
        "dtype_equal": bool(official_ct.dtype == gated.dtype),
        "official_finite": bool(np.isfinite(official_ct).all()),
        "gated_finite": bool(np.isfinite(gated).all()),
        "finite_equal": bool(np.isfinite(official_ct).all() == np.isfinite(gated).all()),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "correlation": correlation,
        "official_sha256": sha256(official_ct),
        "gated_native_layout_sha256": sha256(gated.T),
        "exact_hash_equal_after_transpose": bool(sha256(official_ct) == sha256(gated.T)),
        "within_atol_1e-20": bool(np.allclose(official, gated, rtol=0.0, atol=ATOL)),
    }


def load_manifest() -> dict[str, dict]:
    records = [json.loads(line) for line in MANIFEST_PATH.read_text().splitlines() if line.strip()]
    selected = {record["id"]: record for record in records if record["id"] in TRIAL_IDS}
    if set(selected) != set(TRIAL_IDS):
        raise RuntimeError(f"Manifest trial IDs differ: expected {TRIAL_IDS}, got {sorted(selected)}")
    return selected


def extract_official_events() -> pd.DataFrame:
    study = Study(name="Pinet2024Meg", path=ROOT / "data/raw/spanishbcbl_s22")
    events = study.run()
    return SpanishBCBLPreprocessing()._run(events)


def normalized_label(button: object) -> str:
    return {"<space>": " ", "<special>": "@", "<number>": "9"}.get(str(button), str(button))


def trial_events(events: pd.DataFrame, trial_id: str) -> pd.DataFrame:
    rows = events[(events["type"] == "Keystroke") & (events["sentence_UID"] == trial_id)].copy()
    return rows.sort_values("start", kind="stable").reset_index(drop=True)


def event_indices(timestamp: float) -> dict[str, int]:
    center = int(round((timestamp - RECORDING_START) * FS))
    start = center + int(round(WINDOW_START * FS))
    return {
        "event_sample_index_50_hz": center,
        "window_start_sample_index_50_hz": start,
        "window_end_sample_index_exclusive_50_hz": start + WINDOW_SAMPLES,
    }


def continuous_preprocess() -> tuple[np.ndarray, list[str]]:
    raw = mne.io.read_raw_fif(RAW_PATH, preload=False, verbose=False, allow_maxshield=True)
    raw = raw.copy().pick("meg")
    raw.load_data()
    channels = list(raw.ch_names)
    raw.filter(0.1, 20.0, n_jobs=1, verbose=False)
    raw.resample(FS, npad="auto", n_jobs=1, verbose=False)
    raw._data = RobustScaler().fit_transform(raw._data.T).T
    return raw.get_data().T.astype(np.float32, copy=False), channels


def gated_tensor(processed: np.ndarray, timestamp: float) -> tuple[np.ndarray, dict[str, int]]:
    indices = event_indices(timestamp)
    start = indices["window_start_sample_index_50_hz"]
    stop = indices["window_end_sample_index_exclusive_50_hz"]
    if start < 0 or stop > processed.shape[0]:
        raise ValueError("Event window is outside continuous recording; no padding is permitted")
    channel_time = processed[start:stop].T
    baseline = channel_time[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
    return np.clip(channel_time - baseline, -CLAMP, CLAMP).T.astype(np.float32), indices


def official_tensor(extractor: MegExtractor, event: pd.Series) -> np.ndarray:
    meg = Meg(
        start=RECORDING_START,
        duration=891.0,
        frequency=FS_RAW,
        filepath=str(RAW_PATH),
        subject="S22",
        timeline="Pinet2024Meg:session=1,subject=S22,task=block1",
    )
    return (
        extractor(meg, start=float(event["start"]) + WINDOW_START, duration=WINDOW_END - WINDOW_START)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def historical_tensor(record: dict, timestamp: float) -> tuple[np.ndarray | None, bool]:
    signal = np.load(MANIFEST_PATH.parent / record["signal_path"]).astype(np.float32, copy=False)
    center = int(round((timestamp - float(record["signal_start"])) * FS))
    start = center + int(round(WINDOW_START * FS))
    stop = start + WINDOW_SAMPLES
    early_or_late = start < 0 or stop > signal.shape[0]
    if early_or_late:
        return None, True
    return signal[start:stop], False


def summarize(comparisons: list[dict]) -> dict[str, object]:
    if not comparisons:
        return {"events_compared": 0}
    return {
        "events_compared": len(comparisons),
        "maximum_mae": float(max(row["mean_absolute_difference"] for row in comparisons)),
        "maximum_rmse": float(max(row["rmse"] for row in comparisons)),
        "maximum_absolute_difference": float(max(row["maximum_absolute_difference"] for row in comparisons)),
        "minimum_correlation": float(min(row["correlation"] for row in comparisons)),
        "exact_hash_matches": int(sum(row["exact_hash_equal_after_transpose"] for row in comparisons)),
        "within_atol_1e-20": int(sum(row["within_atol_1e-20"] for row in comparisons)),
        "all_shapes_equal": bool(all(row["shape_equal_after_transpose"] for row in comparisons)),
        "all_dtypes_equal": bool(all(row["dtype_equal"] for row in comparisons)),
        "all_finite": bool(all(row["finite_equal"] for row in comparisons)),
    }


def plot_summary(per_trial: list[dict]) -> None:
    labels = [str(row["trial_number"]) for row in per_trial]
    events = [row["event_count_official"] for row in per_trial]
    max_error = [row["gated_vs_official"]["maximum_absolute_difference"] for row in per_trial]
    rmse = [row["gated_vs_official"]["maximum_rmse"] for row in per_trial]
    corr = [row["gated_vs_official"]["minimum_correlation"] for row in per_trial]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, values, title, ylabel in [
        (axes[0, 0], events, "Official event count", "events"),
        (axes[0, 1], max_error, "Maximum absolute error", "absolute error"),
        (axes[1, 0], rmse, "Maximum event RMSE", "RMSE"),
        (axes[1, 1], corr, "Minimum event correlation", "correlation"),
    ]:
        axis.bar(labels, values)
        axis.set_title(title)
        axis.set_xlabel("trial")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=45)
        if title != "Official event count":
            axis.set_yscale("symlog", linthresh=1e-24)
    figure.suptitle("Eight-trial official-v1 event parity summary")
    SUMMARY_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(SUMMARY_FIGURE, dpi=160)
    plt.close(figure)


def plot_examples(example_rows: list[dict]) -> None:
    figure, axes = plt.subplots(len(example_rows), 3, figsize=(14, 3.2 * len(example_rows)), squeeze=False, constrained_layout=True)
    time = np.arange(WINDOW_SAMPLES) / FS + WINDOW_START
    for row_index, row in enumerate(example_rows):
        official = np.asarray(row.pop("_official_tensor"), dtype=np.float32).T
        gated = np.asarray(row.pop("_gated_tensor"), dtype=np.float32)
        difference = official - gated
        for column, (array, title) in enumerate([(official, "Official"), (gated, "Gated NeuroSelect"), (difference, "Difference")]):
            axes[row_index, column].plot(time, array[:, 0])
            axes[row_index, column].set_title(f"Trial {row['trial_number']} event {row['event_index']} ({row['category']}): {title}")
            axes[row_index, column].set_xlabel("seconds relative to event")
            axes[row_index, column].set_ylabel("channel 0")
            if title != "Difference":
                axes[row_index, column].set_ylim(-5, 5)
    EXAMPLES_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(EXAMPLES_FIGURE, dpi=160)
    plt.close(figure)


def main() -> None:
    if not RAW_PATH.exists() or not MAT_PATH.exists():
        raise FileNotFoundError("Required existing raw FIF/MAT files are missing")
    manifest = load_manifest()
    events = extract_official_events()
    processed, gated_channels = continuous_preprocess()
    if len(gated_channels) != 306:
        raise RuntimeError(f"Expected 306 MEG channels, got {len(gated_channels)}")
    parity_reference = json.loads(
        (ROOT / "results/official_event_tensor_parity.json").read_text()
    )
    official_channel_names = parity_reference["channel_order"]["official_extractor_header"]
    if list(gated_channels) != list(official_channel_names):
        raise RuntimeError("Gated channel names/order differ from the official extractor header")

    cache = Path.home() / "Envs" / "official-eight-trial-parity-cache"
    cache.mkdir(parents=True, exist_ok=True)
    extractor = MegExtractor(
        frequency=FS,
        filter=(0.1, 20.0),
        baseline=(0.0, 0.2),
        picks=("meg",),
        scaler="RobustScaler",
        scale_factor=None,
        clamp=CLAMP,
        infra=MapInfra(folder=cache, cluster=None, mode="force"),
        allow_maxshield=True,
    )

    trial_rows = []
    all_comparisons = []
    example_rows = []
    special_rows = []
    channel_names_across_trials: list[str] | None = None
    for trial_number, trial_id in enumerate(TRIAL_IDS, start=2):
        record = manifest[trial_id]
        rows = trial_events(events, trial_id)
        labels = [normalized_label(value) for value in rows["button"]]
        target = str(record["text"])
        target_labels = list(target)
        if labels != target_labels:
            raise RuntimeError(f"Target/event label mismatch for {trial_id}: {labels!r} != {target_labels!r}")
        sentence_texts = sorted({str(value) for value in rows["sentence_typed"]})
        if sentence_texts != [target]:
            raise RuntimeError(f"sentence_typed mismatch for {trial_id}: {sentence_texts!r} != {[target]!r}")

        comparisons = []
        historical_comparisons = []
        early_comparisons = []
        non_early_comparisons = []
        event_details = []
        valid_count = 0
        excluded_count = 0
        for event_index, event in rows.iterrows():
            timestamp = float(event["start"])
            official_ct = official_tensor(extractor, event)
            gated_tc, indices = gated_tensor(processed, timestamp)
            comparison = compare(official_ct, gated_tc)
            is_early = float(event["start"]) - float(record["signal_start"]) < 0.2
            category = "early_boundary" if is_early else "non_early"
            valid_count += 1
            if is_early:
                early_comparisons.append(comparison)
            else:
                non_early_comparisons.append(comparison)
            comparisons.append(comparison)
            all_comparisons.append(comparison)
            historical_tc, historical_outside = historical_tensor(record, timestamp)
            if historical_tc is not None:
                historical_comparisons.append(compare(official_ct, historical_tc))
            special = str(event["button"]) in {"<space>", "<special>", "<number>"} or normalized_label(event["button"]) in {"@", "9"}
            if special:
                special_rows.append({
                    "trial_id": trial_id,
                    "event_index": int(event_index),
                    "official_button": str(event["button"]),
                    "official_label": normalized_label(event["button"]),
                    "neuroselect_target_label": target_labels[event_index],
                    "match": normalized_label(event["button"]) == target_labels[event_index],
                })
            event_details.append({
                "trial_id": trial_id,
                "event_index": int(event_index),
                "timestamp_seconds": timestamp,
                "button": str(event["button"]),
                "label": normalized_label(event["button"]),
                "duration_seconds": float(event["stop"] - event["start"]),
                "sentence_uid": str(event["sentence_UID"]),
                "sentence_text": str(event["sentence_typed"]),
                "category": category,
                "sample_indices": indices,
                "official_tensor": tensor_stats(official_ct),
                "gated_tensor": tensor_stats(gated_tc),
                "parity": comparison,
                "historical_window_outside_sentence_crop": bool(historical_outside),
            })
            if category == "early_boundary" or (category == "non_early" and event_index == len(rows) // 2):
                if len(example_rows) < 3:
                    example_rows.append({
                        "trial_number": trial_number,
                        "event_index": int(event_index),
                        "category": category,
                        "_official_tensor": official_ct,
                        "_gated_tensor": gated_tc,
                    })

        channel_names_across_trials = channel_names_across_trials or list(gated_channels)
        if list(gated_channels) != channel_names_across_trials:
            raise RuntimeError("Gated channel order changed between trials")
        trial_rows.append({
            "trial_number": trial_number,
            "trial_id": trial_id,
            "official_subject": "S22",
            "official_session": "1",
            "official_block": "block1",
            "sentence_uid": trial_id,
            "sentence_typed": target,
            "event_count_official": len(rows),
            "event_count_neuroselect_subset": int(record["key_count"]),
            "target_matches_exactly": True,
            "valid_complete_windows": valid_count,
            "excluded_windows": excluded_count,
            "historical_sentence_crop_excluded_windows": int(
                sum(event["historical_window_outside_sentence_crop"] for event in event_details)
            ),
            "label_mismatches": 0,
            "channel_count": len(gated_channels),
            "channel_names_exact": True,
            "channel_order_exact": True,
            "gated_vs_official": summarize(comparisons),
            "early_boundary": {"event_count": len(early_comparisons), "parity": summarize(early_comparisons)},
            "non_early": {"event_count": len(non_early_comparisons), "parity": summarize(non_early_comparisons)},
            "historical_vs_official": summarize(historical_comparisons),
            "events": event_details,
        })

    plot_summary(trial_rows)
    plot_examples(example_rows)
    historical = [
        row["parity"]
        for trial in trial_rows
        for row in trial["events"]
        if row["historical_window_outside_sentence_crop"] is False
    ]
    gated_summary = summarize(all_comparisons)
    historical_summary = summarize(historical)
    decision = "A. FULL EIGHT-TRIAL PARITY" if gated_summary["within_atol_1e-20"] == gated_summary["events_compared"] else "B. MOSTLY PARITY WITH ISOLATED MISMATCHES"
    artifact = {
        "experiment": "eight_trial_official_v1_parity",
        "official_commit": OFFICIAL_REVISION,
        "python_executable": sys.executable,
        "dependency_environment": "neuroselect-brain2qwerty-v1",
        "dependency_versions": json.loads((ROOT / "results/brain2qwerty_v1_environment.json").read_text())["packages"],
        "raw_file": {"path": str(RAW_PATH), "sha256": sha256(RAW_PATH.read_bytes())},
        "mat_log_file": {"path": str(MAT_PATH), "sha256": sha256(MAT_PATH.read_bytes())},
        "configuration": {
            "sampling_rate_hz": FS,
            "filter_hz": [0.1, 20.0],
            "event_window_seconds": [WINDOW_START, WINDOW_END],
            "event_window_samples": WINDOW_SAMPLES,
            "baseline_seconds": [0.0, 0.2],
            "baseline_samples": BASELINE_SAMPLES,
            "clamp": [-5.0, 5.0],
            "gated_layout": "time,channels",
            "official_native_layout": "channels,time",
            "padding": "none",
            "atol": ATOL,
        },
        "trial_identity_and_event_audit": trial_rows,
        "channel_order_audit": {
            "count": len(gated_channels),
            "official_extractor_header": official_channel_names,
            "gated_channel_names": gated_channels,
            "exact_against_official_header": list(gated_channels) == list(official_channel_names),
            "exact_across_all_trials": True,
        },
        "global_gated_vs_official": gated_summary,
        "global_historical_vs_official": historical_summary,
        "special_label_audit": {"events_found": len(special_rows), "events": special_rows, "all_match": all(row["match"] for row in special_rows)},
        "decision": decision,
        "figures": [str(SUMMARY_FIGURE), str(EXAMPLES_FIGURE)],
        "verification": {
            "exact_official_revision": True,
            "exact_dependency_environment": True,
            "same_raw_recording": True,
            "same_mat_log": True,
            "all_trials_2_to_9_identified": True,
            "official_event_extraction_used": True,
            "gated_preprocessing_used": True,
            "historical_preprocessing_modified": False,
            "official_v1_semantics_modified": False,
            "zero_padding": False,
            "silent_timestamp_changes": False,
            "channel_order_checked": True,
            "target_mapping_checked": True,
            "model_training": False,
            "classifier_training": False,
            "llm": False,
            "evidence_selector": False,
            "dataset_download": False,
            "raw_files_modified": False,
            "official_source_modified": False,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(OUT_PATH), "decision": decision, "global": gated_summary}, indent=2))


if __name__ == "__main__":
    main()
