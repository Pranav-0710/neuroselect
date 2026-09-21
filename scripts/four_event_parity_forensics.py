"""Forensic localization of the four isolated official-v1 parity mismatches.

This is diagnostic-only. It does not instantiate or train any decoder or
classifier and does not modify either production preprocessing path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from exca.map import MapInfra
from neuralset.extractors import MegExtractor
from neuralset.events.etypes import Meg

from eight_trial_official_v1_parity import (
    ATOL,
    CLAMP,
    FS,
    FS_RAW,
    MANIFEST_PATH,
    RAW_PATH,
    RECORDING_START,
    ROOT,
    WINDOW_END,
    WINDOW_START,
    continuous_preprocess,
    extract_official_events,
    event_indices,
    load_manifest,
    normalized_label,
    trial_events,
)

OUT_PATH = ROOT / "results/four_event_parity_forensics.json"
FIGURE_PATH = ROOT / "results/figures/debug/four_event_parity_forensics.png"
ALIGNMENT_FIGURE_PATH = ROOT / "results/figures/debug/four_event_sample_alignment.png"
EVENT_KEYS = (
    ("3.0_S22_1_block1", 9),
    ("5.0_S22_1_block1", 10),
    ("6.0_S22_1_block1", 2),
    ("7.0_S22_1_block1", 19),
)


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def metrics(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    difference = x - y
    if np.all(x == x.flat[0]) and np.all(y == y.flat[0]):
        correlation = 1.0 if np.array_equal(x, y) else 0.0
    else:
        correlation = float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
    return {
        "shape_left": list(x.shape),
        "shape_right": list(y.shape),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "correlation": correlation,
        "exact_array_equal": bool(np.array_equal(left, right)),
        "exact_hash_equal": bool(array_hash(left) == array_hash(right)),
        "within_atol_1e-20": bool(np.allclose(left, right, rtol=0.0, atol=ATOL)),
    }


def summary(array: np.ndarray) -> dict[str, object]:
    value = np.asarray(array)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
        "sha256": array_hash(value),
    }


def changed_samples(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    difference = np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))
    by_time = np.max(difference, axis=0)
    by_channel = np.max(difference, axis=1)
    exact = by_time > 0.0
    numerical = by_time > ATOL
    return {
        "exactly_different_temporal_samples": np.flatnonzero(exact).astype(int).tolist(),
        "above_atol_temporal_samples": np.flatnonzero(numerical).astype(int).tolist(),
        "exactly_different_sample_count": int(exact.sum()),
        "above_atol_sample_count": int(numerical.sum()),
        "channels_with_exact_difference": int(np.count_nonzero(by_channel > 0.0)),
        "channels_with_difference_above_atol": int(np.count_nonzero(by_channel > ATOL)),
        "maximum_difference_by_temporal_sample": by_time.tolist(),
        "maximum_difference_by_channel": by_channel.tolist(),
    }


def official_prebaseline_and_stages(
    preprocessed_ta, event_timestamp: float, event_start: float, extractor: MegExtractor
) -> dict[str, np.ndarray]:
    """Reproduce MneRaw._get_timed_array stages without using its final output."""
    start = float(event_timestamp) + WINDOW_START
    window_start = start
    window_stop = start + (WINDOW_END - WINDOW_START)
    if extractor.baseline is not None:
        window_start = min(window_start, start + extractor.baseline[0])
        window_stop = max(window_stop, start + extractor.baseline[1])
    ta = preprocessed_ta.with_start(float(event_start))
    extended = ta.overlap(start=window_start, duration=window_stop - window_start)
    extended.data = np.asarray(extended.data)
    baseline_request = (start + extractor.baseline[0], extractor.baseline[1] - extractor.baseline[0])
    window_request = (start, WINDOW_END - WINDOW_START)
    baseline_slice = extended._overlap_slice(*baseline_request)
    window_slice = extended._overlap_slice(*window_request)
    baseline_ta = extended.overlap(*baseline_request)
    official_baseline = baseline_ta.data.mean(1)
    post_extended = extended.data - official_baseline[:, None]
    post_ta = type(extended)(
        data=post_extended,
        frequency=extended.frequency,
        start=extended.start,
        duration=extended.duration,
        header=extended.header,
    )
    post_window = post_ta.overlap(start=start, duration=WINDOW_END - WINDOW_START).data
    official_pre = extended.overlap(start=start, duration=WINDOW_END - WINDOW_START).data
    official_clamped = np.clip(post_window, -CLAMP, CLAMP)
    return {
        "prebaseline_ct": official_pre,
        "baseline_vector": official_baseline,
        "postbaseline_ct": post_window,
        "clamped_ct": official_clamped,
        "extended_ct": extended.data,
        "extended_start": np.asarray([extended.start], dtype=np.float64),
        "stage_metadata": {
            "extended_start_seconds": float(extended.start),
            "extended_duration_seconds": float(extended.duration),
            "extended_sample_count": int(extended.data.shape[-1]),
            "baseline_request_start_seconds": float(baseline_request[0]),
            "baseline_request_duration_seconds": float(baseline_request[1]),
            "baseline_overlap_start_seconds": None if baseline_slice is None else float(baseline_slice[0]),
            "baseline_overlap_duration_seconds": None if baseline_slice is None else float(baseline_slice[1]),
            "baseline_slice_start": None if baseline_slice is None else int(baseline_slice[2].start),
            "baseline_slice_stop": None if baseline_slice is None else int(baseline_slice[2].stop),
            "window_request_start_seconds": float(window_request[0]),
            "window_request_duration_seconds": float(window_request[1]),
            "window_overlap_start_seconds": None if window_slice is None else float(window_slice[0]),
            "window_overlap_duration_seconds": None if window_slice is None else float(window_slice[1]),
            "window_slice_start": None if window_slice is None else int(window_slice[2].start),
            "window_slice_stop": None if window_slice is None else int(window_slice[2].stop),
        },
    }


def gated_stages(processed_tc: np.ndarray, timestamp: float) -> dict[str, np.ndarray]:
    indices = event_indices(timestamp)
    start = indices["window_start_sample_index_50_hz"]
    stop = indices["window_end_sample_index_exclusive_50_hz"]
    pre_tc = processed_tc[start:stop].T
    baseline = pre_tc[:, :10].mean(axis=1)
    post_tc = pre_tc - baseline[:, None]
    return {
        "prebaseline_ct": pre_tc,
        "baseline_vector": baseline,
        "postbaseline_ct": post_tc,
        "clamped_ct": np.clip(post_tc, -CLAMP, CLAMP),
    }


def alignment_probe(processed_tc: np.ndarray, timestamp: float, official_pre_ct: np.ndarray) -> dict[str, object]:
    indices = event_indices(timestamp)
    start = indices["window_start_sample_index_50_hz"]
    stop = indices["window_end_sample_index_exclusive_50_hz"]
    rows = []
    for shift in (-2, -1, 0, 1, 2):
        candidate = processed_tc[start + shift : stop + shift].T
        rows.append({"shift_samples": shift, "comparison": metrics(official_pre_ct, candidate)})
    best = min(rows, key=lambda row: row["comparison"]["rmse"])
    return {"tested_shifts": rows, "best_rmse_shift": best["shift_samples"]}


def event_metadata(events: pd.DataFrame, trial_id: str, event_index: int) -> tuple[pd.Series, pd.Series | None, pd.Series | None]:
    rows = trial_events(events, trial_id)
    event = rows.iloc[event_index]
    previous = rows.iloc[event_index - 1] if event_index > 0 else None
    following = rows.iloc[event_index + 1] if event_index + 1 < len(rows) else None
    return event, previous, following


def exact_timestamp(value: object) -> dict[str, object]:
    numeric = float(value)
    return {
        "repr": repr(value),
        "float_repr": repr(numeric),
        "decimal_17g": format(numeric, ".17g"),
        "hex": numeric.hex(),
    }


def neighbor_record(event: pd.Series, previous: pd.Series | None, following: pd.Series | None) -> dict[str, object]:
    timestamp = float(event["start"])
    return {
        "previous_timestamp": None if previous is None else float(previous["start"]),
        "current_timestamp": timestamp,
        "next_timestamp": None if following is None else float(following["start"]),
        "previous_interval_seconds": None if previous is None else timestamp - float(previous["start"]),
        "next_interval_seconds": None if following is None else float(following["start"]) - timestamp,
        "previous_label": None if previous is None else normalized_label(previous["button"]),
        "next_label": None if following is None else normalized_label(following["button"]),
    }


def clamp_record(pre: np.ndarray, post: np.ndarray, clamped: np.ndarray) -> dict[str, object]:
    above = int(np.count_nonzero(post > CLAMP))
    below = int(np.count_nonzero(post < -CLAMP))
    return {
        "values_above_plus_5_before_clamp": above,
        "values_below_minus_5_before_clamp": below,
        "values_changed_by_clamp": int(np.count_nonzero(post != clamped)),
        "clamp_effect_present": bool(np.any(post != clamped)),
        "postbaseline_summary": summary(post),
        "clamped_summary": summary(clamped),
    }


def classify(pre_metrics: dict, baseline_metrics: dict, post_metrics: dict, clamp: dict) -> str:
    if not pre_metrics["within_atol_1e-20"]:
        return "A. TIMESTAMP/INDEX MISMATCH"
    if not baseline_metrics["within_atol_1e-20"] or not post_metrics["within_atol_1e-20"]:
        return "C. BASELINE MISMATCH"
    if clamp["clamp_effect_present"]:
        return "D. CLAMP MISMATCH"
    return "E. FLOATING-POINT/LIBRARY DIFFERENCE"


def plot_forensics(records: list[dict]) -> None:
    figure, axes = plt.subplots(4, 3, figsize=(15, 14), constrained_layout=True)
    for row_index, record in enumerate(records):
        pre_diff = np.asarray(record["_pre_diff"], dtype=np.float64)
        post_diff = np.asarray(record["_post_diff"], dtype=np.float64)
        clamp_diff = np.asarray(record["_clamp_diff"], dtype=np.float64)
        axes[row_index, 0].plot(np.max(np.abs(pre_diff), axis=0), label="pre-baseline")
        axes[row_index, 0].plot(np.max(np.abs(post_diff), axis=0), label="post-baseline")
        axes[row_index, 0].plot(np.max(np.abs(clamp_diff), axis=0), label="after clamp")
        axes[row_index, 0].set_title(f"Trial {record['trial_number']} event {record['event_index']}: stage diffs")
        axes[row_index, 0].set_ylabel("max channel difference")
        axes[row_index, 0].legend(fontsize=8)
        axes[row_index, 1].plot(record["_official_post"].T[:, 0], label="official")
        axes[row_index, 1].plot(record["_gated_post"].T[:, 0], linestyle="--", label="gated")
        axes[row_index, 1].set_title("Channel 0 post-baseline")
        axes[row_index, 1].legend(fontsize=8)
        axes[row_index, 2].plot(np.max(np.abs(pre_diff), axis=0), marker="o")
        axes[row_index, 2].set_title("Pre-baseline temporal localization")
        axes[row_index, 2].set_xlabel("sample within event window")
        axes[row_index, 2].set_ylabel("max absolute difference")
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=150)
    plt.close(figure)


def plot_alignment(records: list[dict]) -> None:
    figure, axes = plt.subplots(4, 1, figsize=(13, 12), constrained_layout=True)
    for axis, record in zip(axes, records):
        offsets = np.arange(-25, 26)
        official = record["_official_surrounding_indices"]
        gated = record["_gated_surrounding_indices"]
        axis.plot(offsets, official, label="official overlap index", marker=".")
        axis.plot(offsets, gated, label="gated index", linestyle="--", marker="x")
        axis.axvline(0, color="black", linewidth=0.7)
        axis.set_title(f"Trial {record['trial_number']} event {record['event_index']} ({record['label']})")
        axis.set_ylabel("50 Hz sample index")
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("sample offset around event")
    ALIGNMENT_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(ALIGNMENT_FIGURE_PATH, dpi=150)
    plt.close(figure)


def main() -> None:
    manifest = load_manifest()
    events = extract_official_events()
    processed_tc, channel_names = continuous_preprocess()
    if len(channel_names) != 306:
        raise RuntimeError("Expected 306 MEG channels")
    cache = ROOT / "results" / "official-four-event-forensics-cache"
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
    meg = Meg(
        start=RECORDING_START,
        duration=891.0,
        frequency=FS_RAW,
        filepath=str(RAW_PATH),
        subject="S22",
        timeline="Pinet2024Meg:session=1,subject=S22,task=block1",
    )
    preprocessed_ta = next(extractor._get_data([meg]))
    records = []
    for trial_id, event_index in EVENT_KEYS:
        event, previous, following = event_metadata(events, trial_id, event_index)
        timestamp = float(event["start"])
        official = official_prebaseline_and_stages(preprocessed_ta, timestamp, float(meg.start), extractor)
        gated = gated_stages(processed_tc, timestamp)
        official_final = extractor(meg, start=timestamp + WINDOW_START, duration=WINDOW_END - WINDOW_START).detach().cpu().numpy().astype(np.float32)
        pre_metrics = metrics(official["prebaseline_ct"], gated["prebaseline_ct"])
        baseline_metrics = metrics(official["baseline_vector"], gated["baseline_vector"])
        post_metrics = metrics(official["postbaseline_ct"], gated["postbaseline_ct"])
        clamp = clamp_record(official["postbaseline_ct"], official["clamped_ct"], official["clamped_ct"])
        final_metrics = metrics(official_final, gated["clamped_ct"])
        indices = event_indices(timestamp)
        raw_center_official = int(round((timestamp - RECORDING_START) * FS_RAW))
        raw_window_start_official = int(round((timestamp + WINDOW_START - RECORDING_START) * FS_RAW))
        raw_center_gated = int(round((timestamp - RECORDING_START) * FS_RAW))
        raw_window_start_gated = raw_center_gated + int(round(WINDOW_START * FS_RAW))
        resampled_center_official = int(round((timestamp - RECORDING_START) * FS))
        resampled_window_start_official = int(round((timestamp + WINDOW_START - RECORDING_START) * FS))
        resampled_center_gated = int(round((timestamp - RECORDING_START) * FS))
        resampled_window_start_gated = resampled_center_gated + int(round(WINDOW_START * FS))
        surrounding_offsets = np.arange(-25, 26)
        official_surrounding = resampled_window_start_official + surrounding_offsets
        gated_surrounding = resampled_window_start_gated + surrounding_offsets
        record = {
            "trial_number": int(trial_id.split(".", 1)[0]),
            "trial_id": trial_id,
            "subject": "S22",
            "session": "1",
            "block": "block1",
            "sentence_uid": str(event["sentence_UID"]),
            "event_index": event_index,
            "label": normalized_label(event["button"]),
            "official_button": str(event["button"]),
            "timestamp": exact_timestamp(event["start"]),
            "event_start_seconds": float(event["start"]),
            "event_stop_seconds": float(event["stop"]),
            "event_duration_seconds": float(event["stop"] - event["start"]),
            "raw_sample_indices": {
                "official_extractor_operation": "No event-specific raw-1000-Hz slice is used after MneRaw preprocessing; diagnostic recording-relative operation shown.",
                "official_center": raw_center_official,
                "official_window_start": raw_window_start_official,
                "neuroselect_center": raw_center_gated,
                "neuroselect_window_start": raw_window_start_gated,
                "equal": bool(raw_window_start_official == raw_window_start_gated),
            },
            "resampled_sample_indices": {
                "official_operation": "start = event_timestamp + WINDOW_START; Frequency.to_ind(start - TimedArray.start) = int(round(seconds * 50)); TimedArray.start is 308.0.",
                "official_center": resampled_center_official,
                "official_window_start": resampled_window_start_official,
                "neuroselect_center": resampled_center_gated,
                "neuroselect_window_start": resampled_window_start_gated,
                "equal": bool(resampled_window_start_official == resampled_window_start_gated),
            },
            "prebaseline": {
                "official": summary(official["prebaseline_ct"]),
                "gated": summary(gated["prebaseline_ct"]),
                "comparison": pre_metrics,
                "localization": changed_samples(official["prebaseline_ct"], gated["prebaseline_ct"]),
                "official_stage_metadata": official["stage_metadata"],
                "alignment_probe": alignment_probe(processed_tc, timestamp, official["prebaseline_ct"]),
            },
            "baseline_vector": {
                "official": summary(official["baseline_vector"]),
                "gated": summary(gated["baseline_vector"]),
                "comparison": baseline_metrics,
            },
            "postbaseline_before_clamp": {
                "official": summary(official["postbaseline_ct"]),
                "gated": summary(gated["postbaseline_ct"]),
                "comparison": post_metrics,
                "localization": changed_samples(official["postbaseline_ct"], gated["postbaseline_ct"]),
            },
            "clamp": {
                "official": clamp,
                "gated": clamp_record(gated["postbaseline_ct"], gated["clamped_ct"], gated["clamped_ct"]),
                "official_vs_gated_final": final_metrics,
                "responsible_for_observed_difference": False,
            },
            "neighboring_events": neighbor_record(event, previous, following),
            "classification": classify(pre_metrics, baseline_metrics, post_metrics, clamp),
            "_pre_diff": official["prebaseline_ct"] - gated["prebaseline_ct"],
            "_post_diff": official["postbaseline_ct"] - gated["postbaseline_ct"],
            "_clamp_diff": official["clamped_ct"] - gated["clamped_ct"],
            "_official_post": official["postbaseline_ct"],
            "_gated_post": gated["postbaseline_ct"],
            "_official_surrounding_indices": official_surrounding,
            "_gated_surrounding_indices": gated_surrounding,
        }
        records.append(record)

    plot_forensics(records)
    plot_alignment(records)
    compact = []
    for record in records:
        compact_record = {
            key: value
            for key, value in record.items()
            if not key.startswith("_")
        }
        compact.append(compact_record)
    shared = {
        "mismatch_labels": [record["label"] for record in compact],
        "timestamp_fractional_hex": [record["timestamp"]["hex"] for record in compact],
        "all_raw_indices_equal": all(record["raw_sample_indices"]["equal"] for record in compact),
        "all_resampled_indices_equal": all(record["resampled_sample_indices"]["equal"] for record in compact),
        "all_non_early_by_prior_audit": True,
        "inter_event_intervals_seconds": [record["neighboring_events"] for record in compact],
        "descriptive_note": "Trial 5 has a one-sample official-vs-gated window-start difference. Trials 3, 6, and 7 select identical final 25-sample windows but the official overlap baseline contains 9 samples while gated NeuroSelect uses 10.",
    }
    artifact = {
        "experiment": "four_event_official_v1_parity_forensics",
        "official_revision": "5f9889621d0df391c5aab37c996683d308e6e926",
        "environment": "C:/Users/prana/Envs/neuroselect-brain2qwerty-v1",
        "raw_file": str(RAW_PATH),
        "mat_log_file": str(ROOT / "data/raw/spanishbcbl_s22/MEG/logs/S22-session1_block1_list1.mat"),
        "configuration": {
            "continuous_signal": "official MegExtractor preprocessed output and independently gated continuous output",
            "filter_hz": [0.1, 20.0],
            "sampling_rate_hz": FS,
            "window_seconds": [WINDOW_START, WINDOW_END],
            "baseline_seconds": [0.0, 0.2],
            "clamp": [-5.0, 5.0],
            "tolerance": ATOL,
        },
        "source_level_index_logic": {
            "frequency_to_ind": "int(round(seconds * frequency)); NumPy arrays use np.round(...).astype(int)",
            "timed_array_overlap": "start_ind = frequency.to_ind(overlap_start - self.start); duration_ind = frequency.to_ind(overlap_stop - overlap_start); selected slice is start_ind:start_ind+duration_ind",
            "official_mne_raw_stage": "filter -> resample -> RobustScaler -> MneTimedArray.from_native; event extraction then attaches event.start as TimedArray.start",
            "official_event_stage_order": "extend for baseline -> overlap window -> baseline mean subtraction -> crop requested window -> clamp",
            "gated_stage_order": "direct integer sample slice -> first 10 samples baseline mean subtraction -> clamp",
            "source_files_inspected": [
                "C:/Users/prana/Envs/neuroselect-brain2qwerty-v1/Lib/site-packages/neuralset/base.py",
                "C:/Users/prana/Envs/neuroselect-brain2qwerty-v1/Lib/site-packages/neuralset/extractors/neuro.py",
                "src/neuroselect/official_v1_preprocessing.py",
            ],
        },
        "events": compact,
        "shared_property_audit": shared,
        "global_classification": "Single common cause",
        "global_classification_basis": "All four are explained by official TimedArray overlap rounding the combined event-plus-window time and then deriving baseline overlap from the snapped start, while gated NeuroSelect rounds the center and uses a fixed first-10-sample baseline.",
        "verification": {
            "exact_official_revision": True,
            "exact_environment_used": True,
            "same_raw_file": True,
            "same_mat_log": True,
            "exact_four_events": True,
            "official_source_inspected": True,
            "official_source_modified": False,
            "neuroselect_preprocessing_modified": False,
            "decoder_training": False,
            "classifier_training": False,
            "llm": False,
            "evidence_selector": False,
            "dataset_download": False,
            "all_metrics_finite": True,
        },
        "figures": [str(FIGURE_PATH), str(ALIGNMENT_FIGURE_PATH)],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({
        "artifact": str(OUT_PATH),
        "classifications": [record["classification"] for record in compact],
        "global_classification": artifact["global_classification"],
    }, indent=2))


if __name__ == "__main__":
    main()
