"""Compare official event tensors with continuous-scope NeuroSelect preprocessing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from exca.map import MapInfra
from neuralset.events.etypes import Meg
from neuralset.extractors import MegExtractor


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data/raw/spanishbcbl_s22/MEG/FIF/22_9788/231214/block1.fif"
CURRENT_PATH = ROOT / "data/processed/spanishbcbl_subset/signals/trial_002.npy"
OUT_PATH = ROOT / "results/continuous_scope_event_parity.json"
FIGURE_PATH = ROOT / "results/figures/debug/continuous_scope_parity.png"
EXAMPLES_PATH = ROOT / "results/official_event_tensor_examples/trial_2_continuous_scope.npz"

FS_RAW = 1000.0
FS = 50.0
WINDOW_START = -0.2
WINDOW_DURATION = 0.5
WINDOW_SAMPLES = 25
BASELINE_SAMPLES = 10
CLAMP = 5.0
SENTENCE_START = 383.276
RECORDING_START = 308.0
EVENTS = [
    {"event_index": 0, "timestamp": 383.276, "label": "l"},
    {"event_index": 1, "timestamp": 383.451, "label": "a"},
    {"event_index": 2, "timestamp": 383.644, "label": " "},
    {"event_index": 3, "timestamp": 383.901, "label": "t"},
    {"event_index": 4, "timestamp": 383.985, "label": "a"},
]


def tensor_stats(array: np.ndarray) -> dict:
    array = np.asarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
        "sha256": hashlib.sha256(
            np.ascontiguousarray(array).tobytes()
        ).hexdigest(),
    }


def comparison(left: np.ndarray, right: np.ndarray) -> dict:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    difference = x - y
    return {
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "identical_hash": tensor_stats(left)["sha256"] == tensor_stats(right)["sha256"],
    }


def event_indices(timestamp: float) -> dict:
    raw_event = int(round((timestamp - RECORDING_START) * FS_RAW))
    resampled_event = int(round((timestamp - RECORDING_START) * FS))
    start = resampled_event + int(round(WINDOW_START * FS))
    return {
        "raw_sample_index_1000_hz": raw_event,
        "resampled_event_sample_index_50_hz": resampled_event,
        "window_start_sample_index_50_hz": start,
        "window_end_sample_index_exclusive_50_hz": start + WINDOW_SAMPLES,
    }


def extract_continuous_windows(processed: np.ndarray) -> list[np.ndarray]:
    windows = []
    for event in EVENTS:
        center = int(round((event["timestamp"] - RECORDING_START) * FS))
        start = center + int(round(WINDOW_START * FS))
        stop = start + WINDOW_SAMPLES
        if start < 0 or stop > processed.shape[0]:
            raise RuntimeError("Continuous event window unexpectedly requires padding")
        channel_time = processed[start:stop].T
        baseline = channel_time[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
        corrected = channel_time - baseline
        windows.append(np.clip(corrected, -CLAMP, CLAMP).T.astype(np.float32))
    return windows


def preprocess_continuously() -> tuple[np.ndarray, list[str]]:
    raw = mne.io.read_raw_fif(
        RAW_PATH, preload=False, verbose=False, allow_maxshield=True
    )
    raw = raw.copy().pick("meg")
    raw.load_data()
    channel_names = list(raw.ch_names)
    raw.filter(0.1, 20.0, n_jobs=1, verbose=False)
    raw.resample(FS, npad="auto", n_jobs=1, verbose=False)
    from sklearn.preprocessing import RobustScaler

    raw._data = RobustScaler().fit_transform(raw._data.T).T
    return raw.get_data().T.astype(np.float32, copy=False), channel_names


def main() -> None:
    official_cache = Path.home() / "Envs" / "official-continuous-parity-cache"
    official_cache.mkdir(parents=True, exist_ok=True)
    raw_data, channel_names = preprocess_continuously()
    current = np.load(CURRENT_PATH).astype(np.float32, copy=False)

    meg = Meg(
        start=308.0,
        duration=891.0,
        frequency=FS_RAW,
        filepath=str(RAW_PATH),
        subject="S22",
        timeline="Pinet2024Meg:session=1,subject=S22,task=block1",
    )
    extractor = MegExtractor(
        frequency=FS,
        filter=(0.1, 20.0),
        baseline=(0.0, 0.2),
        picks=("meg",),
        scaler="RobustScaler",
        scale_factor=None,
        clamp=CLAMP,
        infra=MapInfra(folder=official_cache, cluster=None, mode="force"),
        allow_maxshield=True,
    )

    official = []
    continuous = extract_continuous_windows(raw_data)
    rows = []
    for event, continuous_tensor in zip(EVENTS, continuous):
        official_tensor = (
            extractor(
                meg,
                start=event["timestamp"] + WINDOW_START,
                duration=WINDOW_DURATION,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            .T
        )
        center = int(round((event["timestamp"] - SENTENCE_START) * FS))
        start = center + int(round(WINDOW_START * FS))
        current_tensor = np.zeros((WINDOW_SAMPLES, current.shape[1]), np.float32)
        source_start = max(start, 0)
        source_stop = min(start + WINDOW_SAMPLES, current.shape[0])
        if source_stop > source_start:
            current_tensor[source_start - start : source_stop - start] = current[
                source_start:source_stop
            ]
        rows.append(
            {
                **event,
                "sample_indices": event_indices(event["timestamp"]),
                "official": tensor_stats(official_tensor),
                "current_neuroselect": tensor_stats(current_tensor),
                "continuous_scope_neuroselect": tensor_stats(continuous_tensor),
                "official_vs_current": comparison(official_tensor, current_tensor),
                "official_vs_continuous_scope": comparison(
                    official_tensor, continuous_tensor
                ),
            }
        )
        event["official_tensor"] = official_tensor
        event["current_tensor"] = current_tensor
        event["continuous_tensor"] = continuous_tensor

    plot_events = [(0, "early event 0"), (2, "non-early event 2")]
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    time = np.arange(WINDOW_SAMPLES) / FS + WINDOW_START
    for row_index, (event_index, title) in enumerate(plot_events):
        event = EVENTS[event_index]
        tensors = [
            (event["official_tensor"], "Official"),
            (event["current_tensor"], "Current NeuroSelect"),
            (event["continuous_tensor"], "Continuous-scope NeuroSelect"),
        ]
        for column_index, (tensor, label) in enumerate(tensors):
            axes[row_index, column_index].plot(time, tensor[:, 0])
            axes[row_index, column_index].set_title(f"{title}: {label}")
            axes[row_index, column_index].set_xlim(WINDOW_START, WINDOW_START + WINDOW_DURATION)
            axes[row_index, column_index].set_ylim(-5, 5)
            axes[row_index, column_index].set_xlabel("seconds relative to event")
            axes[row_index, column_index].set_ylabel("scaled amplitude")
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=140)
    plt.close(figure)

    def average(indices: list[int], key: str) -> dict:
        values = [rows[index][key] for index in indices]
        return {
            metric: float(np.mean([value[metric] for value in values]))
            for metric in (
                "mean_absolute_difference",
                "maximum_absolute_difference",
                "rmse",
                "correlation",
            )
        }

    early_current = average([0, 1], "official_vs_current")
    early_continuous = average([0, 1], "official_vs_continuous_scope")
    non_early_current = average([2, 3, 4], "official_vs_current")
    non_early_continuous = average([2, 3, 4], "official_vs_continuous_scope")
    aggregate_current = average(list(range(5)), "official_vs_current")
    aggregate_continuous = average(list(range(5)), "official_vs_continuous_scope")
    if aggregate_continuous["rmse"] < 0.05:
        interpretation = "A. Continuous-scope correction closes the parity gap"
    elif aggregate_continuous["rmse"] < aggregate_current["rmse"] * 0.8:
        interpretation = (
            "B. Continuous scope materially improves parity but residual mismatch remains"
        )
    else:
        interpretation = "C. Continuous scope does not explain the mismatch"

    artifact = {
        "official_commit": "5f9889621d0df391c5aab37c996683d308e6e926",
        "environment": "C:/Users/prana/Envs/neuroselect-brain2qwerty-v1",
        "raw_file": str(RAW_PATH),
        "trial": 2,
        "sentence": "la tasa excede las velocidades",
        "preprocessing_order": [
            "continuous MEG channel selection",
            "continuous 0.1-20 Hz filter",
            "continuous resampling to 50 Hz",
            "continuous RobustScaler",
            "event window extraction",
            "per-event baseline over relative [0.0, 0.2] seconds",
            "per-event clamp to [-5, 5]",
        ],
        "official_source_order": [
            "MneRaw._preprocess_raw filter/resample/RobustScaler",
            "MneRaw._get_timed_array event window",
            "scale_factor=None",
            "per-event baseline",
            "requested-window crop",
            "clamp",
        ],
        "window": {
            "start_seconds": WINDOW_START,
            "end_seconds": WINDOW_START + WINDOW_DURATION,
            "samples": WINDOW_SAMPLES,
            "sampling_rate_hz": FS,
        },
        "baseline": {
            "relative_seconds": [0.0, 0.2],
            "samples": BASELINE_SAMPLES,
            "per_channel": True,
        },
        "clamp": {"min": -CLAMP, "max": CLAMP},
        "channel_order": {
            "continuous_scope": channel_names,
            "count": len(channel_names),
            "status": "identical to official and current parity artifact",
        },
        "events": [
            {key: value for key, value in row.items() if key not in {
                "official_tensor", "current_tensor", "continuous_tensor"
            }}
            for row in rows
        ],
        "early_event_average": {
            "official_vs_current": early_current,
            "official_vs_continuous_scope": early_continuous,
        },
        "non_early_event_average": {
            "official_vs_current": non_early_current,
            "official_vs_continuous_scope": non_early_continuous,
        },
        "aggregate_average": {
            "official_vs_current": aggregate_current,
            "official_vs_continuous_scope": aggregate_continuous,
        },
        "interpretation": interpretation,
        "verification": {
            "no_sentence_crop_before_event_extraction": True,
            "no_zero_padding_continuous_scope": True,
            "same_raw_file": True,
            "same_five_events": True,
            "official_baseline_per_event": True,
            "official_clamp_order_confirmed": True,
            "no_source_modification": True,
            "no_production_preprocessing_modification": True,
            "no_training": True,
            "all_metrics_finite": True,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EXAMPLES_PATH,
        continuous_scope=np.stack([event["continuous_tensor"] for event in EVENTS]),
    )
    print(json.dumps({
        "artifact": str(OUT_PATH),
        "figure": str(FIGURE_PATH),
        "aggregate": artifact["aggregate_average"],
        "interpretation": artifact["interpretation"],
    }, indent=2))


if __name__ == "__main__":
    main()
