"""Phase 5D: descriptive audit of existing processed real-data trials."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neuroselect.data import load_signal, read_manifest


TRAIN_IDS = [f"{index}.0_S22_1_block1" for index in range(2, 8)]
EVAL_IDS = [f"{index}.0_S22_1_block1" for index in range(8, 10)]
BANDS = ((0.1, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 20.0))


def scalar_stats(signal: np.ndarray, target_length: int, duration: float) -> dict:
    differences = np.diff(signal, axis=0)
    channel_std = signal.std(axis=0)
    return {
        "target_length": target_length,
        "T": int(signal.shape[0]),
        "channels": int(signal.shape[1]),
        "T_over_target": float(signal.shape[0] / target_length),
        "duration_seconds": duration,
        "mean": float(signal.mean()),
        "std": float(signal.std()),
        "median": float(np.median(signal)),
        "minimum": float(signal.min()),
        "maximum": float(signal.max()),
        "percentile_5": float(np.percentile(signal, 5)),
        "percentile_95": float(np.percentile(signal, 95)),
        "mean_abs_first_difference": float(np.abs(differences).mean()),
        "std_first_difference": float(differences.std()),
        "mean_squared_energy": float(np.mean(signal**2)),
        "rms": float(np.sqrt(np.mean(signal**2))),
        "mean_channel_std": float(channel_std.mean()),
        "median_channel_std": float(np.median(channel_std)),
        "near_zero_channel_fraction": float(np.mean(channel_std < 1e-6)),
    }


def frequency_summary(signal: np.ndarray, sampling_rate: float = 50.0) -> dict:
    demeaned = signal - signal.mean(axis=0, keepdims=True)
    frequencies = np.fft.rfftfreq(signal.shape[0], 1.0 / sampling_rate)
    power = np.abs(np.fft.rfft(demeaned, axis=0)) ** 2
    band_power = {}
    total = power[(frequencies >= 0.1) & (frequencies <= 20.0)].mean()
    for low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < high)
        band_power[f"{low:g}-{high:g}Hz"] = float(power[mask].mean() / max(total, 1e-12))
    return band_power


def normalized_distance(values: dict, pooled: np.ndarray) -> float:
    vector = np.array([values["mean"], values["std"], values["rms"]], dtype=float)
    center = np.array([pooled.mean(), pooled.std(), np.sqrt(np.mean(pooled**2))])
    scale = np.maximum(np.abs(center), 1e-6)
    return float(np.linalg.norm((vector - center) / scale))


def aggregate(rows: list[dict], key: str) -> dict:
    values = np.array([row["statistics"][key] for row in rows], dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "evaluation_over_training_mean": None,
    }


def main() -> None:
    manifest_path = Path("data/processed/spanishbcbl_subset/manifest.jsonl")
    records = read_manifest(manifest_path)
    records_by_id = {record["id"]: record for record in records}
    signal_root = manifest_path.parent
    rows = []
    signals = {}
    for record in records:
        signal = load_signal(signal_root / record["signal_path"])
        signals[record["id"]] = signal
        rows.append({
            "trial_id": record["id"],
            "trial_number": int(record["id"].split(".", 1)[0]),
            "chronological_order": int(record["id"].split(".", 1)[0]) - 2,
            "split": "train" if record["id"] in TRAIN_IDS else "evaluation",
            "target": record["text"],
            "statistics": scalar_stats(
                signal,
                len(record["text"]),
                float(record["signal_duration"]),
            ),
            "spectral_relative_power": frequency_summary(signal),
        })

    pooled_train = np.concatenate([signals[trial_id].ravel() for trial_id in TRAIN_IDS])
    pooled_eval = np.concatenate([signals[trial_id].ravel() for trial_id in EVAL_IDS])
    for row in rows:
        signal = signals[row["trial_id"]]
        row["distance_to_pooled_train"] = normalized_distance(
            row["statistics"], pooled_train
        )
        row["distance_to_pooled_evaluation"] = normalized_distance(
            row["statistics"], pooled_eval
        )

    statistic_keys = list(rows[0]["statistics"])
    aggregates = {}
    for key in statistic_keys:
        train_values = np.array([row["statistics"][key] for row in rows if row["split"] == "train"], dtype=float)
        eval_values = np.array([row["statistics"][key] for row in rows if row["split"] == "evaluation"], dtype=float)
        train_mean = float(train_values.mean())
        aggregates[key] = {
            "training_mean": train_mean,
            "training_std": float(train_values.std()),
            "evaluation_mean": float(eval_values.mean()),
            "evaluation_std": float(eval_values.std()),
            "evaluation_over_training_mean": (
                float(eval_values.mean() / train_mean) if abs(train_mean) > 1e-12 else None
            ),
        }
    spectral_aggregates = {}
    for band in rows[0]["spectral_relative_power"]:
        train_values = [row["spectral_relative_power"][band] for row in rows if row["split"] == "train"]
        eval_values = [row["spectral_relative_power"][band] for row in rows if row["split"] == "evaluation"]
        spectral_aggregates[band] = {
            "training_mean": float(np.mean(train_values)),
            "training_std": float(np.std(train_values)),
            "evaluation_mean": float(np.mean(eval_values)),
            "evaluation_std": float(np.std(eval_values)),
        }

    result = {
        "source_manifest": str(manifest_path),
        "preprocessing": "existing processed signals; no preprocessing performed",
        "train_trial_ids": TRAIN_IDS,
        "evaluation_trial_ids": EVAL_IDS,
        "near_zero_definition": "channel standard deviation < 1e-6",
        "sampling_rate_hz": 50.0,
        "broad_frequency_bands_hz": BANDS,
        "per_trial": rows,
        "aggregate_statistics": aggregates,
        "aggregate_spectral_relative_power": spectral_aggregates,
        "distribution_distance_definition": "L2 distance between mean, std, and RMS after componentwise normalization by pooled-training/evaluation absolute centers",
        "all_statistics_finite": bool(
            all(
                np.isfinite(value)
                for row in rows
                for value in row["statistics"].values()
            )
        ),
    }
    output = Path("results/trial_distribution_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    informative = ["T", "T_over_target", "duration_seconds", "std", "rms", "mean_abs_first_difference"]
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for axis, key in zip(axes.flat, informative):
        train = [row["statistics"][key] for row in rows if row["split"] == "train"]
        evaluation = [row["statistics"][key] for row in rows if row["split"] == "evaluation"]
        axis.boxplot([train, evaluation], labels=["train", "eval"])
        axis.set_title(key)
    figure.tight_layout()
    figure_path = Path("results/figures/debug/trial_distribution_statistics.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=120)
    plt.close(figure)

    spectral_figure, spectral_axis = plt.subplots(figsize=(11, 5))
    x = np.arange(len(rows))
    width = 0.2
    for index, band in enumerate(rows[0]["spectral_relative_power"]):
        spectral_axis.bar(
            x + (index - 1.5) * width,
            [row["spectral_relative_power"][band] for row in rows],
            width,
            label=band,
        )
    spectral_axis.set_xticks(x, [row["trial_number"] for row in rows])
    spectral_axis.set_xlabel("trial number")
    spectral_axis.set_ylabel("relative mean power")
    spectral_axis.set_title("Broad-band spectral summary")
    spectral_axis.legend()
    spectral_figure.tight_layout()
    spectral_path = Path("results/figures/debug/trial_spectral_summary.png")
    spectral_figure.savefig(spectral_path, dpi=120)
    plt.close(spectral_figure)
    print(json.dumps({"trials": len(rows), "all_statistics_finite": result["all_statistics_finite"]}, indent=2))


if __name__ == "__main__":
    main()
