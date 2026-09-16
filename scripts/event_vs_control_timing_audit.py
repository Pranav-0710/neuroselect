"""Phase 5G: paired event-centered versus temporal-control windows."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neuroselect.data import load_signal, read_manifest


SEED = 33
FS = 50.0
WINDOWS = {
    "event_centered": (-0.2, 0.3),
    "temporal_control": (-1.2, -0.7),
}
TRIAL_IDS = [f"{i}.0_S22_1_block1" for i in range(2, 10)]


def label_button(button: str) -> str:
    return {"<space>": " ", "<special>": "@", "<number>": "9"}.get(button, button)


def metric_result(y_true: list[str], prediction: np.ndarray) -> dict:
    labels = sorted(set(y_true) | set(prediction))
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
    }


def classifier() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    )


def load_paired_events() -> tuple[dict[str, list[dict]], dict[str, np.ndarray]]:
    manifest_path = Path("data/processed/spanishbcbl_subset/manifest.jsonl")
    records = {record["id"]: record for record in read_manifest(manifest_path)}
    events = pd.read_pickle("data/raw/spanishbcbl_s22/events_clean.pkl")
    events = events[
        (events["type"] == "Keystroke")
        & events["sentence_UID"].isin(TRIAL_IDS)
    ]
    signals = {}
    paired = {name: [] for name in WINDOWS}
    original_count = len(events)
    for trial_id in TRIAL_IDS:
        record = records[trial_id]
        signal = load_signal(manifest_path.parent / record["signal_path"])
        signals[trial_id] = signal
        trial_events = events[events["sentence_UID"] == trial_id].sort_values("start")
        trial_start = float(record["signal_start"])
        for event_index, (_, event) in enumerate(trial_events.iterrows()):
            center = int(round((float(event["start"]) - trial_start) * FS))
            valid = {}
            for condition, (window_start, window_end) in WINDOWS.items():
                start = center + int(round(window_start * FS))
                stop = center + int(round(window_end * FS))
                valid[condition] = start >= 0 and stop <= signal.shape[0] and stop - start == 25
            if not all(valid.values()):
                continue
            event_info = {
                "trial_id": trial_id,
                "event_index": event_index,
                "character": label_button(str(event["button"])),
                "event_time": float(event["start"]),
            }
            for condition, (window_start, window_end) in WINDOWS.items():
                start = center + int(round(window_start * FS))
                stop = center + int(round(window_end * FS))
                event_info[condition] = signal[start:stop].copy()
            for condition in WINDOWS:
                paired[condition].append(event_info.copy())
    if len(paired["event_centered"]) != len(paired["temporal_control"]):
        raise RuntimeError("Window conditions do not share paired event count")
    return paired, signals


def feature(window: np.ndarray) -> np.ndarray:
    return np.concatenate([window.mean(axis=0), window.std(axis=0)])


def cross_trial(rows: list[dict]) -> dict:
    folds = []
    for trial_id in TRIAL_IDS:
        train = [row for row in rows if row["trial_id"] != trial_id]
        test = [row for row in rows if row["trial_id"] == trial_id]
        x_train = np.stack([feature(row["window"]) for row in train])
        y_train = [row["character"] for row in train]
        x_test = np.stack([feature(row["window"]) for row in test])
        y_test = [row["character"] for row in test]
        model = classifier()
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        majority = max(set(y_train), key=y_train.count)
        folds.append({
            "held_out_trial": trial_id,
            "metrics": metric_result(y_test, prediction),
            "majority_accuracy": float(np.mean(np.asarray(y_test) == majority)),
        })
    return {
        "folds": folds,
        "aggregate": {
            key: float(np.mean([fold["metrics"][key] for fold in folds]))
            for key in ("accuracy", "macro_f1", "balanced_accuracy")
        },
        "majority_accuracy": float(np.mean([fold["majority_accuracy"] for fold in folds])),
    }


def within_trial(rows: list[dict]) -> dict:
    folds = []
    for trial_id in TRIAL_IDS:
        trial = [row for row in rows if row["trial_id"] == trial_id]
        cut = max(1, int(len(trial) * 0.7))
        train, test = trial[:cut], trial[cut:]
        model = classifier()
        model.fit(
            np.stack([feature(row["window"]) for row in train]),
            [row["character"] for row in train],
        )
        prediction = model.predict(np.stack([feature(row["window"]) for row in test]))
        folds.append({
            "trial_id": trial_id,
            "metrics": metric_result([row["character"] for row in test], prediction),
        })
    return {
        "folds": folds,
        "aggregate": {
            key: float(np.mean([fold["metrics"][key] for fold in folds]))
            for key in ("accuracy", "macro_f1", "balanced_accuracy")
        },
    }


def main() -> None:
    paired, _ = load_paired_events()
    event_rows = [{"trial_id": row["trial_id"], "event_index": row["event_index"], "character": row["character"], "window": row["event_centered"]} for row in paired["event_centered"]]
    control_rows = [{"trial_id": row["trial_id"], "event_index": row["event_index"], "character": row["character"], "window": row["temporal_control"]} for row in paired["temporal_control"]]
    class_counts = dict(pd.Series([row["character"] for row in event_rows]).value_counts().sort_index())
    results = {
        "seed": SEED,
        "sampling_rate_hz": FS,
        "windows": {
            "event_centered": {"start_seconds": -0.2, "end_seconds": 0.3, "samples": 25},
            "temporal_control": {"start_seconds": -1.2, "end_seconds": -0.7, "samples": 25},
        },
        "original_retained_events": 240,
        "paired_usable_events": len(event_rows),
        "excluded_events": 240 - len(event_rows),
        "class_counts_paired": {str(key): int(value) for key, value in class_counts.items()},
        "same_event_indices": [
            row["event_index"] for row in event_rows
        ] == [row["event_index"] for row in control_rows],
        "feature_definition": "concatenated mean and standard deviation across 25 time samples for each of 306 channels",
        "classifier": "StandardScaler + balanced LogisticRegression(max_iter=1000)",
        "cross_trial": {
            "event_centered": cross_trial(event_rows),
            "temporal_control": cross_trial(control_rows),
        },
        "within_trial": {
            "event_centered": within_trial(event_rows),
            "temporal_control": within_trial(control_rows),
        },
    }
    event_cross = results["cross_trial"]["event_centered"]["aggregate"]
    control_cross = results["cross_trial"]["temporal_control"]["aggregate"]
    event_within = results["within_trial"]["event_centered"]["aggregate"]
    control_within = results["within_trial"]["temporal_control"]["aggregate"]
    results["differences"] = {
        "cross_trial": {key: event_cross[key] - control_cross[key] for key in event_cross},
        "within_trial": {key: event_within[key] - control_within[key] for key in event_within},
        "per_trial_accuracy": {
            trial_id: next(f["metrics"]["accuracy"] for f in results["cross_trial"]["event_centered"]["folds"] if f["held_out_trial"] == trial_id)
            - next(f["metrics"]["accuracy"] for f in results["cross_trial"]["temporal_control"]["folds"] if f["held_out_trial"] == trial_id)
            for trial_id in TRIAL_IDS
        },
    }
    result_path = Path("results/event_vs_control_timing_audit.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    names = ["accuracy", "macro_f1", "balanced_accuracy"]
    x = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(x - 0.18, [event_cross[name] for name in names], 0.36, label="event-centered")
    axis.bar(x + 0.18, [control_cross[name] for name in names], 0.36, label="temporal control")
    axis.set_xticks(x, names)
    axis.set_ylim(0, 1)
    axis.set_title("Event-centered vs temporal-control classification")
    axis.legend()
    figure.tight_layout()
    figure.savefig("results/figures/debug/event_vs_control_results.png", dpi=120)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4))
    differences = [results["differences"]["per_trial_accuracy"][trial_id] for trial_id in TRIAL_IDS]
    axis.bar([trial_id.split(".", 1)[0] for trial_id in TRIAL_IDS], differences)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("held-out trial")
    axis.set_ylabel("event-centered accuracy - control accuracy")
    axis.set_title("Per-trial timing comparison")
    figure.tight_layout()
    figure.savefig("results/figures/debug/event_vs_control_per_trial.png", dpi=120)
    plt.close(figure)
    print(json.dumps({"paired_events": len(event_rows), "cross_trial": results["differences"]["cross_trial"], "within_trial": results["differences"]["within_trial"]}, indent=2))


if __name__ == "__main__":
    main()
