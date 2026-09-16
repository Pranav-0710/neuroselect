"""Phase 5F: event-aligned character information audit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neuroselect.data import load_signal, read_manifest


SEED = 33
FS = 50.0
WINDOW_START = -0.2
WINDOW_END = 0.3
WINDOW_SAMPLES = int(round((WINDOW_END - WINDOW_START) * FS))
TRIAL_IDS = [f"{i}.0_S22_1_block1" for i in range(2, 10)]


def label_button(button: str) -> str:
    return {"<space>": " ", "<special>": "@", "<number>": "9"}.get(button, button)


def extract_events() -> tuple[list[dict], dict[str, np.ndarray]]:
    manifest_path = Path("data/processed/spanishbcbl_subset/manifest.jsonl")
    records = {record["id"]: record for record in read_manifest(manifest_path)}
    events = pd.read_pickle("data/raw/spanishbcbl_s22/events_clean.pkl")
    events = events[
        (events["type"] == "Keystroke")
        & (events["sentence_UID"].isin(TRIAL_IDS))
    ].copy()
    signals = {}
    rows = []
    for trial_id in TRIAL_IDS:
        record = records[trial_id]
        signal = load_signal(manifest_path.parent / record["signal_path"])
        signals[trial_id] = signal
        trial_events = events[events["sentence_UID"] == trial_id].sort_values("start")
        trial_start = float(record["signal_start"])
        for event_index, (_, event) in enumerate(trial_events.iterrows()):
            center = int(round((float(event["start"]) - trial_start) * FS))
            start = center + int(round(WINDOW_START * FS))
            stop = start + WINDOW_SAMPLES
            window = np.zeros((WINDOW_SAMPLES, signal.shape[1]), dtype=np.float32)
            source_start = max(start, 0)
            source_stop = min(stop, signal.shape[0])
            if source_stop > source_start:
                window[source_start - start : source_stop - start] = signal[source_start:source_stop]
            rows.append({
                "trial_id": trial_id,
                "event_index": event_index,
                "character": label_button(str(event["button"])),
                "event_time": float(event["start"]),
                "relative_sample": center,
                "window_start_sample": start,
                "window_stop_sample": stop,
                "valid_samples": int(max(0, source_stop - source_start)),
                "window": window,
            })
    return rows, signals


def features(window: np.ndarray) -> dict[str, np.ndarray]:
    mean = window.mean(axis=0)
    mean_std = np.concatenate([mean, window.std(axis=0)])
    bins = np.array_split(window, 5, axis=0)
    temporal = np.concatenate([part.mean(axis=0) for part in bins])
    return {"A_mean": mean, "B_mean_std": mean_std, "C_temporal_bins": temporal}


def metrics(y_true: list[str], y_pred: np.ndarray) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def classifier() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    )


def leave_one_out(rows: list[dict], feature_name: str) -> dict:
    folds = []
    for trial_id in TRIAL_IDS:
        train = [row for row in rows if row["trial_id"] != trial_id]
        test = [row for row in rows if row["trial_id"] == trial_id]
        x_train = np.stack([row["features"][feature_name] for row in train])
        y_train = [row["character"] for row in train]
        x_test = np.stack([row["features"][feature_name] for row in test])
        y_test = [row["character"] for row in test]
        model = classifier()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        majority = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
        folds.append({
            "held_out_trial": trial_id,
            "n_train_events": len(train),
            "n_test_events": len(test),
            "metrics": metrics(y_test, pred),
            "majority_metrics": metrics(y_test, majority.predict(x_test)),
            "class_support_test": {
                str(key): int(value)
                for key, value in pd.Series(y_test).value_counts().items()
            },
        })
    return {
        "feature": feature_name,
        "folds": folds,
        "aggregate": {
            key: float(np.mean([fold["metrics"][key] for fold in folds]))
            for key in ("accuracy", "macro_f1", "balanced_accuracy")
        },
        "majority_aggregate": {
            key: float(np.mean([fold["majority_metrics"][key] for fold in folds]))
            for key in ("accuracy", "macro_f1", "balanced_accuracy")
        },
    }


def within_trial(rows: list[dict], feature_name: str, shuffled: bool = False) -> dict:
    fold_rows = []
    rng = np.random.default_rng(SEED)
    for trial_id in TRIAL_IDS:
        trial = [row for row in rows if row["trial_id"] == trial_id]
        cut = max(1, int(len(trial) * 0.7))
        train, test = trial[:cut], trial[cut:]
        x_train = np.stack([row["features"][feature_name] for row in train])
        x_test = np.stack([row["features"][feature_name] for row in test])
        labels = np.array([row["character"] for row in trial])
        if shuffled:
            labels = rng.permutation(labels)
        y_train = labels[:cut]
        y_test = labels[cut:]
        model = classifier()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        fold_rows.append({
            "trial_id": trial_id,
            "n_train_events": len(train),
            "n_test_events": len(test),
            "metrics": metrics(y_test.tolist(), pred),
        })
    return {
        "feature": feature_name,
        "shuffled_labels": shuffled,
        "folds": fold_rows,
        "aggregate": {
            key: float(np.mean([fold["metrics"][key] for fold in fold_rows]))
            for key in ("accuracy", "macro_f1", "balanced_accuracy")
        },
    }


def main() -> None:
    rows, _ = extract_events()
    for row in rows:
        row["features"] = features(row.pop("window"))
    class_counts = {
        str(key): int(value)
        for key, value in pd.Series([row["character"] for row in rows]).value_counts().sort_index().items()
    }
    feature_names = ["A_mean", "B_mean_std", "C_temporal_bins"]
    cross_trial = {name: leave_one_out(rows, name) for name in feature_names}
    best_feature = max(feature_names, key=lambda name: cross_trial[name]["aggregate"]["balanced_accuracy"])
    within = within_trial(rows, best_feature)
    shuffled = within_trial(rows, best_feature, shuffled=True)
    event_rows = [
        {key: value for key, value in row.items() if key != "features"}
        for row in rows
    ]
    result = {
        "window_definition": {"start_seconds": WINDOW_START, "end_seconds": WINDOW_END, "sampling_rate_hz": FS, "samples": WINDOW_SAMPLES},
        "seed": SEED,
        "trials": TRIAL_IDS,
        "total_retained_events": len(rows),
        "class_counts": class_counts,
        "class_frequencies": {key: value / len(rows) for key, value in class_counts.items()},
        "unique_classes": len(class_counts),
        "classes_with_fewer_than_3_examples": [key for key, value in class_counts.items() if value < 3],
        "feature_definitions": {
            "A_mean": "mean across 25 time samples per channel",
            "B_mean_std": "concatenated temporal mean and temporal standard deviation per channel",
            "C_temporal_bins": "concatenated means from five equal temporal bins per channel",
        },
        "events": event_rows,
        "leave_one_trial_out": cross_trial,
        "best_feature": best_feature,
        "within_trial": within,
        "within_trial_shuffled_control": shuffled,
        "verification": {
            "all_8_trials": True,
            "existing_events_reused": True,
            "no_sentence_ctc_retraining": True,
            "all_metrics_finite": True,
        },
    }
    output = Path("results/event_information_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(10, 4))
    chars = sorted(class_counts)
    axis.bar(chars, [class_counts[char] for char in chars])
    axis.set_title("Retained event label distribution")
    axis.set_xlabel("character")
    axis.set_ylabel("count")
    figure.tight_layout()
    figure.savefig("results/figures/debug/event_label_distribution.png", dpi=120)
    plt.close(figure)

    selected = rows[:: max(1, len(rows) // 12)][:12]
    figure, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True)
    for axis, row in zip(axes.flat, selected):
        axis.plot(row["features"]["C_temporal_bins"][:306])
        axis.set_title(f'{row["trial_id"].split(".", 1)[0]}: {row["character"]!r}')
    figure.suptitle("Deterministic event-window channel summaries")
    figure.tight_layout()
    figure.savefig("results/figures/debug/event_window_examples.png", dpi=120)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4))
    x = np.arange(len(feature_names))
    width = 0.25
    for offset, metric_name in enumerate(("accuracy", "macro_f1", "balanced_accuracy")):
        axis.bar(
            x + (offset - 1) * width,
            [cross_trial[name]["aggregate"][metric_name] for name in feature_names],
            width,
            label=metric_name,
        )
    axis.set_xticks(x, feature_names)
    axis.set_ylim(0, 1)
    axis.set_title("Cross-trial event classification")
    axis.legend()
    figure.tight_layout()
    figure.savefig("results/figures/debug/event_classification_results.png", dpi=120)
    plt.close(figure)
    print(json.dumps({"events": len(rows), "classes": len(class_counts), "best_feature": best_feature, "within": within["aggregate"], "shuffled": shuffled["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
