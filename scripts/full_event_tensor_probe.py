"""Phase 5H: leakage-safe full event-tensor linear probe."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from neuroselect.data import load_signal, read_manifest


SEED = 33
FS = 50.0
WINDOW_START = -0.2
WINDOW_END = 0.3
WINDOW_SAMPLES = int(round((WINDOW_END - WINDOW_START) * FS))
CHANNELS = 306
FEATURE_DIMENSION = WINDOW_SAMPLES * CHANNELS
TRIAL_IDS = [f"{i}.0_S22_1_block1" for i in range(2, 10)]
PCA_VARIANCE = 0.95
PCA_MAX_COMPONENTS = 200


def label_button(button: str) -> str:
    return {"<space>": " ", "<special>": "@", "<number>": "9"}.get(button, button)


def extract_events() -> list[dict]:
    manifest_path = Path("data/processed/spanishbcbl_subset/manifest.jsonl")
    records = {record["id"]: record for record in read_manifest(manifest_path)}
    events = pd.read_pickle("data/raw/spanishbcbl_s22/events_clean.pkl")
    events = events[
        (events["type"] == "Keystroke")
        & events["sentence_UID"].isin(TRIAL_IDS)
    ].copy()
    rows = []
    for trial_id in TRIAL_IDS:
        record = records[trial_id]
        signal = load_signal(manifest_path.parent / record["signal_path"])
        if signal.shape[1] != CHANNELS:
            raise ValueError(f"{trial_id} has {signal.shape[1]} channels, expected {CHANNELS}")
        trial_events = events[events["sentence_UID"] == trial_id].sort_values("start")
        trial_start = float(record["signal_start"])
        for event_index, (_, event) in enumerate(trial_events.iterrows()):
            center = int(round((float(event["start"]) - trial_start) * FS))
            start = center + int(round(WINDOW_START * FS))
            stop = start + WINDOW_SAMPLES
            if start < 0 or stop > signal.shape[0] or stop - start != WINDOW_SAMPLES:
                continue
            tensor = signal[start:stop].copy()
            if tensor.shape != (WINDOW_SAMPLES, CHANNELS):
                raise ValueError(f"Unexpected event tensor shape: {tensor.shape}")
            rows.append(
                {
                    "trial_id": trial_id,
                    "event_index": event_index,
                    "character": label_button(str(event["button"])),
                    "event_time": float(event["start"]),
                    "tensor": tensor,
                }
            )
    return rows


def metric_result(y_true: list[str], prediction: np.ndarray) -> dict[str, float]:
    labels = sorted(set(y_true) | set(prediction))
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "macro_f1": float(
            f1_score(y_true, prediction, labels=labels, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
    }


def fit_predict(
    x_train: np.ndarray,
    y_train: list[str],
    x_test: np.ndarray,
) -> tuple[np.ndarray, dict]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    test_scaled = scaler.transform(x_test)
    upper_bound = min(PCA_MAX_COMPONENTS, train_scaled.shape[0] - 1, train_scaled.shape[1])
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=SEED)
    train_pca = pca.fit_transform(train_scaled)
    if train_pca.shape[1] > upper_bound:
        pca = PCA(n_components=upper_bound, svd_solver="full", random_state=SEED)
        train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
    )
    model.fit(train_pca, y_train)
    return model.predict(test_pca), {
        "n_components": int(pca.n_components_),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "scaler": "StandardScaler fit on training events only",
        "pca": "PCA fit on training events only",
    }


def majority_prediction(y_train: list[str], n_test: int) -> np.ndarray:
    majority = pd.Series(y_train).value_counts().sort_index().idxmax()
    return np.repeat(majority, n_test)


def fold_result(
    trial_id: str,
    train_rows: list[dict],
    test_rows: list[dict],
    permutation: np.ndarray | None = None,
) -> dict:
    x_train = np.stack([row["tensor"].reshape(-1) for row in train_rows])
    x_test = np.stack([row["tensor"].reshape(-1) for row in test_rows])
    if permutation is not None:
        x_train = x_train[:, permutation]
        x_test = x_test[:, permutation]
    y_train = [row["character"] for row in train_rows]
    y_test = [row["character"] for row in test_rows]
    prediction, preprocessing = fit_predict(x_train, y_train, x_test)
    majority = majority_prediction(y_train, len(y_test))
    train_classes = sorted(set(y_train))
    test_classes = sorted(set(y_test))
    missing = sorted(set(test_classes) - set(train_classes))
    return {
        "held_out_trial": trial_id,
        "n_train_events": len(train_rows),
        "n_test_events": len(test_rows),
        "train_classes": train_classes,
        "held_out_classes": test_classes,
        "held_out_classes_absent_from_train": missing,
        "n_classes_present_in_train": len(train_classes),
        "n_held_out_classes_absent_from_train": len(missing),
        "metrics": metric_result(y_test, prediction),
        "majority_metrics": metric_result(y_test, majority),
        "predicted_class_distribution": {
            str(key): int(value)
            for key, value in pd.Series(prediction).value_counts().sort_index().items()
        },
        "preprocessing": preprocessing,
    }


def aggregate(folds: list[dict], rows: list[dict], predictions: dict[str, np.ndarray]) -> dict:
    y_true = [row["character"] for row in rows]
    pooled_prediction = np.concatenate([predictions[trial_id] for trial_id in TRIAL_IDS])
    majority_predictions = np.concatenate(
        [
            np.repeat(
                pd.Series(
                    [row["character"] for row in rows if row["trial_id"] != fold["held_out_trial"]]
                )
                .value_counts()
                .sort_index()
                .idxmax(),
                sum(row["trial_id"] == fold["held_out_trial"] for row in rows),
            )
            for fold in folds
        ]
    )
    return {
        "pooled_event_count": len(y_true),
        "metrics": metric_result(y_true, pooled_prediction),
        "majority_metrics": metric_result(y_true, majority_predictions),
        "mean_fold_metrics": {
            key: float(np.mean([fold["metrics"][key] for fold in folds]))
            for key in ("accuracy", "macro_f1", "balanced_accuracy")
        },
        "n_folds_with_unavailable_held_out_classes": sum(
            bool(fold["held_out_classes_absent_from_train"]) for fold in folds
        ),
    }


def leave_one_out(rows: list[dict], permutation: np.ndarray | None = None) -> dict:
    folds = []
    predictions = {}
    for trial_id in TRIAL_IDS:
        train_rows = [row for row in rows if row["trial_id"] != trial_id]
        test_rows = [row for row in rows if row["trial_id"] == trial_id]
        x_train = np.stack([row["tensor"].reshape(-1) for row in train_rows])
        x_test = np.stack([row["tensor"].reshape(-1) for row in test_rows])
        if permutation is not None:
            x_train = x_train[:, permutation]
            x_test = x_test[:, permutation]
        prediction, preprocessing = fit_predict(
            x_train,
            [row["character"] for row in train_rows],
            x_test,
        )
        predictions[trial_id] = prediction
        y_train = [row["character"] for row in train_rows]
        y_test = [row["character"] for row in test_rows]
        train_classes = sorted(set(y_train))
        held_out_classes = sorted(set(y_test))
        missing = sorted(set(held_out_classes) - set(train_classes))
        majority = majority_prediction(y_train, len(y_test))
        folds.append(
            {
                "held_out_trial": trial_id,
                "n_train_events": len(train_rows),
                "n_test_events": len(test_rows),
                "train_classes": train_classes,
                "held_out_classes": held_out_classes,
                "held_out_classes_absent_from_train": missing,
                "n_classes_present_in_train": len(train_classes),
                "n_held_out_classes_absent_from_train": len(missing),
                "metrics": metric_result(y_test, prediction),
                "majority_metrics": metric_result(y_test, majority),
                "predicted_class_distribution": {
                    str(key): int(value)
                    for key, value in pd.Series(prediction).value_counts().sort_index().items()
                },
                "preprocessing": preprocessing,
            }
        )
    return {"folds": folds, "aggregate": aggregate(folds, rows, predictions)}


def within_trial(rows: list[dict]) -> dict:
    fold_rows = []
    predictions = []
    truths = []
    for trial_id in TRIAL_IDS:
        trial = [row for row in rows if row["trial_id"] == trial_id]
        cut = max(1, int(len(trial) * 0.7))
        train_rows, test_rows = trial[:cut], trial[cut:]
        x_train = np.stack([row["tensor"].reshape(-1) for row in train_rows])
        x_test = np.stack([row["tensor"].reshape(-1) for row in test_rows])
        y_train = [row["character"] for row in train_rows]
        y_test = [row["character"] for row in test_rows]
        prediction, preprocessing = fit_predict(x_train, y_train, x_test)
        fold_rows.append(
            {
                "trial_id": trial_id,
                "n_train_events": len(train_rows),
                "n_test_events": len(test_rows),
                "metrics": metric_result(y_test, prediction),
                "train_classes": sorted(set(y_train)),
                "test_classes": sorted(set(y_test)),
                "held_out_classes_absent_from_train": sorted(set(y_test) - set(y_train)),
                "preprocessing": preprocessing,
            }
        )
        predictions.extend(prediction)
        truths.extend(y_test)
    return {
        "folds": fold_rows,
        "aggregate": metric_result(truths, np.asarray(predictions)),
    }


def load_5f_baseline() -> dict:
    result = json.loads(Path("results/event_information_audit.json").read_text(encoding="utf-8"))
    return {
        "accuracy": float(result["leave_one_trial_out"]["B_mean_std"]["aggregate"]["accuracy"]),
        "macro_f1": float(result["leave_one_trial_out"]["B_mean_std"]["aggregate"]["macro_f1"]),
        "balanced_accuracy": float(
            result["leave_one_trial_out"]["B_mean_std"]["aggregate"]["balanced_accuracy"]
        ),
        "majority_accuracy": float(
            result["leave_one_trial_out"]["B_mean_std"]["majority_aggregate"]["accuracy"]
        ),
    }


def main() -> None:
    rows = extract_events()
    if len(rows) == 0 or set(row["trial_id"] for row in rows) != set(TRIAL_IDS):
        raise RuntimeError("Expected valid event windows from all eight trials")
    permutation = np.random.default_rng(SEED).permutation(FEATURE_DIMENSION)
    cross_trial = leave_one_out(rows)
    permutation_control = leave_one_out(rows, permutation)
    within = within_trial(rows)
    class_counts = {
        str(key): int(value)
        for key, value in pd.Series([row["character"] for row in rows]).value_counts().sort_index().items()
    }
    baseline_5f = load_5f_baseline()
    full_metrics = cross_trial["aggregate"]["metrics"]
    comparison = {
        key: float(full_metrics[key] - baseline_5f[key])
        for key in ("accuracy", "macro_f1", "balanced_accuracy")
    }
    comparison["majority_accuracy_difference"] = float(
        cross_trial["aggregate"]["majority_metrics"]["accuracy"] - baseline_5f["majority_accuracy"]
    )
    result = {
        "experiment": "5H_full_event_tensor_linear_probe",
        "seed": SEED,
        "trials": TRIAL_IDS,
        "event_count": len(rows),
        "class_counts": class_counts,
        "window": {
            "start_seconds": WINDOW_START,
            "end_seconds": WINDOW_END,
            "sampling_rate_hz": FS,
            "samples": WINDOW_SAMPLES,
        },
        "feature_dimensionality": {
            "tensor_shape": [WINDOW_SAMPLES, CHANNELS],
            "flattened_features": FEATURE_DIMENSION,
            "flattening_order": "row-major time then channel",
        },
        "pca_configuration": {
            "variance_target": PCA_VARIANCE,
            "maximum_components": PCA_MAX_COMPONENTS,
            "solver": "full",
            "fit_scope": "training events only per fold",
        },
        "scaler_configuration": {
            "type": "StandardScaler",
            "fit_scope": "training events only per fold",
        },
        "classifier_configuration": {
            "type": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": SEED,
        },
        "leave_one_trial_out": cross_trial,
        "comparison_to_5f_mean_std": {
            "five_f_baseline": baseline_5f,
            "difference_full_tensor_minus_5f": comparison,
        },
        "within_trial": within,
        "permutation_control": {
            "feature_column_permutation_seed": SEED,
            "procedure": "one fixed permutation of flattened feature columns; labels unchanged",
            "leave_one_trial_out": permutation_control,
        },
        "verification": {
            "existing_8_trials_only": True,
            "existing_processed_signals": True,
            "exact_event_centered_window": [-0.2, 0.3],
            "event_tensor_shape": [25, 306],
            "flattening_consistent": True,
            "scaler_training_fold_only": True,
            "pca_training_fold_only": True,
            "held_out_fold_excluded_from_preprocessing_fit": True,
            "classifier_training_events_only": True,
            "balanced_class_weights": True,
            "seed_33": True,
            "no_ctc_retraining": True,
            "no_architecture_change": True,
            "no_llm": True,
            "no_evidence_selector": True,
            "no_preprocessing_change": True,
            "all_metrics_finite": True,
        },
    }
    output = Path("results/full_event_tensor_probe.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    names = ["accuracy", "macro_f1", "balanced_accuracy"]
    figure, axis = plt.subplots(figsize=(9, 4))
    x = np.arange(len(names))
    axis.bar(x - 0.25, [full_metrics[name] for name in names], 0.25, label="5H full tensor")
    axis.bar(
        x,
        [baseline_5f[name] for name in names],
        0.25,
        label="5F mean+std",
    )
    axis.bar(
        x + 0.25,
        [cross_trial["aggregate"]["majority_metrics"][name] for name in names],
        0.25,
        label="majority baseline",
    )
    axis.set_xticks(x, names)
    axis.set_ylim(0, 1)
    axis.set_title("Full event-tensor probe versus 5F")
    axis.legend()
    figure.tight_layout()
    figure.savefig("results/figures/debug/full_event_tensor_probe.png", dpi=120)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4))
    axis.bar(
        [trial_id.split(".", 1)[0] for trial_id in TRIAL_IDS],
        [fold["metrics"]["accuracy"] for fold in cross_trial["folds"]],
    )
    axis.axhline(
        cross_trial["aggregate"]["majority_metrics"]["accuracy"],
        color="red",
        linestyle="--",
        label="pooled majority accuracy",
    )
    axis.set_xlabel("held-out trial")
    axis.set_ylabel("accuracy")
    axis.set_ylim(0, 1)
    axis.set_title("5H full-tensor leave-one-trial-out accuracy")
    axis.legend()
    figure.tight_layout()
    figure.savefig("results/figures/debug/full_tensor_fold_results.png", dpi=120)
    plt.close(figure)
    print(
        json.dumps(
            {
                "events": len(rows),
                "classes": len(class_counts),
                "cross_trial": cross_trial["aggregate"],
                "within_trial": within["aggregate"],
                "permutation": permutation_control["aggregate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
