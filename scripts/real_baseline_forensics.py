"""Run Phase 3 diagnostics on the downloaded SpanishBCBL subset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader

import studies
from brain2qwerty_v1.transforms import Brain2QwertyV1Splitter, SpanishBCBLPreprocessing
from brain2qwerty_v1.utils import BUTTON_MAPPING, CHAR_INDEX
from neuralset.events import Study
from neuroselect.data import collate_batch
from neuroselect.diagnostics import validate_ctc_geometry
from neuroselect.metrics import cer
from neuroselect.models import ConvCTC
from neuroselect.training import run_epoch, seed_everything
from neuroselect.vocab_spanishbcbl import (
    CHAR_TO_ID,
    VOCAB,
    decode_ctc,
    encode_text,
)


def stats(array: np.ndarray) -> dict:
    finite = np.asarray(array)[np.isfinite(array)]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "median": float(np.median(finite)),
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
        "near_zero_channel_fraction": float(
            np.mean(np.std(array, axis=0) < 1e-8)
        ),
        "saturated_channel_fraction": float(
            np.mean(np.any(np.abs(array) >= 5.0, axis=0))
        ),
    }


def load_events(root: Path) -> pd.DataFrame:
    events = Study(name="Pinet2024Meg", path=root).run()
    events = SpanishBCBLPreprocessing()._run(events)
    events.to_pickle(root / "events_clean.pkl")
    return Brain2QwertyV1Splitter()._run(events)


def raw_path(root: Path) -> Path:
    files = [p for p in (root / "MEG" / "FIF").rglob("*.fif") if "tapping" not in p.name.lower()]
    if len(files) != 1:
        raise RuntimeError(f"Expected one MEG file, found {len(files)}")
    return files[0]


def prepare_continuous_raw(
    raw: mne.io.BaseRaw,
) -> tuple[mne.io.BaseRaw, mne.io.BaseRaw]:
    filtered = raw.copy().load_data()
    filtered.filter(0.1, 20.0, n_jobs=1, verbose=False)
    resampled = filtered.copy().resample(50.0, npad="auto", n_jobs=1, verbose=False)
    resampled._data = RobustScaler().fit_transform(resampled._data.T).T
    return filtered, resampled


def process_trial(
    raw: mne.io.BaseRaw,
    filtered_raw: mne.io.BaseRaw,
    processed_raw: mne.io.BaseRaw,
    sentence: pd.Series,
) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    raw_segment = raw.copy().crop(
        tmin=float(sentence["start"]),
        tmax=float(sentence["stop"]),
        include_tmax=False,
    )
    raw_data = raw_segment.get_data().T.astype(np.float32)
    filtered_segment = filtered_raw.copy().crop(
        tmin=float(sentence["start"]),
        tmax=float(sentence["stop"]),
        include_tmax=False,
    )
    filtered = filtered_segment.get_data().T.astype(np.float32)
    segment = processed_raw.copy().crop(
        tmin=float(sentence["start"]),
        tmax=float(sentence["stop"]),
        include_tmax=False,
    )
    resampled = segment.get_data().T.astype(np.float32)
    baseline_count = max(1, int(round(min(0.2, len(raw_data) / 50.0) * 50)))
    baseline_corrected = resampled - resampled[:baseline_count].mean(axis=0, keepdims=True)
    scaled = baseline_corrected
    clamped = np.clip(scaled, -5.0, 5.0).astype(np.float32)
    stages = {
        "raw": stats(raw_data),
        "after_filter": stats(filtered),
        "after_resampling": stats(resampled),
        "after_robust_scaling": stats(scaled),
        "after_clamping": stats(clamped),
        "percentage_clipped": float(np.mean(np.abs(scaled) > 5.0) * 100),
        "per_channel_std_after_clamping": np.std(clamped, axis=0).tolist(),
    }
    return clamped, stages, {
        "raw": raw_data,
        "after_filter": filtered,
        "after_resampling": resampled,
        "standardized": (
            (baseline_corrected - np.mean(baseline_corrected, axis=0))
            / np.maximum(np.std(baseline_corrected, axis=0), 1e-6)
        ).astype(np.float32),
        "after_robust_scaling": scaled,
        "after_clamping": clamped,
    }


def train_one(
    signal: np.ndarray,
    target: list[int],
    learning_rate: float,
    epochs: int,
    seed: int = 33,
    clip: bool = True,
) -> dict:
    seed_everything(seed)
    item = {
        "signal": torch.from_numpy(signal),
        "target": torch.tensor(target, dtype=torch.long),
        "meta": {"text": decode_ctc(target), "id": "diagnostic"},
    }
    loader = DataLoader([item], 1, shuffle=False, collate_fn=collate_batch)
    model = ConvCTC(signal.shape[1], len(VOCAB), hidden=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    start = time.perf_counter()
    for _ in range(epochs):
        if clip:
            losses.append(run_epoch(model, loader, optimizer, torch.device("cpu")))
        else:
            model.train()
            batch = next(iter(loader))
            log_probs = model(batch["signals"]).log_softmax(-1).transpose(0, 1)
            loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)(
                log_probs,
                batch["targets"],
                batch["input_lengths"],
                batch["target_lengths"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        logits = model(batch["signals"])
        indices = logits.argmax(-1)[0, : batch["input_lengths"][0]].tolist()
    prediction = decode_ctc(indices)
    reference = decode_ctc(target)
    return {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "gradient_clipping": clip,
        "initial_loss": losses[0],
        "minimum_loss": min(losses),
        "final_loss": losses[-1],
        "cer": cer(reference, prediction),
        "prediction": prediction,
        "decoded_indices": indices,
        "blank_fraction": float(np.mean(np.asarray(indices) == 0)),
        "seconds": time.perf_counter() - start,
    }


def artificial_signal(target: list[int], time_steps: int, channels: int) -> np.ndarray:
    rng = np.random.default_rng(33)
    signal = rng.normal(0, 0.05, size=(time_steps, channels)).astype(np.float32)
    span = max(4, time_steps // len(target))
    for index, token in enumerate(target):
        start = index * span
        stop = min(time_steps, start + span)
        signal[start:stop, token % channels] += 3.0
    return signal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    seed_everything(33)
    root = args.data_root
    events = load_events(root)
    sentences = events[
        (events["type"] == "Sentence") & (events["is_percep"] == False)  # noqa: E712
        & events["trial_id"].isin(range(2, 10))
    ].sort_values("trial_id")
    path = raw_path(root)
    raw = mne.io.read_raw_fif(path, preload=False, verbose=False, allow_maxshield=True).pick("meg")
    filtered_raw, processed_raw = prepare_continuous_raw(raw)
    output = Path("results")
    debug = output / "figures" / "debug"
    debug.mkdir(parents=True, exist_ok=True)

    mapping = [
        {
            "official_label": label,
            "official_index": int(index),
            "neuroselect_index": int(index + 1),
            "decoded_symbol": CHAR_INDEX[index],
            "source": "vendor/brain2qwerty/brain2qwerty_v1/utils.py",
        }
        for label, index in BUTTON_MAPPING.items()
        if label in {"<space>", "<special>", "<number>"} or len(label) == 1
    ]
    mapping.extend(
        [{
            "official_label": "<blank>",
            "official_index": None,
            "neuroselect_index": 0,
            "decoded_symbol": "<blank>",
            "source": "NeuroSelect CTC convention",
        }]
    )

    all_rows = []
    reconstruction = []
    preprocessing = {}
    first_signal = None
    first_sentence = None
    for _, sentence in sentences.iterrows():
        keys = events[
            (events["type"] == "Keystroke")
            & (events["sentence_UID"] == sentence["sentence_UID"])
        ].sort_values("start")
        raw_labels = keys["button"].tolist()
        mapped = "".join(
            {"<space>": " ", "<special>": "@", "<number>": "9"}.get(label, label)
            for label in raw_labels
        )
        manifest_text = str(sentence["sentence_typed"])
        target = encode_text(manifest_text)
        signal, stage_stats, stage_arrays = process_trial(
            raw, filtered_raw, processed_raw, sentence
        )
        geometry = validate_ctc_geometry(len(signal), target)
        row = {
            "trial_id": int(sentence["trial_id"]),
            "sentence_uid": str(sentence["sentence_UID"]),
            "subject": str(sentence["subject"]),
            "session": str(sentence["session"]),
            "block": str(sentence["task"]),
            "source_file": str(path),
            "original_sampling_rate": float(raw.info["sfreq"]),
            "original_channel_count": len(raw.ch_names),
            "original_recording_duration": float(raw.times[-1] - raw.times[0]),
            "processed_duration": float(len(signal) / 50.0),
            "processed_shape": list(signal.shape),
            "target_text": manifest_text,
            "target_length": len(target),
            "encoded_target_indices": target,
            "decoded_target_text": decode_ctc(target),
            "unique_target_classes": sorted(set(target)),
            "input_length_T": len(signal),
            "target_length_U": len(target),
            "T_over_U": geometry["T_over_U"],
            "repeated_adjacent_target": geometry["has_repeated_adjacent_target"],
            "ctc_geometry": geometry,
            "initial_ctc_loss": None,
            "final_ctc_loss": None,
            "final_greedy_decoded_output": None,
            "decoded_output_indices": None,
            "event_count": len(keys),
            "events": [
                {
                    "event_number": int(index + 1),
                    "timestamp_seconds": float(event.start),
                    "raw_label": str(event.button),
                    "mapped_character": {"<space>": " ", "<special>": "@", "<number>": "9"}.get(event.button, event.button),
                }
                for index, event in keys.iterrows()
            ],
        }
        all_rows.append(row)
        reconstruction.append({
            "trial_id": row["trial_id"],
            "raw_event_labels": raw_labels,
            "mapped_characters": list(mapped),
            "event_derived_text": mapped,
            "official_sentence": str(sentence["sentence"]),
            "manifest_text": manifest_text,
            "ctc_target_text": decode_ctc(target),
            "exact_match": mapped == manifest_text == str(sentence["sentence"]) == decode_ctc(target),
        })
        preprocessing[str(row["trial_id"])] = stage_stats
        if first_signal is None:
            first_signal, first_sentence = signal, sentence
            first_arrays = stage_arrays

    # One-trial trace and learning-rate diagnostic.
    target = encode_text(str(first_sentence["sentence_typed"]))
    baseline = train_one(first_signal, target, 1e-3, args.epochs)
    debug_trial = all_rows[0]
    debug_trial.update({
        "initial_ctc_loss": baseline["initial_loss"],
        "final_ctc_loss": baseline["final_loss"],
        "final_greedy_decoded_output": baseline["prediction"],
        "decoded_output_indices": baseline["decoded_indices"],
    })
    optimization = {
        "trial_id": debug_trial["trial_id"],
        "learning_rates": [
            train_one(first_signal, target, rate, args.epochs)
            for rate in (1e-3, 3e-4, 1e-4)
        ],
        "no_clipping_1e-3": train_one(first_signal, target, 1e-3, args.epochs, clip=False),
    }
    normalization_diagnostic = {
        name: train_one(array, target, 1e-3, args.epochs)
        for name, array in {
            "official_compatible": first_arrays["after_clamping"],
            "per_channel_standardization": first_arrays["standardized"],
            "raw_after_filter_resampling": first_arrays["after_resampling"],
        }.items()
    }

    # Random-target and realistic-dimension artificial-target controls.
    rng = np.random.default_rng(7)
    random_target = rng.integers(1, len(VOCAB), size=len(target)).tolist()
    controls = {
        "random_target": train_one(first_signal, random_target, 1e-3, args.epochs),
        "artificially_encoded_target": train_one(
            artificial_signal(target, len(first_signal), first_signal.shape[1]),
            target,
            1e-3,
            args.epochs,
        ),
    }

    # Diagnostic figures.
    timestamps = np.arange(len(first_signal)) / 50.0
    keys = events[
        (events["type"] == "Keystroke")
        & (events["sentence_UID"] == first_sentence["sentence_UID"])
    ].sort_values("start")
    event_times = keys["start"].to_numpy() - float(first_sentence["start"])
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    axes[0].plot(timestamps, first_signal[:, :5])
    axes[0].set_title("First five processed MEG channels")
    axes[1].eventplot(event_times, orientation="horizontal")
    axes[1].set_title("Keystroke event timestamps")
    axes[2].hist(np.diff(event_times), bins=20)
    axes[2].set_title("Inter-keystroke interval density")
    axes[2].set_xlabel("seconds")
    fig.savefig(debug / "trial_002_timeline_and_events.png", dpi=120)
    plt.close(fig)
    representative = event_times[len(event_times) // 2]
    center = int(representative * 50)
    lo, hi = max(0, center - 25), min(len(first_signal), center + 26)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot((np.arange(lo, hi) - center) / 50.0, first_signal[lo:hi, :5])
    ax.set_title("Event-aligned signal around representative keystroke")
    ax.set_xlabel("seconds relative to event")
    fig.tight_layout()
    fig.savefig(debug / "trial_002_event_aligned.png", dpi=120)
    plt.close(fig)

    (output / "real_baseline_debug_trial.json").write_text(json.dumps(debug_trial, indent=2), encoding="utf-8")
    (output / "mapping_reconstruction_examples.json").write_text(json.dumps(reconstruction[:3], indent=2), encoding="utf-8")
    (output / "alignment_audit.json").write_text(json.dumps({
        "official_event_definition": "Keystroke events are behavioral keypresses aligned to STI101 trigger events.",
        "timestamps": "seconds; MNE sample indices divided by raw.info['sfreq']",
        "original_sampling_rate": float(raw.info["sfreq"]),
        "offset": "official alignment matches trigger sequences; no additional offset applied",
        "every_target_character_has_retained_event": all(
            len(r["raw_event_labels"]) == len(encode_text(r["manifest_text"]))
            for r in reconstruction
        ),
        "trials": all_rows,
    }, indent=2), encoding="utf-8")
    (output / "preprocessing_statistics.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    (output / "ctc_geometry_report.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    (output / "optimization_diagnostic.json").write_text(
        json.dumps(
            {
                "optimization": optimization,
                "normalization_diagnostic": normalization_diagnostic,
                "controls": controls,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "trials": len(all_rows),
        "mapping_exact_matches": sum(r["exact_match"] for r in reconstruction),
        "ctc_feasible": sum(r["ctc_geometry"]["structurally_feasible"] for r in all_rows),
        "one_trial_initial_loss": baseline["initial_loss"],
        "one_trial_final_loss": baseline["final_loss"],
        "one_trial_prediction": baseline["prediction"],
    }, indent=2))


if __name__ == "__main__":
    main()
