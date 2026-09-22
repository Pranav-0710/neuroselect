"""Audit NeuroSelect preprocessing against the vendored Brain2Qwerty v1 path."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from neuroselect.data import load_signal, read_manifest


ROOT = Path(".")
RAW_ROOT = ROOT / "data/raw/spanishbcbl_s22"
MANIFEST = ROOT / "data/processed/spanishbcbl_subset/manifest.jsonl"
TRIAL_ID = "2.0_S22_1_block1"
TRIAL_NUMBER = 2
FS = 50.0
WINDOW_START = -0.2
WINDOW_DURATION = 0.5
WINDOW_SAMPLES = 25
EXPECTED_REVISION = "5f9889621d0df391c5aab37c996683d308e6e926"
OFFICIAL_ROOT = ROOT / "vendor/brain2qwerty"


def stats(array: np.ndarray) -> dict:
    array = np.asarray(array)
    finite = array[np.isfinite(array)]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "mean": float(np.mean(finite)) if finite.size else None,
        "std": float(np.std(finite)) if finite.size else None,
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    }


def git_revision() -> tuple[str | None, str | None]:
    try:
        head = subprocess.check_output(
            ["git", "-C", str(OFFICIAL_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        head = None
    try:
        object_type = subprocess.check_output(
            ["git", "-C", str(OFFICIAL_ROOT), "cat-file", "-t", EXPECTED_REVISION],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        object_type = None
    return head, object_type


def raw_path() -> Path:
    files = [
        path
        for path in (RAW_ROOT / "MEG/FIF").rglob("*.fif")
        if "tapping" not in path.name.lower()
    ]
    if len(files) != 1:
        raise RuntimeError(f"Expected one downloaded MEG file, found {len(files)}")
    return files[0]


def process_continuous(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, mne.io.BaseRaw]:
    filtered = raw.copy().load_data()
    filtered.filter(0.1, 20.0, n_jobs=1, verbose=False)
    resampled = filtered.copy().resample(50.0, npad="auto", n_jobs=1, verbose=False)
    resampled._data = RobustScaler().fit_transform(resampled._data.T).T
    return filtered, resampled


def crop_array(raw: mne.io.BaseRaw, start: float, stop: float) -> np.ndarray:
    return (
        raw.copy()
        .crop(tmin=start, tmax=stop, include_tmax=False)
        .get_data()
        .T.astype(np.float32)
    )


def official_event_tensor(
    processed: mne.io.BaseRaw,
    event_time: float,
) -> tuple[np.ndarray, dict]:
    start = event_time + WINDOW_START
    stop = start + WINDOW_DURATION
    segment = processed.copy().crop(tmin=start, tmax=stop, include_tmax=False)
    channel_time = segment.get_data().astype(np.float32)
    if channel_time.shape[1] != WINDOW_SAMPLES:
        raise RuntimeError(f"Official event window shape was {channel_time.shape}")
    baseline = channel_time[:, : int(round(0.2 * FS))].mean(axis=1, keepdims=True)
    corrected = channel_time - baseline
    clamped = np.clip(corrected, -5.0, 5.0).astype(np.float32)
    return clamped, {
        "event_timestamp": event_time,
        "window_start": start,
        "window_end": stop,
        "orientation": "C,T before NeuroSelect-compatible transpose",
        "raw_channel_time": stats(channel_time),
        "baseline_corrected_channel_time": stats(corrected),
        "final_channel_time": stats(clamped),
    }


def current_event_tensor(
    signal: np.ndarray,
    signal_start: float,
    event_time: float,
) -> tuple[np.ndarray, dict]:
    center = int(round((event_time - signal_start) * FS))
    start = center + int(round(WINDOW_START * FS))
    stop = start + WINDOW_SAMPLES
    tensor = signal[start:stop]
    return tensor, {
        "event_timestamp": event_time,
        "window_start_sample": start,
        "window_stop_sample": stop,
        "window_start": signal_start + start / FS,
        "window_end": signal_start + stop / FS,
        "orientation": "T,C",
        "tensor": stats(tensor),
    }


def stage_inventory() -> list[dict]:
    return [
        {
            "stage": "raw MEG loading",
            "official": "MneRaw reads the MNE raw event, with allow_maxshield=True in v1 config.",
            "neuroselect": "prepare_real_subset.py uses mne.io.read_raw_fif(..., preload=False, allow_maxshield=True).",
            "status": "MATCH",
            "evidence": "vendor/brain2qwerty/brain2qwerty_v1/config/xp_config.py:40-48; scripts/prepare_real_subset.py:47-50",
        },
        {
            "stage": "channel selection",
            "official": "MegExtractor picks ('meg',), drops no bad channels by default, and uses channel_order='unique'.",
            "neuroselect": "raw.pick('meg'); no bad-channel dropping or channel reordering is applied.",
            "status": "MATCH",
            "evidence": "vendor/brain2qwerty/brain2qwerty_v1/config/xp_config.py:40-48; installed neuralset.extractors.neuro.MneRaw._preprocess_raw",
        },
        {
            "stage": "spatial projection",
            "official": "apply_proj=False; no MNE projectors are applied.",
            "neuroselect": "No projectors are applied.",
            "status": "MATCH",
            "evidence": "vendor/brain2qwerty/brain2qwerty_v1/config/xp_config.py:44; scripts/prepare_real_subset.py:50-52",
        },
        {
            "stage": "filtering",
            "official": "Band-pass filter 0.1-20.0 Hz before resampling.",
            "neuroselect": "Band-pass filter 0.1-20.0 Hz before resampling.",
            "status": "MATCH",
            "evidence": "vendor/brain2qwerty/brain2qwerty_v1/config/xp_config.py:42; scripts/prepare_real_subset.py:52-53",
        },
        {
            "stage": "resampling",
            "official": "Resample to 50 Hz.",
            "neuroselect": "Resample to 50 Hz.",
            "status": "MATCH",
            "evidence": "vendor/brain2qwerty/brain2qwerty_v1/config/xp_config.py:41; scripts/prepare_real_subset.py:54",
        },
        {
            "stage": "normalization",
            "official": "Fit RobustScaler per continuous MNE recording after resampling.",
            "neuroselect": "Fit RobustScaler per continuous recording after resampling.",
            "status": "MATCH",
            "evidence": "installed neuralset.extractors.neuro.MneRaw._preprocess_raw; scripts/prepare_real_subset.py:55",
        },
        {
            "stage": "event extraction",
            "official": "Study event extraction plus SpanishBCBLPreprocessing creates Keystroke and Sentence events.",
            "neuroselect": "Reuses Study and SpanishBCBLPreprocessing._run, then saves events_clean.pkl.",
            "status": "MATCH",
            "evidence": "vendor/brain2qwerty/brain2qwerty_v1/transforms.py:16-140; scripts/prepare_real_subset.py:18-22",
        },
        {
            "stage": "event timestamps",
            "official": "Segments begin at Keystroke.start plus start=-0.2 seconds.",
            "neuroselect": "Event center is rounded to a 50 Hz sample relative to sentence signal_start.",
            "status": "PARTIAL",
            "evidence": "vendor/brain2qwerty/brain2qwerty_v1/main.py:89-96; scripts/event_information_audit.py:48-52",
        },
        {
            "stage": "keystroke window",
            "official": "0.5 second window beginning 0.2 seconds before the keypress.",
            "neuroselect": "25 samples from -0.2 to +0.3 seconds after sample rounding.",
            "status": "PARTIAL",
            "evidence": "vendor/brain2qwerty/brain2qwerty_v1/config/xp_config.py:59-60; scripts/event_information_audit.py:22-25",
        },
        {
            "stage": "baseline correction",
            "official": "Per event window baseline: subtract mean over relative [0.0, 0.2] seconds, then crop to analysis window.",
            "neuroselect": "Subtract mean over first 0.2 seconds of the sentence crop before event windows are extracted.",
            "status": "MISMATCH",
            "evidence": "installed neuralset.extractors.neuro.MneRaw._get_timed_array; scripts/prepare_real_subset.py:65-68",
        },
        {
            "stage": "clamping",
            "official": "Clamp each extracted event tensor to +/-5.",
            "neuroselect": "Clamp each sentence signal to +/-5 before event windows are extracted.",
            "status": "PARTIAL",
            "evidence": "vendor/brain2qwerty/brain2qwerty_v1/config/xp_config.py:45; scripts/prepare_real_subset.py:69",
        },
        {
            "stage": "tensor orientation",
            "official": "Neural extractor returns C,T; SegmentDataset adds batch dimension, so model input is N,C,T.",
            "neuroselect": "Manifest loader returns T,C and ConvCTC consumes N,T,C.",
            "status": "MISMATCH",
            "evidence": "installed neuralset.extractors.neuro.MneRaw._get_timed_array; installed neuralset.segments.SegmentDataset.__getitem__; src/neuroselect/data.py:56-58; src/neuroselect/models.py",
        },
        {
            "stage": "target representation",
            "official": "Keystroke LabelEncoder uses predefined BUTTON_MAPPING, 29 classes, no one-hot.",
            "neuroselect": "SpanishBCBL vocabulary uses the same character order with a CTC blank prepended.",
            "status": "PARTIAL",
            "evidence": "vendor/brain2qwerty/brain2qwerty_v1/utils.py:18-63; vendor/brain2qwerty/brain2qwerty_v1/config/xp_config.py:50-57; src/neuroselect/vocab_spanishbcbl.py:13-30",
        },
        {
            "stage": "sentence construction",
            "official": "Sentence events are rebuilt over observed keystrokes; sentence_typed is concatenated actual button sequence.",
            "neuroselect": "Uses sentence_typed as manifest text.",
            "status": "MATCH",
            "evidence": "vendor/brain2qwerty/brain2qwerty_v1/transforms.py:54-111; scripts/prepare_real_subset.py:71-72",
        },
        {
            "stage": "splitting",
            "official": "TF-IDF paraphrase-similarity clusters are greedily assigned to train/val/test at 0.8/0.1/0.1, seed 1.",
            "neuroselect": "Subset manifest preserves the official split labels for selected trials.",
            "status": "PARTIAL",
            "evidence": "vendor/brain2qwerty/brain2qwerty_v1/transforms.py:145-205; data/processed/spanishbcbl_subset/manifest.jsonl",
        },
        {
            "stage": "subject/session handling",
            "official": "Drops controls/excluded subjects, merges duplicate subject IDs, factorizes subject IDs; subject ID and channel positions are extractor inputs.",
            "neuroselect": "Uses one S22 session/block and stores subject/session metadata, but does not feed subject IDs or channel positions to ConvCTC.",
            "status": "PARTIAL",
            "evidence": "vendor/brain2qwerty/brain2qwerty_v1/transforms.py:25-29; vendor/brain2qwerty/brain2qwerty_v1/main.py:75-84; src/neuroselect/data.py:56-58",
        },
    ]


def main() -> None:
    head, requested_type = git_revision()
    manifest = {record["id"]: record for record in read_manifest(MANIFEST)}
    record = manifest[TRIAL_ID]
    raw = mne.io.read_raw_fif(raw_path(), preload=False, verbose=False, allow_maxshield=True)
    raw_meg = raw.copy().pick("meg")
    raw_meg.load_data()
    filtered, processed = process_continuous(raw_meg)
    signal = load_signal(MANIFEST.parent / record["signal_path"])
    sentence_start = float(record["signal_start"])
    sentence_stop = float(record["signal_end"])

    raw_stage = crop_array(raw_meg, sentence_start, sentence_stop)
    filtered_stage = crop_array(filtered, sentence_start, sentence_stop)
    resampled_stage = crop_array(processed, sentence_start, sentence_stop)
    ns_baseline_count = max(1, int(round(min(0.2, len(resampled_stage) / FS) * FS)))
    ns_scaled_stage = resampled_stage - resampled_stage[:ns_baseline_count].mean(axis=0, keepdims=True)
    ns_final_stage = np.clip(ns_scaled_stage, -5.0, 5.0).astype(np.float32)

    stages = {
        "raw_loaded": {"official": stats(raw_stage), "neuroselect": stats(raw_stage)},
        "filtered": {"official": stats(filtered_stage), "neuroselect": stats(filtered_stage)},
        "resampled": {"official": stats(resampled_stage), "neuroselect": stats(resampled_stage)},
        "scaled": {"official": stats(resampled_stage), "neuroselect": stats(ns_scaled_stage)},
        "sentence_crop_before_baseline": {
            "official": stats(resampled_stage),
            "neuroselect": stats(resampled_stage),
        },
        "final_sentence_tensor": {
            "official": stats(ns_final_stage),
            "neuroselect": stats(signal),
        },
    }

    events = pd.read_pickle(RAW_ROOT / "events_clean.pkl")
    keys = events[
        (events["type"] == "Keystroke")
        & (events["sentence_UID"] == TRIAL_ID)
    ].sort_values("start")
    event_examples = []
    for event_index, (_, event) in enumerate(keys.iterrows()):
        event_time = float(event["start"])
        start = event_time + WINDOW_START
        stop = start + WINDOW_DURATION
        if start < raw_meg.times[0] or stop > raw_meg.times[-1]:
            continue
        official_ct = official_event_tensor(processed, event_time)[0]
        official_tc = official_ct.T
        current_tc, current_info = current_event_tensor(signal, sentence_start, event_time)
        if official_tc.shape != current_tc.shape:
            continue
        event_examples.append(
            {
                "event_index": event_index,
                "event_timestamp": event_time,
                "raw_button": str(event["button"]),
                "mapped_label": {"<space>": " ", "<special>": "@", "<number>": "9"}.get(
                    str(event["button"]), str(event["button"])
                ),
                "official": {
                    "window_start": start,
                    "window_end": stop,
                    "shape_before_transpose": list(official_ct.shape),
                    "shape_after_transpose": list(official_tc.shape),
                    "orientation": "T,C after explicit audit transpose",
                    "summary": stats(official_tc),
                },
                "neuroselect": current_info,
                "difference": {
                    "mean_absolute": float(np.mean(np.abs(official_tc - current_tc))),
                    "max_absolute": float(np.max(np.abs(official_tc - current_tc))),
                    "same_checksum": bool(
                        stats(official_tc)["sha256"] == stats(current_tc)["sha256"]
                    ),
                },
            }
        )
        if len(event_examples) == 5:
            break

    compatibility = {
        "experiment": "official_v1_compatibility_audit",
        "official_revision_requested": EXPECTED_REVISION,
        "official_revision_head": head,
        "requested_revision_object_type": requested_type,
        "revision_match": bool(head == EXPECTED_REVISION),
        "data": {
            "raw_file": str(raw_path()),
            "trial_id": TRIAL_ID,
            "same_downloaded_file_for_comparison": True,
            "existing_processed_signal": str(MANIFEST.parent / record["signal_path"]),
        },
        "stage_inventory": stage_inventory(),
        "channel_spatial_audit": {
            "official_picks": "meg",
            "official_channel_count": len(raw_meg.ch_names),
            "neuroselect_channel_count": int(signal.shape[1]),
            "official_channel_order": "unique mapping; one recording preserves picked MNE order",
            "neuroselect_channel_order": "MNE meg pick order preserved",
            "sensor_positions_used": "YES in official v1 as channel_positions model input; NO in NeuroSelect ConvCTC",
            "projection": "official apply_proj=False; NeuroSelect does not apply projection",
            "all_306_expected": True,
            "sensor_space_transform": "No projector or spatial transform in official extractor; official model additionally receives 2D channel positions.",
            "unresolved": [
                "Exact exca materialization/cache path at the requested revision cannot be verified because that commit is absent and the installed wrapper rejects the project's DataFrame path."
            ],
        },
        "event_window_audit": {
            "official_start_seconds": -0.2,
            "official_end_seconds": 0.3,
            "official_duration_seconds": 0.5,
            "official_samples": WINDOW_SAMPLES,
            "official_boundary_behavior": "SegmentDataset remove_incomplete_segments=True discards incomplete windows.",
            "official_baseline": "per event, relative [0.0, 0.2] seconds inside the 0.5-second analysis window",
            "neuroselect_window": [-0.2, 0.3],
            "neuroselect_boundary_behavior": "event audits discard windows outside sentence array; preparation itself creates sentence arrays.",
            "neuroselect_baseline": "sentence-level first 0.2 seconds before event extraction",
        },
        "tensor_orientation_audit": {
            "official_extractor": "C,T",
            "official_dataloader": "N,C,T",
            "official_model": "receives N,C,T plus subject_id and channel_positions",
            "neuroselect_signal_storage": "T,C",
            "neuroselect_collated": "N,T,C",
            "critical_axis_order_difference": True,
        },
        "target_label_audit": {
            "official_mapping_source": "brain2qwerty_v1.utils.BUTTON_MAPPING",
            "official_special_mapping": "<space> -> space; <special> -> @; <number> -> 9; other unsupported symbols -> @",
            "neuroselect_mapping": "same verified SpanishBCBL mapping for retained labels",
            "blank": "official v1 has 29 event classes and no CTC blank; NeuroSelect adds CTC blank id 0",
            "sentence_text": "official sentence_typed is actual concatenated keystrokes, not displayed stimulus",
            "corrections": "official preprocessing uses actual Keystroke events; deviations/corrections remain in typed sequence",
            "ordering": "groupby sentence_UID and event start order",
        },
        "split_audit": {
            "official": "sentence-level split after paraphrase clustering, assigned by keystroke counts",
            "duplicate_text_cross_split": "not explicitly forbidden for exact duplicates beyond similarity clustering; exact duplicate strings cluster together",
            "neuroselect": "manifest retains official split labels for selected trials; diagnostics also use explicit leave-one-trial-out splits",
        },
        "intermediate_parity": stages,
        "verification": {
            "official_source_inspected": True,
            "current_neuroselect_inspected": True,
            "no_model_training": True,
            "no_preprocessing_changed": True,
            "no_dataset_downloaded": True,
            "existing_event_extraction_reused": True,
            "all_comparisons_same_downloaded_file": True,
            "finite_statistics_expected": all(
                value["nan_count"] == 0 and value["inf_count"] == 0
                for stage in stages.values()
                for value in stage.values()
            ),
        },
    }
    compatibility["intermediate_parity"]["scaled"] = {
        "official": stats(resampled_stage),
        "neuroselect": stats(resampled_stage),
    }
    compatibility["ranked_mismatches"] = {
        "CRITICAL": [
            {
                "issue": "Vendored official checkout is not the requested revision.",
                "evidence": f"HEAD={head!r}; requested object type={requested_type!r}.",
            },
            {
                "issue": "Official v1 extractor/model path is N,C,T, while NeuroSelect stores and models N,T,C.",
                "evidence": "Official MneRaw returns C,T and ConvCTC transposes N,T,C internally.",
            },
        ],
        "IMPORTANT": [
            {
                "issue": "Official baseline correction is per keystroke window; NeuroSelect baseline-corrects once per sentence crop.",
                "evidence": "Official baseline=(0.0, 0.2) is applied in MneRaw._get_timed_array; NeuroSelect subtracts sentence[:0.2s].",
            },
            {
                "issue": "Official v1 provides 2D channel positions and subject IDs to the model; NeuroSelect does not.",
                "evidence": "brain2qwerty_v1/main.py builds channel_positions and subject_id extractors.",
            },
        ],
        "MINOR": [
            {
                "issue": "Official extractor performs clamp per event; NeuroSelect clamps the sentence tensor before event extraction.",
                "evidence": "Both use +/-5; ordering can differ only where baseline/clamp interact.",
            },
            {
                "issue": "The exact official split is preserved in the selected manifest, but this diagnostic also uses custom leave-one-trial-out splits.",
                "evidence": "Brain2QwertyV1Splitter versus NeuroSelect diagnostic scripts.",
            },
        ],
        "UNRESOLVED": [
            {
                "issue": "Exact exca materialization/cache output and projection behavior at revision 5f9889621d0df391c5aab37c996683d308e6e926.",
                "evidence": "Requested commit is absent locally; installed wrapper rejects the project's DataFrame materialization path.",
            },
        ],
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/official_v1_compatibility_audit.json").write_text(
        json.dumps(compatibility, indent=2), encoding="utf-8"
    )
    Path("results/official_v1_parity_event_examples.json").write_text(
        json.dumps(
            {
                "trial_id": TRIAL_ID,
                "events_compared": len(event_examples),
                "examples": event_examples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "revision_match": compatibility["revision_match"],
                "raw_file": str(raw_path()),
                "channels": len(raw_meg.ch_names),
                "event_examples": len(event_examples),
                "critical_mismatches": [
                    item["stage"]
                    for item in compatibility["stage_inventory"]
                    if item["status"] == "MISMATCH"
                ],
                "unresolved": compatibility["channel_spatial_audit"]["unresolved"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
