"""Build a small NeuroSelect manifest from the official SpanishBCBL pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

import studies
from brain2qwerty_v1.transforms import Brain2QwertyV1Splitter, SpanishBCBLPreprocessing
from neuralset.events import Study


def build_events(data_root: Path) -> pd.DataFrame:
    study = Study(name="Pinet2024Meg", path=data_root)
    events = study.run()
    return SpanishBCBLPreprocessing()._run(events)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/spanishbcbl_subset"))
    parser.add_argument("--trials", type=int, nargs="+", default=list(range(2, 10)))
    args = parser.parse_args()

    events = build_events(args.data_root)
    args.data_root.joinpath("events_clean.pkl").parent.mkdir(parents=True, exist_ok=True)
    events.to_pickle(args.data_root / "events_clean.pkl")
    events = Brain2QwertyV1Splitter()._run(events)
    sentences = events[
        (events["type"] == "Sentence")
        & (events["is_percep"] == False)  # noqa: E712
        & events["trial_id"].isin(args.trials)
    ].copy()
    if len(sentences) < 3:
        raise RuntimeError("Fewer than three complete production sentences selected")

    candidates = list((args.data_root / "MEG" / "FIF").rglob("*.fif"))
    candidates = [p for p in candidates if "tapping" not in p.name.lower()]
    if len(candidates) != 1:
        raise RuntimeError("Could not uniquely identify the downloaded MEG file")
    raw_path = candidates[0]

    raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False, allow_maxshield=True)
    raw = raw.pick("meg")
    args.output_root.mkdir(parents=True, exist_ok=True)
    signal_root = args.output_root / "signals"
    signal_root.mkdir(exist_ok=True)
    records = []
    alignment_examples = []
    for _, sentence in sentences.sort_values("trial_id").iterrows():
        start = float(sentence["start"])
        end = float(sentence["stop"])
        segment = raw.copy().crop(tmin=start, tmax=end, include_tmax=False)
        segment.load_data()
        segment.filter(0.1, 20.0, n_jobs=1, verbose=False)
        segment.resample(50.0, npad="auto", n_jobs=1, verbose=False)
        data = segment.get_data().T.astype(np.float32, copy=False)
        data = RobustScaler().fit_transform(data).astype(np.float32)
        data = np.clip(data, -5.0, 5.0)
        signal_name = f"trial_{int(sentence['trial_id']):03d}.npy"
        np.save(signal_root / signal_name, data)
        target = str(sentence["sentence_typed"])
        record = {
            "id": str(sentence["sentence_UID"]),
            "signal_path": f"signals/{signal_name}",
            "text": target,
            "subject": str(sentence["subject"]),
            "session": str(sentence["session"]),
            "split": str(sentence["split"]),
            "vocab": "spanishbcbl",
            "source_file": str(raw_path),
            "modality": "MEG",
            "number_of_channels": int(data.shape[1]),
            "sampling_rate": 50.0,
            "recording_start": start,
            "recording_end": end,
            "event_count": int((events["sentence_UID"] == sentence["sentence_UID"]).sum()),
            "sentence_id": str(sentence["sentence_UID"]),
            "signal_start": start,
            "signal_end": end,
            "signal_duration": end - start,
            "key_count": int(
                ((events["type"] == "Keystroke") & (events["sentence_UID"] == sentence["sentence_UID"])).sum()
            ),
        }
        records.append(record)
        if len(alignment_examples) < 3:
            keys = events[
                (events["type"] == "Keystroke")
                & (events["sentence_UID"] == sentence["sentence_UID"])
            ].sort_values("start")
            alignment_examples.append(
                {
                    "trial_id": int(sentence["trial_id"]),
                    "sentence_uid": str(sentence["sentence_UID"]),
                    "ground_truth": target,
                    "key_count": len(keys),
                    "signal_duration": end - start,
                    "first_event": keys.iloc[0]["start"],
                    "last_event": keys.iloc[-1]["stop"],
                    "label_sequence": target,
                }
            )

    manifest = args.output_root / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    Path("data/manifests").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv("data/manifests/spanishbcbl_meg_subset.csv", index=False)
    Path("results").mkdir(exist_ok=True)
    Path("results/real_data_alignment_examples.json").write_text(
        json.dumps(alignment_examples, indent=2), encoding="utf-8"
    )
    try:
        import matplotlib.pyplot as plt

        first = np.load(signal_root / "trial_002.npy")
        figure_dir = Path("results/figures")
        figure_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.plot(first[:, :5])
        plt.title("SpanishBCBL trial 002: first five MEG channels")
        plt.xlabel("50 Hz sample")
        plt.ylabel("robust-scaled amplitude")
        plt.tight_layout()
        plt.savefig(figure_dir / "spanishbcbl_trial_002.png", dpi=120)
        plt.close()
    except ImportError:
        pass
    print(json.dumps({"trials": len(records), "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
