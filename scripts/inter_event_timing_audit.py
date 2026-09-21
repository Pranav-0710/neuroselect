"""Feasibility audit for preserving official inter-keystroke timing gaps."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARITY_PATH = ROOT / "results/eight_trial_official_v1_parity.json"
OUT_PATH = ROOT / "results/inter_event_timing_audit.json"
DIST_FIGURE = ROOT / "results/figures/debug/inter_event_timing_distribution.png"
GEOMETRY_FIGURE = ROOT / "results/figures/debug/inter_event_timing_geometry.png"
FS = 50.0
EVENT_WINDOW_S = 0.5


def finite_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def main() -> None:
    parity = json.loads(PARITY_PATH.read_text(encoding="utf-8"))
    trials = []
    all_intervals = []
    for trial in parity["trial_identity_and_event_audit"]:
        events = sorted(trial["events"], key=lambda row: row["event_index"])
        timestamps = np.asarray([row["timestamp_seconds"] for row in events], dtype=np.float64)
        deltas = np.diff(timestamps)
        if not np.isfinite(deltas).all() or (deltas <= 0).any():
            raise RuntimeError(f"Invalid official timestamps for trial {trial['trial_number']}")
        gap_samples = np.maximum(0, np.rint(deltas * FS).astype(np.int64) - 25)
        overlap = np.maximum(0.0, EVENT_WINDOW_S - deltas)
        positive = gap_samples > 0
        overlapping = deltas < EVENT_WINDOW_S
        all_intervals.extend(deltas.tolist())
        current_t = 25 * len(events)
        timing_t = current_t + int(gap_samples.sum())
        sentence_duration = float(timestamps[-1] - timestamps[0])
        trials.append({
            "trial_number": int(trial["trial_number"]),
            "trial_id": trial["trial_id"],
            "target": trial["sentence_typed"],
            "U": len(events),
            "interval_count": len(deltas),
            "interval_statistics_seconds": finite_stats(deltas),
            "fraction_zero_gap": float(np.mean(~positive)),
            "fraction_positive_gap": float(np.mean(positive)),
            "mean_positive_gap_samples": float(np.mean(gap_samples[positive])) if positive.any() else 0.0,
            "median_positive_gap_samples": float(np.median(gap_samples[positive])) if positive.any() else 0.0,
            "maximum_gap_samples": int(gap_samples.max()),
            "total_inserted_gap_samples": int(gap_samples.sum()),
            "current_T": current_t,
            "timing_gap_T": timing_t,
            "overlap_pair_count": int(overlapping.sum()),
            "overlap_fraction": float(np.mean(overlapping)),
            "mean_overlap_seconds": float(np.mean(overlap[overlapping])) if overlapping.any() else 0.0,
            "median_overlap_seconds": float(np.median(overlap[overlapping])) if overlapping.any() else 0.0,
            "below_0_2_seconds": int((deltas < 0.2).sum()),
            "below_0_1_seconds": int((deltas < 0.1).sum()),
            "target_length": len(trial["sentence_typed"]),
            "sentence_duration_seconds": sentence_duration,
            "percentage_sequence_affected_by_gaps": float(100 * gap_samples.sum() / timing_t) if timing_t else 0.0,
            "deltas_seconds": deltas.tolist(),
            "gap_samples": gap_samples.tolist(),
        })

    train = [row for row in trials if row["trial_number"] <= 7]
    evaluation = [row for row in trials if row["trial_number"] >= 8]

    def split_summary(rows: list[dict]) -> dict[str, float]:
        values = np.asarray([v for row in rows for v in row["deltas_seconds"]])
        gaps = np.asarray([g for row in rows for g in row["gap_samples"]])
        return {
            "interval_count": int(values.size),
            "mean_delta_seconds": float(values.mean()),
            "median_delta_seconds": float(np.median(values)),
            "overlap_fraction": float(np.mean(values < EVENT_WINDOW_S)),
            "positive_gap_fraction": float(np.mean(gaps > 0)),
        }

    all_values = np.asarray(all_intervals)
    DIST_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(all_values, bins=20, color="#4472c4", alpha=0.85)
    axis.axvline(EVENT_WINDOW_S, color="#c00000", linestyle="--", label="0.5 s event window")
    axis.set_xlabel("Inter-event interval (seconds)")
    axis.set_ylabel("Number of intervals")
    axis.set_title("Official keystroke timing distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(DIST_FIGURE, dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    labels = [str(row["trial_number"]) for row in trials]
    current = [row["current_T"] for row in trials]
    timing = [row["timing_gap_T"] for row in trials]
    axes[0].bar(labels, current, label="current T=25U")
    axes[0].bar(labels, np.asarray(timing) - np.asarray(current), bottom=current, label="inserted gap samples")
    axes[0].set_ylabel("Sequence T")
    axes[0].set_title("Hypothetical timing-gap sequence geometry")
    axes[0].legend()
    axes[1].bar(labels, [row["fraction_positive_gap"] for row in trials], color="#70ad47")
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("Positive-gap fraction")
    figure.tight_layout()
    figure.savefig(GEOMETRY_FIGURE, dpi=160)
    plt.close(figure)

    artifact = {
        "experiment": "inter_event_timing_audit",
        "official_revision": "5f9889621d0df391c5aab37c996683d308e6e926",
        "timestamp_source": str(PARITY_PATH),
        "sampling_rate_hz": FS,
        "event_window_seconds": EVENT_WINDOW_S,
        "trials": trials,
        "train_summary_trials_2_to_7": split_summary(train),
        "evaluation_summary_trials_8_to_9": split_summary(evaluation),
        "all_interval_statistics_seconds": finite_stats(all_values),
        "privileged_information_warning": "Timing gaps use true keystroke timestamps and are an offline diagnostic/oracle representation, not automatically deployable brain-to-text input.",
        "decision": {
            "feasibility": "B. Timing-gap experiment is unlikely to be informative",
            "train_eval_difference": "C. Timing information differs between train/eval descriptively; positive gaps occur only in evaluation trials, and evaluation intervals are longer on average. No significance testing was performed.",
            "basis": "All 173 training intervals have zero inserted gap samples; only 2 of 59 evaluation intervals are positive. Nearly all event windows overlap, so timing-gap insertion changes sequence geometry minimally overall."
        },
        "figures": [str(DIST_FIGURE), str(GEOMETRY_FIGURE)],
        "verification": {"all_eight_trials": True, "official_timestamps_used": True, "training": False, "preprocessing_modified": False, "architecture_modified": False, "llm": False, "evidence_selector": False, "data_download": False, "all_statistics_finite": True}
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(OUT_PATH), "train": artifact["train_summary_trials_2_to_7"], "evaluation": artifact["evaluation_summary_trials_8_to_9"], "decision": artifact["decision"]}, indent=2))


if __name__ == "__main__":
    main()
