"""Combine the per-run expanded CTC results into the phase artifacts.

Reads every JSON written by `scripts/s22_expanded_ctc.py` and produces the
corrected CTC baseline artifact, the no-signal control artifact, and the two
figures. Pure aggregation: nothing is trained here.
"""

from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results/runs/s22_expanded_ctc"
CTC_OUT = ROOT / "results/s22_expanded_ctc_baseline.json"
CONTROL_OUT = ROOT / "results/s22_expanded_no_signal_control.json"
CTC_FIGURE = ROOT / "results/figures/debug/s22_expanded_ctc.png"
CONTROL_FIGURE = ROOT / "results/figures/debug/s22_no_signal_control.png"

SEEDS = (33, 123, 777)
PRIMARY_SPLIT = "C"
CONTROL_SPLIT = "C"
PREVIOUS_EIGHT_TRIAL_CTC = 0.9194
PREVIOUS_EIGHT_TRIAL_CONTROL = 1.082


def load_runs() -> dict[tuple[str, str, int], dict]:
    runs = {}
    for path in sorted(RUN_DIR.glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        runs[(run["split"], run["mode"], run["seed"])] = run
    return runs


def character_distribution(runs: list[dict], key: str) -> dict:
    counter: collections.Counter[str] = collections.Counter()
    for run in runs:
        for row in run[key]:
            counter.update(row["decoded"])
    total = sum(counter.values())
    if not total:
        return {"total_characters": 0, "distinct_characters": 0, "top_character_fraction": 0.0,
                "normalized_entropy": 0.0, "histogram": {}}
    probabilities = np.array([count / total for count in counter.values()])
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return {
        "total_characters": total,
        "distinct_characters": len(counter),
        "top_character": counter.most_common(1)[0][0],
        "top_character_fraction": counter.most_common(1)[0][1] / total,
        "normalized_entropy": entropy / math.log(29),
        "histogram": {character: count for character, count in counter.most_common()},
    }


def across_seeds(runs: list[dict], side: str) -> dict:
    def stack(key: str) -> list[float]:
        return [run[side][key] for run in runs]
    summary = {}
    for key in ("mean_cer", "mean_wer", "mean_blank_argmax_fraction", "mean_blank_probability",
                "mean_decoded_target_ratio", "mean_decoded_length", "mean_longest_run",
                "mean_adjacent_repeat_fraction", "mean_most_frequent_character_fraction",
                "mean_unique_decoded_characters"):
        values = stack(key)
        summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "per_seed": values}
    summary["empty_decodes_per_seed"] = [run[side]["empty_decodes"] for run in runs]
    summary["per_block"] = {
        block: {
            "mean_cer": float(np.mean([run[side]["per_block"][block]["mean_cer"] for run in runs])),
            "mean_wer": float(np.mean([run[side]["per_block"][block]["mean_wer"] for run in runs])),
            "mean_blank_argmax_fraction": float(np.mean([run[side]["per_block"][block]["mean_blank_argmax_fraction"] for run in runs])),
            "sentences": runs[0][side]["per_block"][block]["sentences"],
        }
        for block in runs[0][side]["per_block"]
    }
    return summary


def split_summary(runs: list[dict]) -> dict:
    reference = runs[0]
    return {
        "split": reference["split"],
        "split_name": reference["split_name"],
        "classification": reference["classification"],
        "train_blocks": reference["data"]["train_blocks"],
        "test_blocks": reference["data"]["test_blocks"],
        "train_sentences": reference["data"]["train_sentences"],
        "test_sentences": reference["data"]["test_sentences"],
        "train_events": reference["data"]["train_events"],
        "test_events": reference["data"]["test_events"],
        "seeds": [run["seed"] for run in runs],
        "batching": reference["batching"],
        "optimizer": reference["optimizer"],
        "ctc_loss": {
            "initial_per_seed": [run["initial_ctc_loss"] for run in runs],
            "final_per_seed": [run["final_ctc_loss"] for run in runs],
            "minimum_per_seed": [run["minimum_ctc_loss"] for run in runs],
        },
        "train": across_seeds(runs, "train_aggregate"),
        "evaluation": across_seeds(runs, "evaluation_aggregate"),
        "evaluation_character_distribution": character_distribution(runs, "evaluation_predictions"),
        "training_seconds_per_seed": [run["training_seconds"] for run in runs],
        "example_decodes": [
            {"seed": run["seed"], "sentence_UID": row["sentence_UID"], "block": row["block"],
             "target": row["target"], "decoded": row["decoded"], "cer": row["cer"]}
            for run in runs[:1] for row in run["evaluation_predictions"][:4]
        ],
        "per_sentence_evaluation": {
            str(run["seed"]): [
                {k: row[k] for k in ("sentence_UID", "block", "trial_id", "events", "target_length",
                                     "decoded_length", "cer", "wer", "blank_argmax_fraction",
                                     "decoded_target_ratio", "longest_run", "most_frequent_character_fraction")}
                for row in run["evaluation_predictions"]
            ]
            for run in runs
        },
    }


def main() -> None:
    runs = load_runs()
    real_splits = sorted({split for split, mode, _ in runs if mode == "real"})
    ctc_summaries = {}
    for split in real_splits:
        seed_runs = [runs[(split, "real", seed)] for seed in SEEDS if (split, "real", seed) in runs]
        if seed_runs:
            ctc_summaries[split] = split_summary(seed_runs)

    ctc_artifact = {
        "task": "Step 5 corrected event-sequence CTC baseline on the expanded S22 data",
        "representation": "official_v1_event_sequence: ordered (25,306) event tensors concatenated to (25U,306)",
        "model": "Conv1D + BiGRU + Linear + CTC (unchanged ConvCTC)",
        "decoder": "existing verified greedy CTC decoder",
        "vocabulary": "SpanishBCBL 30-class vocabulary including blank",
        "seeds": list(SEEDS),
        "primary_split": PRIMARY_SPLIT,
        "primary_split_reason": "C is sentence-disjoint and spans both sessions: train on both list1 blocks, evaluate on both list2 blocks.",
        "secondary_splits": [split for split in real_splits if split != PRIMARY_SPLIT],
        "splits_excluded_from_primary_results": ["F", "G"],
        "splits_excluded_reason": "F and G are CROSS-SESSION WITH SENTENCE OVERLAP and cannot support a generalization claim.",
        "splits": ctc_summaries,
        "verification": {
            "architecture_changed": False,
            "decoder_changed": False,
            "vocabulary_changed": False,
            "official_v1_preprocessing": True,
            "zero_padding_in_batching": False,
            "llm_used": False,
            "evidence_selector_used": False,
        },
    }
    CTC_OUT.write_text(json.dumps(ctc_artifact, indent=2, ensure_ascii=False), encoding="utf-8")

    control_runs = [runs[(CONTROL_SPLIT, "control", seed)] for seed in SEEDS if (CONTROL_SPLIT, "control", seed) in runs]
    control_artifact = None
    if control_runs:
        control_summary = split_summary(control_runs)
        real_summary = ctc_summaries[CONTROL_SPLIT]

        def delta(path: str) -> dict:
            real_value = real_summary["evaluation"][path]["mean"]
            control_value = control_summary["evaluation"][path]["mean"]
            return {
                "real_labels": real_value,
                "permuted_labels": control_value,
                "difference_control_minus_real": control_value - real_value,
            }

        control_artifact = {
            "task": "Step 6 no-signal target-permutation control on split C",
            "procedure": {
                "signals": "unchanged official-v1 event sequences",
                "sequence_lengths": "unchanged",
                "targets_permuted": "training sentences only",
                "shuffle_seed": control_runs[0]["target_permutation"]["shuffle_seed"],
                "derangement_enforced": True,
                "fixed_points": control_runs[0]["target_permutation"]["fixed_points"],
                "same_permutation_for_all_model_seeds": True,
                "evaluation_targets": "correctly paired, never permuted",
            },
            "target_permutation": control_runs[0]["target_permutation"],
            "control": control_summary,
            "real_reference": {
                "split": CONTROL_SPLIT,
                "evaluation": real_summary["evaluation"],
                "evaluation_character_distribution": real_summary["evaluation_character_distribution"],
            },
            "comparison": {
                "evaluation_cer": delta("mean_cer"),
                "evaluation_wer": delta("mean_wer"),
                "blank_argmax_fraction": delta("mean_blank_argmax_fraction"),
                "decoded_target_ratio": delta("mean_decoded_target_ratio"),
                "most_frequent_character_fraction": delta("mean_most_frequent_character_fraction"),
                "character_frequency_concentration": {
                    "real_top_character_fraction": real_summary["evaluation_character_distribution"]["top_character_fraction"],
                    "control_top_character_fraction": control_summary["evaluation_character_distribution"]["top_character_fraction"],
                    "real_normalized_entropy": real_summary["evaluation_character_distribution"]["normalized_entropy"],
                    "control_normalized_entropy": control_summary["evaluation_character_distribution"]["normalized_entropy"],
                },
                "train_ctc_loss_final": {
                    "real_labels": real_summary["ctc_loss"]["final_per_seed"],
                    "permuted_labels": control_summary["ctc_loss"]["final_per_seed"],
                },
                "significance_testing_performed": False,
                "significance_note": "Three seeds on one subject. No significance test is claimed.",
            },
            "previous_eight_trial_control_mean_eval_cer": PREVIOUS_EIGHT_TRIAL_CONTROL,
        }
        CONTROL_OUT.write_text(json.dumps(control_artifact, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- CTC figure ---------------------------------------------------
    CTC_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    order = [PRIMARY_SPLIT] + [s for s in real_splits if s != PRIMARY_SPLIT]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    positions = np.arange(len(order))
    width = 0.35
    train_cer = [ctc_summaries[s]["train"]["mean_cer"]["mean"] for s in order]
    eval_cer = [ctc_summaries[s]["evaluation"]["mean_cer"]["mean"] for s in order]
    eval_std = [ctc_summaries[s]["evaluation"]["mean_cer"]["std"] for s in order]
    axes[0].bar(positions - width / 2, train_cer, width, label="train CER", color="#3a6ea5")
    axes[0].bar(positions + width / 2, eval_cer, width, yerr=eval_std, capsize=4, label="held-out CER", color="#e0a458")
    axes[0].axhline(PREVIOUS_EIGHT_TRIAL_CTC, color="grey", linestyle="--", linewidth=1.2,
                    label=f"8-trial dev eval CER = {PREVIOUS_EIGHT_TRIAL_CTC}")
    axes[0].axhline(1.0, color="black", linestyle=":", linewidth=1.0, label="CER = 1.0")
    for position, value in zip(positions, eval_cer):
        axes[0].text(position + width / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(positions, [f"{s}{' (primary)' if s == PRIMARY_SPLIT else ''}" for s in order])
    axes[0].set_ylabel("CER")
    axes[0].set_title("Expanded-data CTC: CER by split (3 seeds)")
    axes[0].legend(fontsize=7.5)

    axes[1].bar(positions, [ctc_summaries[s]["evaluation"]["mean_blank_argmax_fraction"]["mean"] for s in order],
                0.5, color="#7a4988")
    for position, split in zip(positions, order):
        value = ctc_summaries[split]["evaluation"]["mean_blank_argmax_fraction"]["mean"]
        axes[1].text(position, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(positions, order)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("blank argmax fraction")
    axes[1].set_title("Held-out blank regime")

    for split in order:
        axes[2].plot(ctc_summaries[split]["ctc_loss"]["final_per_seed"], marker="o", label=f"{split} final")
    axes[2].set_xticks(range(len(SEEDS)), [str(seed) for seed in SEEDS])
    axes[2].set_xlabel("seed")
    axes[2].set_ylabel("final training CTC loss")
    axes[2].set_title("Final training loss per seed")
    axes[2].legend(fontsize=8)
    figure.suptitle("Step 5: corrected event-sequence CTC baseline on 9,650 keystrokes", fontsize=12)
    figure.tight_layout()
    figure.savefig(CTC_FIGURE, dpi=160)
    plt.close(figure)

    # ---- control figure -----------------------------------------------
    if control_artifact:
        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        labels = ["real labels", "permuted labels"]
        colours = ["#3a6ea5", "#c8553d"]
        cer_values = [real_summary["evaluation"]["mean_cer"]["mean"], control_summary["evaluation"]["mean_cer"]["mean"]]
        cer_errors = [real_summary["evaluation"]["mean_cer"]["std"], control_summary["evaluation"]["mean_cer"]["std"]]
        axes[0].bar(labels, cer_values, yerr=cer_errors, capsize=5, color=colours)
        for index, value in enumerate(cer_values):
            axes[0].text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        axes[0].axhline(1.0, color="black", linestyle=":", linewidth=1.0)
        axes[0].set_ylabel("held-out CER")
        axes[0].set_title("Split C held-out CER")

        keys = [("mean_blank_argmax_fraction", "blank fraction"),
                ("mean_decoded_target_ratio", "decoded/target length"),
                ("mean_most_frequent_character_fraction", "top-char fraction")]
        positions = np.arange(len(keys))
        width = 0.35
        axes[1].bar(positions - width / 2, [real_summary["evaluation"][k]["mean"] for k, _ in keys], width,
                    label="real labels", color=colours[0])
        axes[1].bar(positions + width / 2, [control_summary["evaluation"][k]["mean"] for k, _ in keys], width,
                    label="permuted labels", color=colours[1])
        axes[1].set_xticks(positions, [label for _, label in keys], fontsize=9)
        axes[1].set_title("Held-out output statistics")
        axes[1].legend(fontsize=8)

        real_hist = real_summary["evaluation_character_distribution"]["histogram"]
        control_hist = control_summary["evaluation_character_distribution"]["histogram"]
        characters = sorted(set(real_hist) | set(control_hist),
                            key=lambda c: -(real_hist.get(c, 0) + control_hist.get(c, 0)))[:12]
        positions = np.arange(len(characters))
        real_total = max(real_summary["evaluation_character_distribution"]["total_characters"], 1)
        control_total = max(control_summary["evaluation_character_distribution"]["total_characters"], 1)
        axes[2].bar(positions - width / 2, [real_hist.get(c, 0) / real_total for c in characters], width,
                    label="real labels", color=colours[0])
        axes[2].bar(positions + width / 2, [control_hist.get(c, 0) / control_total for c in characters], width,
                    label="permuted labels", color=colours[1])
        axes[2].set_xticks(positions, [repr(c)[1:-1] if c != " " else "space" for c in characters], fontsize=8)
        axes[2].set_ylabel("share of decoded characters")
        axes[2].set_title("Decoded character-frequency concentration")
        axes[2].legend(fontsize=8)
        figure.suptitle("Step 6: target-permutation negative control (split C, shuffle seed 2026)", fontsize=12)
        figure.tight_layout()
        figure.savefig(CONTROL_FIGURE, dpi=160)
        plt.close(figure)

    print(json.dumps({
        "runs_found": sorted(f"{split}/{mode}/{seed}" for split, mode, seed in runs),
        "ctc_artifact": str(CTC_OUT.relative_to(ROOT)),
        "control_artifact": str(CONTROL_OUT.relative_to(ROOT)) if control_artifact else None,
        "evaluation_cer": {split: ctc_summaries[split]["evaluation"]["mean_cer"]["mean"] for split in ctc_summaries},
        "control_evaluation_cer": control_summary["evaluation"]["mean_cer"]["mean"] if control_runs else None,
    }, indent=2))


if __name__ == "__main__":
    main()
