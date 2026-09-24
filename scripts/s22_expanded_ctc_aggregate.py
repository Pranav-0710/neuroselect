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


def edit_components(reference: str, hypothesis: str) -> tuple[int, int, int, int]:
    """Levenshtein alignment with backtrace.

    Returns (substitutions, deletions, insertions, hits) against `reference`,
    so that CER == (S + D + I) / len(reference). A deletion is a reference
    character missing from the hypothesis; an insertion is a hypothesis
    character absent from the reference.
    """
    rows, columns = len(reference) + 1, len(hypothesis) + 1
    cost = [[0] * columns for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(columns):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, columns):
            cost[i][j] = min(
                cost[i - 1][j] + 1,                                        # deletion
                cost[i][j - 1] + 1,                                        # insertion
                cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
            )
    substitutions = deletions = insertions = hits = 0
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] and reference[i - 1] == hypothesis[j - 1]:
            hits += 1
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + 1:
            substitutions += 1
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    return substitutions, deletions, insertions, hits


def rates(totals: dict) -> dict:
    """Length-sensitive rates plus the length-robust aligned-character F1."""
    reference = max(totals["reference_characters"], 1)
    hypothesis = max(totals["hypothesis_characters"], 1)
    precision = totals["hits"] / hypothesis
    recall = totals["hits"] / reference
    return {
        "substitution_rate": totals["substitutions"] / reference,
        "deletion_rate": totals["deletions"] / reference,
        "insertion_rate": totals["insertions"] / reference,
        "micro_cer": (totals["substitutions"] + totals["deletions"] + totals["insertions"]) / reference,
        "hit_rate": recall,
        "character_precision": precision,
        "character_recall": recall,
        "character_f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
        "length_ratio": totals["hypothesis_characters"] / reference,
    }


def error_decomposition(runs: list[dict], side: str) -> dict:
    """Aggregate S/D/I over every held-out sentence of every seed.

    Reported because CER alone cannot separate "decodes better" from "emits a
    shorter string": CER is normalized by target length, so a short hypothesis
    caps its own insertion count.
    """
    totals = {"substitutions": 0, "deletions": 0, "insertions": 0, "hits": 0,
              "reference_characters": 0, "hypothesis_characters": 0}
    per_seed = []
    mismatches = 0
    for run in runs:
        seed_totals = dict.fromkeys(totals, 0)
        for row in run[side]:
            s, d, i, hits = edit_components(row["target"], row["decoded"])
            if abs((s + d + i) / max(len(row["target"]), 1) - row["cer"]) > 1e-9:
                mismatches += 1
            for key, value in (("substitutions", s), ("deletions", d), ("insertions", i),
                               ("hits", hits), ("reference_characters", len(row["target"])),
                               ("hypothesis_characters", len(row["decoded"]))):
                seed_totals[key] += value
                totals[key] += value
        per_seed.append({
            "seed": run["seed"],
            **rates(seed_totals),
            **{f"{k}_count": v for k, v in seed_totals.items()},
        })
    return {
        "definition": "CER = (S + D + I) / N, N = target characters; counts pooled over all seeds",
        "length_robust_note": (
            "CER and hit recall are both length-sensitive in opposite directions: a shorter hypothesis "
            "caps its insertions and so lowers CER, while a longer hypothesis gets more chances to align "
            "a character and so raises recall. character_f1 is the harmonic mean of aligned-character "
            "precision and recall and is the length-robust comparison."
        ),
        "counts": totals,
        **rates(totals),
        "per_seed": per_seed,
        "components_disagree_with_reported_cer": mismatches,
    }


def unrelated_sentence_reference(real_runs: list[dict]) -> dict:
    """What character F1 does an unrelated real Spanish sentence already score?

    No model and no randomness: every held-out sentence is scored against the
    training sentence closest to it in length (ties broken by sentence UID).
    This anchors the F1 scale, because Spanish letter statistics and a roughly
    correct length alone recover a non-trivial share of characters.
    """
    run = real_runs[0]
    training = sorted(
        {row["sentence_UID"]: row["target"] for row in run["train_predictions"]}.items()
    )
    totals = {"substitutions": 0, "deletions": 0, "insertions": 0, "hits": 0,
              "reference_characters": 0, "hypothesis_characters": 0}
    for row in run["evaluation_predictions"]:
        target = row["target"]
        _, hypothesis = min(training, key=lambda item: (abs(len(item[1]) - len(target)), item[0]))
        s, d, i, hits = edit_components(target, hypothesis)
        for key, value in (("substitutions", s), ("deletions", d), ("insertions", i), ("hits", hits),
                           ("reference_characters", len(target)), ("hypothesis_characters", len(hypothesis))):
            totals[key] += value
    return {
        "what_it_is": "each held-out sentence scored against the length-matched training sentence; no model involved",
        "counts": totals,
        **rates(totals),
    }


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
        "evaluation_error_decomposition": error_decomposition(runs, "evaluation_predictions"),
        "unrelated_sentence_reference": unrelated_sentence_reference(runs),
        "train_error_decomposition": error_decomposition(runs, "train_predictions"),
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
        "reporting_framework": {
            "primary_metric": "character_f1 (aligned-character precision/recall harmonic mean), under evaluation_error_decomposition",
            "primary_metric_reason": (
                "CER and recall are length-sensitive in opposite directions. Within the real-label "
                "condition alone, held-out CER moves substantially with the decoded length ratio while "
                "character F1 stays essentially flat, so CER conflates which characters are recovered "
                "with how aggressively the decoder emits or withholds them."
            ),
            "secondary_metrics": "S/D/I rates per target character and the decoded-length ratio",
            "control": "target permutation, a no-valid-signal-assignment control, not a competing decoder",
            "reference": "length-matched training-sentence textual anchor (no model)",
            "cer": "reported for continuity with earlier phases, explicitly interpreted as sensitive to output length",
        },
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
            "what_this_control_is": (
                "A no-valid-signal-assignment control. Permuting the training targets destroys the "
                "signal-to-sentence correspondence while leaving the signals, the sequence lengths and "
                "the optimization untouched. It is not a competing decoder and its CER is not an "
                "accuracy baseline: it bounds what this setup produces when no valid assignment exists."
            ),
            "unrelated_sentence_reference": real_summary["unrelated_sentence_reference"],
            "error_decomposition_comparison": {
                "why": (
                    "CER is normalized by target length, so a hypothesis that is simply shorter caps "
                    "its own insertion count and can score a lower CER without decoding anything better. "
                    "Reporting S, D and I separately alongside the length ratio shows directly whether "
                    "a CER difference comes from better character recovery or from output length."
                ),
                "real_labels": {
                    k: real_summary["evaluation_error_decomposition"][k]
                    for k in ("substitution_rate", "deletion_rate", "insertion_rate", "hit_rate",
                              "character_precision", "character_recall", "character_f1",
                              "micro_cer", "length_ratio", "counts")
                },
                "permuted_labels": {
                    k: control_summary["evaluation_error_decomposition"][k]
                    for k in ("substitution_rate", "deletion_rate", "insertion_rate", "hit_rate",
                              "character_precision", "character_recall", "character_f1",
                              "micro_cer", "length_ratio", "counts")
                },
                "difference_control_minus_real": {
                    k: control_summary["evaluation_error_decomposition"][k]
                       - real_summary["evaluation_error_decomposition"][k]
                    for k in ("substitution_rate", "deletion_rate", "insertion_rate", "hit_rate",
                              "character_precision", "character_recall", "character_f1",
                              "micro_cer", "length_ratio")
                },
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
            "conclusion": (
                "Under this protocol, the correctly paired neural condition does not yet demonstrate an "
                "advantage over the no-valid-signal-assignment control on length-robust character F1."
            ),
            "conclusion_scope": [
                "One subject (S22).",
                "One small set of recordings from that subject: four blocks across two sessions.",
                "Three model seeds per condition.",
                "The comparison is made on the split C evaluation partition.",
                "This is not evidence that neural information is absent; it is a failure of this protocol "
                "to demonstrate an advantage.",
            ],
            "anchor_wording": (
                "A length-matched training-sentence textual anchor achieved higher character F1 than the "
                "neural decoder on this development set. The anchor is well-formed Spanish text and so "
                "benefits from ordinary language structure, while the decoder emits character soup; the "
                "comparison shows that the decoder's output does not exceed a simple text prior in "
                "character-level structure."
            ),
        }
        CONTROL_OUT.write_text(json.dumps(control_artifact, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- CTC figure ---------------------------------------------------
    # Character F1 leads; CER is shown beside the length ratio it tracks.
    CTC_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    order = [PRIMARY_SPLIT] + [s for s in real_splits if s != PRIMARY_SPLIT]
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    positions = np.arange(len(order))
    width = 0.35

    top_left = axes[0][0]
    f1 = [ctc_summaries[s]["evaluation_error_decomposition"]["character_f1"] for s in order]
    per_seed_f1 = [[row["character_f1"] for row in ctc_summaries[s]["evaluation_error_decomposition"]["per_seed"]]
                   for s in order]
    top_left.bar(positions, f1, 0.55, color="#3a6ea5", label="held-out character F1")
    for position, values in zip(positions, per_seed_f1):
        top_left.scatter([position] * len(values), values, color="black", zorder=3, s=18,
                         label="per seed" if position == 0 else None)
    for position, split in zip(positions, order):
        anchor_value = ctc_summaries[split]["unrelated_sentence_reference"]["character_f1"]
        top_left.plot([position - 0.3, position + 0.3], [anchor_value, anchor_value], color="#c8553d",
                      linewidth=2, label="length-matched text anchor" if position == 0 else None)
        top_left.text(position, f1[position], f"{f1[position]:.3f}", ha="center", va="bottom", fontsize=9)
    if control_artifact:
        value = control_summary["evaluation_error_decomposition"]["character_f1"]
        top_left.plot([-0.3, 0.3], [value, value], color="#7a4988", linewidth=2, linestyle="--",
                      label="no-valid-assignment control (C)")
    top_left.set_xticks(positions, [f"{s}{' (primary)' if s == PRIMARY_SPLIT else ''}" for s in order])
    top_left.set_ylabel("aligned-character F1")
    top_left.set_title("PRIMARY: held-out character F1 (length-robust)")
    top_left.legend(fontsize=7.5)

    top_right = axes[0][1]
    eval_cer = [ctc_summaries[s]["evaluation_error_decomposition"]["micro_cer"] for s in order]
    length = [ctc_summaries[s]["evaluation_error_decomposition"]["length_ratio"] for s in order]
    top_right.bar(positions - width / 2, eval_cer, width, label="held-out CER", color="#e0a458")
    top_right.bar(positions + width / 2, length, width, label="decoded / target length", color="#9bbfd4")
    top_right.axhline(PREVIOUS_EIGHT_TRIAL_CTC, color="grey", linestyle="--", linewidth=1.2,
                      label=f"8-trial dev eval CER = {PREVIOUS_EIGHT_TRIAL_CTC}")
    for position, value in zip(positions, eval_cer):
        top_right.text(position - width / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    top_right.set_xticks(positions, order)
    top_right.set_title("CER is length-sensitive: CER beside decoded-length ratio")
    top_right.legend(fontsize=7.5)

    bottom_left = axes[1][0]
    components = [("substitution_rate", "S/N"), ("deletion_rate", "D/N"), ("insertion_rate", "I/N")]
    bars = 0.8 / len(components)
    for index, (key, label) in enumerate(components):
        bottom_left.bar(positions + (index - 1) * bars,
                        [ctc_summaries[s]["evaluation_error_decomposition"][key] for s in order],
                        bars, label=label)
    bottom_left.set_xticks(positions, order)
    bottom_left.set_ylabel("rate per target character")
    bottom_left.set_title("Held-out error decomposition: CER = (S + D + I) / N")
    bottom_left.legend(fontsize=8)

    bottom_right = axes[1][1]
    train_f1 = [ctc_summaries[s]["train_error_decomposition"]["character_f1"] for s in order]
    bottom_right.bar(positions - width / 2, train_f1, width, label="training character F1", color="#3a6ea5")
    bottom_right.bar(positions + width / 2, f1, width, label="held-out character F1", color="#e0a458")
    for position, value in zip(positions, train_f1):
        bottom_right.text(position - width / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    bottom_right.set_xticks(positions, order)
    bottom_right.set_ylim(0, 1.1)
    bottom_right.set_ylabel("aligned-character F1")
    bottom_right.set_title("Training fit vs held-out transfer")
    bottom_right.legend(fontsize=8)

    figure.suptitle("Step 5: event-sequence CTC on 9,650 keystrokes, 256 sentences, 4 blocks, 2 sessions",
                    fontsize=13)
    figure.tight_layout()
    figure.savefig(CTC_FIGURE, dpi=160)
    plt.close(figure)

    # ---- control figure -----------------------------------------------
    if control_artifact:
        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        labels = ["real labels", "permuted labels"]
        colours = ["#3a6ea5", "#c8553d"]
        width = 0.35
        real_f1 = [row["character_f1"] for row in real_summary["evaluation_error_decomposition"]["per_seed"]]
        control_f1 = [row["character_f1"] for row in control_summary["evaluation_error_decomposition"]["per_seed"]]
        pooled = [real_summary["evaluation_error_decomposition"]["character_f1"],
                  control_summary["evaluation_error_decomposition"]["character_f1"]]
        axes[0].bar(labels, pooled, 0.55, color=colours)
        for index, values in enumerate([real_f1, control_f1]):
            axes[0].scatter([index] * len(values), values, color="black", zorder=3, s=22,
                            label="per seed" if index == 0 else None)
            axes[0].text(index, pooled[index], f"{pooled[index]:.3f}", ha="center", va="bottom", fontsize=9)
        anchor_value = real_summary["unrelated_sentence_reference"]["character_f1"]
        axes[0].axhline(anchor_value, color="#2e7d32", linewidth=2, linestyle="--",
                        label=f"length-matched text anchor = {anchor_value:.3f}")
        axes[0].set_ylabel("aligned-character F1")
        axes[0].set_title("PRIMARY: held-out character F1 (length-robust)")
        axes[0].legend(fontsize=7.5)

        keys = [("micro_cer", "CER"), ("length_ratio", "decoded/target length")]
        positions = np.arange(len(keys))
        real_errors_side = real_summary["evaluation_error_decomposition"]
        control_errors_side = control_summary["evaluation_error_decomposition"]
        axes[1].bar(positions - width / 2, [real_errors_side[k] for k, _ in keys], width,
                    label="real labels", color=colours[0])
        axes[1].bar(positions + width / 2, [control_errors_side[k] for k, _ in keys], width,
                    label="permuted labels", color=colours[1])
        for position, (key, _) in zip(positions, keys):
            axes[1].text(position - width / 2, real_errors_side[key], f"{real_errors_side[key]:.3f}",
                         ha="center", va="bottom", fontsize=8)
            axes[1].text(position + width / 2, control_errors_side[key], f"{control_errors_side[key]:.3f}",
                         ha="center", va="bottom", fontsize=8)
        axes[1].set_xticks(positions, [label for _, label in keys], fontsize=9)
        axes[1].set_title("Lower control CER tracks its shorter output")
        axes[1].legend(fontsize=8)

        real_errors = real_summary["evaluation_error_decomposition"]
        control_errors = control_summary["evaluation_error_decomposition"]
        components = [("substitution_rate", "subst."), ("deletion_rate", "delet."),
                      ("insertion_rate", "insert."), ("character_recall", "recall"),
                      ("character_precision", "precision"), ("character_f1", "char F1")]
        positions = np.arange(len(components))
        axes[2].bar(positions - width / 2, [real_errors[k] for k, _ in components], width,
                    label="real labels", color=colours[0])
        axes[2].bar(positions + width / 2, [control_errors[k] for k, _ in components], width,
                    label="permuted labels", color=colours[1])
        for position, (key, _) in zip(positions, components):
            axes[2].text(position - width / 2, real_errors[key], f"{real_errors[key]:.2f}",
                         ha="center", va="bottom", fontsize=7)
            axes[2].text(position + width / 2, control_errors[key], f"{control_errors[key]:.2f}",
                         ha="center", va="bottom", fontsize=7)
        axes[2].set_xticks(positions, [label for _, label in components], fontsize=9)
        axes[2].set_ylabel("rate per target character")
        axes[2].set_title("Held-out error decomposition: CER = (S + D + I) / N")
        axes[2].legend(fontsize=8)
        figure.suptitle("Step 6: no-valid-signal-assignment control (split C, target permutation, shuffle seed 2026)", fontsize=12)
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
