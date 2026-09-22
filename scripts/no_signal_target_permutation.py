"""5V negative control: permute training targets across unchanged event-sequence signals."""

from __future__ import annotations

import collections
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPT = ROOT / "scripts/official_v1_event_sequence_ctc_baseline.py"
REAL_RESULTS_PATH = ROOT / "results/official_v1_event_sequence_ctc_baseline.json"
OUT_PATH = ROOT / "results/no_signal_target_permutation.json"
FIGURE_PATH = ROOT / "results/figures/debug/no_signal_target_permutation.png"
OUTPUTS_FIGURE_PATH = ROOT / "results/figures/debug/no_signal_control_outputs.png"
SHUFFLE_SEED = 2026

# Reuse the committed 5U adapter, model, training loop, decoder and metrics unchanged.
_spec = importlib.util.spec_from_file_location("event_sequence_baseline", BASELINE_SCRIPT)
baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline)


def derangement(trials: list[int], seed: int) -> tuple[dict[int, int], int]:
    """Draw permutations from one seeded generator until none is a fixed point."""
    rng = np.random.default_rng(seed)
    draws = 0
    while True:
        draws += 1
        order = [trials[i] for i in rng.permutation(len(trials))]
        if all(source != target for source, target in zip(trials, order)):
            return dict(zip(trials, order)), draws


def output_structure(rows: list[dict]) -> list[dict]:
    structure = []
    for row in rows:
        decoded = row["decoded"]
        counts = collections.Counter(decoded)
        top, top_count = counts.most_common(1)[0] if counts else ("", 0)
        structure.append({
            "trial_number": row["trial_number"],
            "decoded": decoded,
            "decoded_target_ratio": row["decoded_target_ratio"],
            "blank_argmax_fraction": row["blank_argmax_fraction"],
            "most_frequent_character": top,
            "most_frequent_character_fraction": top_count / len(decoded) if decoded else 0.0,
            "unique_decoded_characters": len(counts),
        })
    return structure


def main() -> None:
    real = json.loads(REAL_RESULTS_PATH.read_text(encoding="utf-8"))
    real_hashes = {row["trial_number"]: row["sequence_sha256"] for row in real["shape_audit"]}

    processor = baseline.NeuroSelectOfficialV1EventPreprocessing.from_raw(
        baseline.RAW_PATH, recording_start_seconds=308.0
    )
    items = baseline.build_items(processor)
    by_trial = {item["trial_number"]: item for item in items}
    train_trials = list(baseline.TRAIN_TRIALS)

    permutation, draws = derangement(train_trials, SHUFFLE_SEED)
    control_items = []
    for item in items:
        if item["trial_number"] in baseline.TRAIN_TRIALS:
            source = by_trial[permutation[item["trial_number"]]]
            # Same signal object; only the training target is replaced.
            control_items.append({**item, "target": source["target"], "target_ids": source["target_ids"].clone()})
        else:
            control_items.append(item)

    audit = []
    for item in control_items:
        number = item["trial_number"]
        original = by_trial[number]
        values = item["sequence"]
        audit.append({
            "trial_number": number,
            "trial_id": item["trial_id"],
            "split": "train" if number in baseline.TRAIN_TRIALS else "evaluation",
            "target_source_trial": permutation.get(number, number),
            "original_target": original["target"],
            "control_target": item["target"],
            "original_target_length": len(original["target"]),
            "control_target_length": len(item["target"]),
            "U_signal_events": item["events"],
            "T": int(values.shape[0]),
            "channels": int(values.shape[1]),
            "T_equals_25U": int(values.shape[0]) == 25 * item["events"],
            "T_at_least_control_target_length": int(values.shape[0]) >= len(item["target"]),
            "signal_sha256": baseline.sha256(values),
            "signal_identical_to_5U": baseline.sha256(values) == real_hashes[number],
            "signal_is_same_object_as_unpermuted": values is original["sequence"],
            "finite": bool(np.isfinite(values).all()),
        })
    train_audit = [row for row in audit if row["split"] == "train"]
    eval_audit = [row for row in audit if row["split"] == "evaluation"]
    train_texts = [by_trial[t]["target"] for t in train_trials]
    checks = {
        "all_six_training_targets_exactly_once": sorted(r["control_target"] for r in train_audit) == sorted(train_texts),
        "training_target_texts_all_distinct": len(set(train_texts)) == len(train_texts),
        "no_training_signal_paired_with_own_target": all(r["control_target"] != r["original_target"] for r in train_audit),
        "no_training_control_target_equals_any_evaluation_target": not {r["control_target"] for r in train_audit} & {r["original_target"] for r in eval_audit},
        "all_signals_identical_to_5U": all(r["signal_identical_to_5U"] for r in audit),
        "evaluation_trials_untouched": all(r["control_target"] == r["original_target"] for r in eval_audit),
        "event_order_unchanged": all(item["official_event_order"] for item in control_items),
        "T_equals_25U": all(r["T_equals_25U"] for r in audit),
        "all_tensors_finite": all(r["finite"] for r in audit),
        "ctc_geometry_feasible": all(r["T_at_least_control_target_length"] for r in audit),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Control audit failed: {checks}")

    # Record every ConvCTC forward input so the artifact shows what the model actually received.
    forward_inputs = collections.Counter()

    def record_input(module, args):
        if isinstance(module, baseline.ConvCTC):
            if len(args) != 1 or not isinstance(args[0], torch.Tensor):
                raise RuntimeError("ConvCTC received something other than a single tensor")
            forward_inputs[(tuple(args[0].shape), str(args[0].dtype))] += 1

    hook = torch.nn.modules.module.register_module_forward_pre_hook(record_input)
    try:
        seed_results = [baseline.train_seed(control_items, seed) for seed in baseline.SEEDS]
    finally:
        hook.remove()

    after_hashes = {item["trial_number"]: baseline.sha256(item["sequence"]) for item in control_items}
    signals_unchanged_after_training = all(after_hashes[n] == real_hashes[n] for n in after_hashes)

    real_by_seed = {result["seed"]: result for result in real["seed_results"]}
    comparison = []
    for result in seed_results:
        seed = result["seed"]
        real_seed = real_by_seed[seed]
        comparison.append({
            "seed": seed,
            "real_train_cer": real_seed["train_aggregate"]["mean_cer"],
            "control_train_cer": result["train_aggregate"]["mean_cer"],
            "real_eval_cer": real_seed["evaluation_aggregate"]["mean_cer"],
            "control_eval_cer": result["evaluation_aggregate"]["mean_cer"],
            "control_minus_real_eval_cer": result["evaluation_aggregate"]["mean_cer"] - real_seed["evaluation_aggregate"]["mean_cer"],
        })
    real_eval = [row["real_eval_cer"] for row in comparison]
    control_eval = [row["control_eval_cer"] for row in comparison]
    control_train = [row["control_train_cer"] for row in comparison]
    cross_seed = {
        "control_train_cer_mean": float(np.mean(control_train)),
        "control_train_cer_std": float(np.std(control_train)),
        "control_eval_cer_mean": float(np.mean(control_eval)),
        "control_eval_cer_std": float(np.std(control_eval)),
        "real_eval_cer_mean": float(np.mean(real_eval)),
        "real_eval_cer_std": float(np.std(real_eval)),
        "control_minus_real_eval_cer_mean": float(np.mean(control_eval) - np.mean(real_eval)),
        "std_convention": "population (numpy ddof=0)",
    }

    structure = [{
        "seed": result["seed"],
        "real_evaluation": output_structure(real_by_seed[result["seed"]]["evaluation_predictions"]),
        "control_evaluation": output_structure(result["evaluation_predictions"]),
    } for result in seed_results]

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    seeds = [str(seed) for seed in baseline.SEEDS]
    x = np.arange(len(seeds))
    width = 0.38
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, key, title in ((axes[0], "eval", "Evaluation CER (trials 8-9)"), (axes[1], "train", "Training CER (trials 2-7)")):
        real_values = [row[f"real_{key}_cer"] for row in comparison]
        control_values = [row[f"control_{key}_cer"] for row in comparison]
        axis.bar(x - width / 2, real_values, width, label="real labels (5U)", color="#3a6ea5")
        axis.bar(x + width / 2, control_values, width, label="target permutation (5V)", color="#c8553d")
        for offset, values in ((-width / 2, real_values), (width / 2, control_values)):
            for position, value in zip(x + offset, values):
                axis.text(position, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
        axis.set_xticks(x, seeds)
        axis.set_xlabel("model seed")
        axis.set_title(title)
    axes[0].set_ylabel("CER")
    axes[0].legend(loc="upper left")
    figure.suptitle(f"Real-label vs target-permutation control (shuffle seed {SHUFFLE_SEED})")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(len(seed_results), 1, figsize=(12, 7), squeeze=False)
    for axis, result in zip(axes[:, 0], seed_results):
        axis.axis("off")
        axis.set_title(f"control model seed {result['seed']}", loc="left")
        axis.text(0.01, 0.8, "\n".join(
            f"trial {row['trial_number']}  CER={row['cer']:.3f}\n  target : {row['target']!r}\n  decoded: {row['decoded']!r}"
            for row in result["evaluation_predictions"]
        ), fontsize=8.5, va="top", family="monospace")
    figure.tight_layout()
    figure.savefig(OUTPUTS_FIGURE_PATH, dpi=160)
    plt.close(figure)

    artifact = {
        "experiment": "5V_no_signal_target_permutation",
        "official_revision": real["official_revision"],
        "representation": "official_v1_event_sequence",
        "shuffle_seed": SHUFFLE_SEED,
        "permutation_method": "numpy default_rng(2026).permutation drawn repeatedly from one generator until no fixed point",
        "permutation_draws": draws,
        "permutation_signal_trial_to_target_trial": {str(k): v for k, v in permutation.items()},
        "training_pair_audit": train_audit,
        "evaluation_pair_audit": eval_audit,
        "audit_checks": checks,
        "model_seeds": list(baseline.SEEDS),
        "model": "Conv1D + BiGRU + Linear + CTC (existing ConvCTC, via committed 5U train_seed)",
        "optimizer": {"name": "Adam", "learning_rate": baseline.LEARNING_RATE, "gradient_clip_max_norm": baseline.CLIP_NORM, "epochs": baseline.EPOCHS},
        "decoder": "existing greedy CTC decoder",
        "seed_results": seed_results,
        "cross_seed": cross_seed,
        "real_vs_control": comparison,
        "output_structure": structure,
        "model_input_audit": {
            "forward_inputs": [{"shape": list(shape), "dtype": dtype, "calls": count} for (shape, dtype), count in sorted(forward_inputs.items())],
            "model_receives_only": "single float32 MEG tensor (1, 25U, 306)",
            "not_fed_to_model": ["trial ID", "sentence UID", "target length", "text metadata", "subject ID", "explicit label feature"],
            "caveat": "Input length T = 25U equals 25 x the number of keystrokes, so sequence length implicitly carries the original sentence length for every trial in both real and control conditions. Under the permutation this length matches the original, not the control, training target.",
        },
        "figures": [str(FIGURE_PATH), str(OUTPUTS_FIGURE_PATH)],
        "verification": {
            "official_v1_preprocessing_used": True,
            "historical_preprocessing_used": False,
            "signals_unchanged_after_training": signals_unchanged_after_training,
            "single_permutation_shared_by_all_seeds": True,
            "evaluation_targets_permuted": False,
            "additional_controls": False,
            "additional_seeds": False,
            "no_llm": True,
            "no_evidence_selector": True,
            "no_data_download": True,
            "checkpoints_saved": False,
        },
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(OUT_PATH), "permutation": artifact["permutation_signal_trial_to_target_trial"], "cross_seed": cross_seed}, indent=2))


if __name__ == "__main__":
    main()
