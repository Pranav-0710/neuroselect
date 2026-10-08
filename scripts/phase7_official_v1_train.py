"""Phase 7 Tasks 6, 8, 9: exact official Brain2Qwerty v1 training (GPU only).

One invocation = one run: one split, one seed, real or matched no-valid-signal
control. Training is refused without CUDA; ``--dry-run`` builds and checks
everything (dataset hashes, split, loaders, control relabelling, exact model,
trainer) and scores a few test sentences with untrained weights to exercise
the evaluation path, without any optimisation step.

    # official split, seed 33, real
    python scripts/phase7_official_v1_train.py --split official_v1_clean --seed 33 \
        --output-dir results/runs/phase7/official_v1_clean/real_seed33
    # matched control (training targets deranged, seed 2026; same everything else)
    python scripts/phase7_official_v1_train.py --split official_v1_clean --seed 33 --control \
        --output-dir results/runs/phase7/official_v1_clean/control_seed33

``--split-manifest`` accepts explicit manifest files instead of ``--split``.
Outputs in the run directory: the official Lightning artefacts (best.ckpt,
last.ckpt, CSV logs, callbacks/test_all_sentences.json) plus
``run_summary.json`` and ``test_predictions.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("phase7_harness", ROOT / "scripts/phase7_official_v1_harness.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
c, h, torch = H.c, H.h, H.torch

DRY_RUN_TEST_SENTENCES = 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-manifest", type=Path, default=H.DATASET_MANIFEST)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--split", choices=sorted(H.SPLIT_CHOICES))
    group.add_argument("--split-manifest", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control", action="store_true",
                        help="matched no-valid-signal control: training targets deranged (seed 2026)")
    parser.add_argument("--devices", type=int, default=None,
                        help="override the official Experiment.devices (8, capped by GPUs present); logged as a deviation")
    parser.add_argument("--dry-run", action="store_true", help="build and check everything; no optimisation step")
    parser.add_argument("--skip-raw-hash", action="store_true", help="skip re-hashing the 8 raw files (logged)")
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    started = time.time()
    if not args.dry_run and not torch.cuda.is_available():
        sys.exit("Phase 7: official v1 training runs on CUDA GPUs only; no CUDA device is available. "
                 "Use --dry-run to check the harness.")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "task": "Phase 7 official Brain2Qwerty v1 run" + (" (dry run)" if args.dry_run else ""),
        "arguments": {k: (v.as_posix() if isinstance(v, Path) else
                          [p.as_posix() for p in v] if isinstance(v, list) else v) for k, v in vars(args).items()},
        "condition": "no_valid_signal_control" if args.control else "real",
        "provenance": H.provenance(),
    }

    summary["dataset"] = H.verify_dataset(args.dataset_manifest, hash_raw=not args.skip_raw_hash)
    if not summary["dataset"]["all_verified"]:
        raise RuntimeError("Dataset hashes do not match the dataset manifest")

    split = c.load_split(args.split_manifest or H.SPLIT_CHOICES[args.split])
    summary["split"] = {k: split[k] for k in ("split_name", "manifest_files", "partition_uid_sha256")}
    summary["split"]["integrity"] = c.check_split(split["partitions"])
    if not summary["split"]["integrity"]["disjoint"]:
        raise RuntimeError(f"Split partitions overlap: {summary['split']['integrity']}")

    control = None
    if args.control:
        control = c.stimulus_derangement(split["partitions"]["train"])
        if control["fixed_points"] or control["same_cluster_pairs"] or control["identical_target_text_pairs"]:
            raise RuntimeError("Control derangement left a valid or near-valid target")
        summary["control"] = {k: v for k, v in control.items() if k != "donor_of_uid"}
        summary["control"]["donor_of_uid"] = control["donor_of_uid"]
        summary["control"]["evaluation_targets"] = "unchanged (validation and test correctly paired)"

    cfg, xp, deviations = H.build_experiment(split, args.seed, output_dir, control=control, devices=args.devices,
                                             local_dry_run=args.dry_run and sys.platform == "win32")
    summary["deviations_from_official"] = deviations
    summary["training_configuration"] = {
        "official_config": json.loads(json.dumps(cfg, default=H.json_default)),
        "seed": xp.seed, "n_epochs": xp.n_epochs, "patience": xp.patience, "grad_max_norm": xp.grad_max_norm,
        "devices_requested": xp.devices, "accelerator_and_devices": list(xp._accelerator()),
        "train_batch_size_per_device": xp.data.batch_size,
        "effective_global_train_batch": xp.data.batch_size * xp._accelerator()[1],
        "note_on_devices": ("The official Experiment defaults to 8 GPUs with DDP (global batch 8 x 64 keystrokes). "
                            "With fewer GPUs the per-device batch stays 64 and the global batch and step count change "
                            "accordingly; the realised value is recorded here."),
    }

    loaders = H.build_loaders(xp)
    summary["loaders"] = H.loader_audit(loaders, split)
    if not all(v["matches_manifest"] for v in summary["loaders"].values()):
        raise RuntimeError(f"Loaders do not match the split manifest: {summary['loaders']}")
    if control is not None:
        summary["control"]["relabelling"] = dict(H.Phase7TrainTargetDerangement.last_stats)

    module = H.build_module(xp, loaders)
    summary["parameters"] = h.parameter_counts(module)
    summary["seconds_to_model"] = time.time() - started

    if args.dry_run:
        trainer = xp._trainer_setup()  # constructed, never fitted
        summary["trainer"] = {"max_epochs": trainer.max_epochs, "gradient_clip_val": trainer.gradient_clip_val,
                              "callbacks": [type(cb).__name__ for cb in trainer.callbacks],
                              "early_stopping": [{"monitor": cb.monitor, "patience": cb.patience, "mode": cb.mode}
                                                 for cb in trainer.callbacks if type(cb).__name__ == "EarlyStopping"]}
        probe = H.evaluate(module, loaders["test"].dataset, xp.data.test_batch_size,
                           max_sentences=DRY_RUN_TEST_SENTENCES)
        summary["dry_run_evaluation_path_check"] = {
            "what": "untrained weights on the first test sentences; exercises scoring only, not a result",
            "sentences": probe["sentences"], "keystrokes": probe["keystrokes"],
            "metric_keys": sorted(k for k in probe if k not in ("rows", "keystroke_order", "predicted_in_order")),
            "finite": bool(probe["keystroke_accuracy"] == probe["keystroke_accuracy"]),
        }
        summary["status"] = "DRY RUN COMPLETE (no training performed)"
    else:
        trainer, official_test = H.run_official(xp, loaders, module)
        if trainer.global_rank != 0:
            return summary
        summary["training"] = {
            "epochs_completed": trainer.current_epoch,
            "global_steps": trainer.global_step,
            "early_stopped": any(getattr(cb, "stopped_epoch", 0) > 0 for cb in trainer.callbacks),
            "best_val_CER": float(trainer.checkpoint_callback.best_model_score)
            if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_score is not None else None,
            "best_checkpoint": trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else None,
            "official_trainer_test_output": official_test,
        }
        test_dataset = loaders["test"].dataset
        final = H.evaluate(module, test_dataset, xp.data.test_batch_size)
        evaluations = {"final_weights": final}
        best_path = summary["training"]["best_checkpoint"]
        if best_path:
            state = torch.load(best_path, map_location=next(module.parameters()).device, weights_only=False)
            module.load_state_dict(state["state_dict"])
            evaluations["best_val_checkpoint"] = H.evaluate(module, test_dataset, xp.data.test_batch_size)
        summary["test_metrics"] = {
            name: {k: v for k, v in result.items() if k not in ("rows", "keystroke_order", "predicted_in_order")}
            for name, result in evaluations.items()}
        summary["test_metrics_primary"] = "final_weights (the official Experiment.run tests the final weights)"
        (output_dir / "test_predictions.json").write_text(json.dumps({
            name: {"rows": result["rows"], "keystroke_order": result["keystroke_order"],
                   "predicted_in_order": result["predicted_in_order"]}
            for name, result in evaluations.items()}, ensure_ascii=False), encoding="utf-8")
        summary["status"] = "TRAINED"
    summary["runtime_seconds"] = time.time() - started
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=H.json_default), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status", "condition", "parameters", "loaders", "runtime_seconds")},
                     indent=2, default=H.json_default))
    return summary


if __name__ == "__main__":
    main()
