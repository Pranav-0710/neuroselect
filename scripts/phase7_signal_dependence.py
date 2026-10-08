"""Phase 7 Task 11: post-training signal-dependence evaluation (real official v1 only).

Loads a trained REAL run (control runs are refused), restores the final weights
(``last.ckpt``: the weights the official Experiment.run tests) and scores the
held-out test partition four times, without retraining:

  A original        unchanged MEG
  B zero_meg        MEG replaced by zeros
  C temporal_perm   one fixed derangement of the 25 time samples (seed 2026), every event
  D channel_perm    one fixed derangement of the 306 channels (seed 2026); channel
                    positions unchanged, so signals reach the wrong sensor slots

Reports character F1, keystroke accuracy, CER and agreement with A. Sensitivity
to these ablations shows that predictions depend on the input; it does not by
itself show meaningful neural decoding.

    python scripts/phase7_signal_dependence.py --run-dir results/runs/phase7/official_v1_clean/real_seed33
    python scripts/phase7_signal_dependence.py --dry-run --split official_v1_clean   # code path, untrained weights
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("phase7_harness", ROOT / "scripts/phase7_official_v1_harness.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
c, h, torch = H.c, H.h, H.torch

CONDITIONS = ("original", "zero_meg", "temporal_permutation", "channel_permutation")
DRY_RUN_SENTENCES = 2


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--split", choices=sorted(H.SPLIT_CHOICES), default="official_v1_clean")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.dry_run:
        split, seed, output_dir = c.load_split(H.SPLIT_CHOICES[args.split]), 33, None
    else:
        if args.run_dir is None:
            sys.exit("--run-dir is required unless --dry-run")
        run = json.loads((args.run_dir / "run_summary.json").read_text(encoding="utf-8"))
        if run["condition"] != "real":
            sys.exit("Signal dependence is evaluated on the real official v1 model only")
        if run["status"] != "TRAINED":
            sys.exit("Run is not trained")
        split = c.load_split([ROOT / p if not Path(p).is_absolute() else Path(p)
                              for p in run["split"]["manifest_files"]])
        if split["partition_uid_sha256"] != run["split"]["partition_uid_sha256"]:
            sys.exit("Split manifests changed since the run")
        seed, output_dir = run["arguments"]["seed"], args.run_dir

    scratch = output_dir or ROOT / "results/runs/phase7/signal_dependence_dry_run"
    _, xp, deviations = H.build_experiment(split, seed, scratch, devices=1,
                                           local_dry_run=args.dry_run and sys.platform == "win32")
    loaders = H.build_loaders(xp)
    module = H.build_module(xp, loaders)
    if not args.dry_run:
        state = torch.load(args.run_dir / "last.ckpt", map_location="cpu", weights_only=False)
        module.load_state_dict(state["state_dict"])
        if torch.cuda.is_available():
            module.to("cuda:0")
    dataset = loaders["test"].dataset
    n_channels, n_times = (int(s) for s in dataset[0].data["neuro"].shape[1:])
    limit = DRY_RUN_SENTENCES if args.dry_run else None

    results = {}
    for condition in CONDITIONS:
        ablation = H.neuro_ablation(condition, n_channels, n_times)
        results[condition] = H.evaluate(module, dataset, xp.data.test_batch_size, ablation=ablation,
                                        max_sentences=limit)
    reference = results["original"]["predicted_in_order"]
    summary = {
        "task": "Phase 7 Task 11: signal dependence of the real official v1 model",
        "status": "DRY RUN (untrained weights; code path only, not a result)" if args.dry_run else "EVALUATED",
        "weights": "untrained" if args.dry_run else "last.ckpt (final weights, as tested by the official Experiment.run)",
        "split": split["split_name"],
        "seed": seed,
        "deviations_from_official": deviations,
        "ablation_seed": H.ABLATION_SEED,
        "conditions": {
            condition: {
                "character_f1": r["character_f1"],
                "keystroke_accuracy": r["keystroke_accuracy"],
                "mean_sentence_cer": r["mean_sentence_cer"],
                "mean_sentence_wer": r["mean_sentence_wer"],
                "prediction_agreement_with_original": c.agreement(r["predicted_in_order"], reference),
                "keystrokes": r["keystrokes"],
            } for condition, r in results.items()},
        "interpretation_rule": ("Sensitivity to ablation shows input dependence only; it is not evidence of "
                                "meaningful neural decoding without the real-vs-control comparison."),
    }
    out = args.output or (scratch / "signal_dependence.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=H.json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=H.json_default))


if __name__ == "__main__":
    main()
