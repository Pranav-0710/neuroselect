"""Phase 7 Task 9: verify that the control differs from the real run only in training targets.

For each clean split, builds the real and the control experiments for the same
model seed and compares their loaders: MEG, subject ids, channel positions and
batch composition must be identical everywhere; targets must be identical on
validation and test and deranged on training. Also checks that the derangement
is the same for every model seed. No model is built and nothing is trained.

    python scripts/phase7_control_harness_check.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("phase7_harness", ROOT / "scripts/phase7_official_v1_harness.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
c, h, torch = H.c, H.h, H.torch

OUT = ROOT / "results/phase7_control_harness.json"
CHECK_BATCHES = 3
SCRATCH = ROOT / "results/runs/phase7/control_check"


def first_batches(loader, n):
    batches = []
    for index, batch in enumerate(loader):
        if index >= n:
            break
        batches.append(batch)
    return batches


def main() -> None:
    report = {"task": "Phase 7 Task 9: matched no-valid-signal control harness check",
              "control_definition": {
                  "what": ("training-target derangement at the stimulus level (unique sentence group), seed 2026, "
                           "constrained to a different group and a different official TF-IDF cluster, within "
                           "length-ordered windows of 8 groups; both occurrences of a group take the donor group's "
                           "same-session occurrence"),
                  "per_keystroke_labels": ("official formulation needs one label per keystroke: a longer donor is "
                                           "truncated, a shorter donor is repeated after a space token"),
                  "unchanged": ["MEG tensors", "subject ids", "channel positions", "keystroke timing and segments",
                                "batch structure", "architecture", "optimizer", "scheduler", "epochs", "early stopping",
                                "model seeds", "validation and test targets"],
                  "same_permutation_across_model_seeds": True,
              },
              "splits": {}}
    local = sys.platform == "win32"
    for name, files in H.SPLIT_CHOICES.items():
        split = c.load_split(files)
        maps = [c.stimulus_derangement(split["partitions"]["train"])["donor_of_uid"] for _ in H.SEEDS]
        control = c.stimulus_derangement(split["partitions"]["train"])
        _, real_xp, _ = H.build_experiment(split, 33, SCRATCH / name / "real", local_dry_run=local)
        _, control_xp, _ = H.build_experiment(split, 33, SCRATCH / name / "control", control=control,
                                              local_dry_run=local)
        real = H.build_loaders(real_xp)
        ctrl = H.build_loaders(control_xp)
        stats = dict(H.Phase7TrainTargetDerangement.last_stats)
        parts = {}
        for part in c.PARTITIONS:
            same_segments = [(s.trigger.timeline, s.trigger.start) for s in real[part].dataset.segments] == \
                            [(s.trigger.timeline, s.trigger.start) for s in ctrl[part].dataset.segments]
            n = CHECK_BATCHES if part == "train" else 1
            identical = {"neuro": True, "subject_id": True, "channel_positions": True}
            targets_equal = []
            for a, b in zip(first_batches(real[part], n), first_batches(ctrl[part], n)):
                for key in identical:
                    identical[key] &= bool(torch.equal(a.data[key], b.data[key]))
                targets_equal.append(float((a.data["feature"] == b.data["feature"]).float().mean()))
            parts[part] = {"segments_identical": same_segments, "batches_compared": n,
                           "inputs_identical": identical,
                           "fraction_of_targets_equal_in_compared_batches": targets_equal}
        report["splits"][name] = {
            "derangement": {k: v for k, v in control.items() if k not in ("donor_of_uid", "group_map")},
            "group_map": control["group_map"],
            "identical_for_seeds": {str(s): m == control["donor_of_uid"] for s, m in zip(H.SEEDS, maps)},
            "relabelling": stats,
            "loaders": parts,
            "pass": (all(p["segments_identical"] and all(p["inputs_identical"].values()) for p in parts.values())
                     and all(v == 1.0 for p in ("val", "test")
                             for v in parts[p]["fraction_of_targets_equal_in_compared_batches"])
                     and max(parts["train"]["fraction_of_targets_equal_in_compared_batches"]) < 0.5
                     and control["fixed_points"] == 0 and control["same_cluster_pairs"] == 0
                     and control["identical_target_text_pairs"] == 0),
        }
    report["all_pass"] = all(s["pass"] for s in report["splits"].values())
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=H.json_default), encoding="utf-8")
    print(json.dumps({n: {"pass": s["pass"], "relabelling": s["relabelling"],
                          "loaders": s["loaders"]} for n, s in report["splits"].items()}, indent=1,
                     default=H.json_default)[:5000])


if __name__ == "__main__":
    main()
