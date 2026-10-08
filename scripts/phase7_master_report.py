"""Phase 7 Tasks 6, 13, 15-17: harness record, figures, tests, integrity and classification.

Reads the Phase 7 artifacts produced by the pinned-environment scripts
(split audit, manifests, dry runs, control check, preflight) and the text
anchor, then writes ``results/phase7_training_harness.json`` and
``results/phase7_master_report.json`` and the three Phase 7 figures. No model
is trained or evaluated here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures/debug"
DRY = RESULTS / "runs/phase7/dry_run"
DATA_ROOT = ROOT / "data/raw/spanishbcbl_s22"
OVERLAY = Path.home() / "Envs/phase6-official-overlay"
PINNED_PYTHON = Path.home() / "Envs/neuroselect-brain2qwerty-v1/Scripts/python.exe"
OFFICIAL_REVISION = "5f9889621d0df391c5aab37c996683d308e6e926"
EXPECTED_PARAMETERS = 623_548_457
SPLITS = ("official_v1_clean", "D_clean", "E_clean")
SEEDS = (33, 123, 777)
PROTECTED = {
    "historical_preprocessing": "scripts/prepare_real_subset.py",
    "official_v1_preprocessing": "src/neuroselect/official_v1_preprocessing.py",
    "official_source": "vendor/brain2qwerty",
    "existing_tests": "tests/test_baseline.py tests/test_official_v1_preprocessing.py",
}
HARNESS_FILES = [
    "scripts/phase7_official_split.py", "scripts/phase7_common.py", "scripts/phase7_dataset_manifest.py",
    "scripts/phase7_official_v1_harness.py", "scripts/phase7_official_v1_train.py",
    "scripts/phase7_gpu_preflight.py", "scripts/phase7_signal_dependence.py",
    "scripts/phase7_control_harness_check.py", "scripts/phase7_text_anchor.py", "scripts/phase6_official_v1_harness.py",
]
OUT_HARNESS = RESULTS / "phase7_training_harness.json"
OUT_MASTER = RESULTS / "phase7_master_report.json"
FIG = {
    "clusters": FIGURES / "phase7_split_clusters.png",
    "composition": FIGURES / "phase7_split_composition.png",
    "preflight": FIGURES / "phase7_gpu_preflight.png",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def run_tests() -> dict:
    basetemp = tempfile.mkdtemp(prefix="neuroselect-pytest-")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", f"--basetemp={basetemp}"],
        cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"},
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    summary = next((line for line in reversed(lines) if re.search(r"\d+ (passed|failed|error)", line)), "")
    counts = {key: 0 for key in ("passed", "failed", "skipped", "errors")}
    for number, word in re.findall(r"(\d+) (passed|failed|skipped|errors?)", summary):
        counts["errors" if word.startswith("error") else word] = int(number)
    return {"command": "python -m pytest tests -q", "exit_code": result.returncode,
            "summary_line": summary.strip("= ").strip(), "collected": sum(counts.values()), **counts}


def raw_file_hashes() -> dict:
    acquisition = read(RESULTS / "s22_acquisition.json")
    expected = {row["remote_path"]: row["actual_sha256"] for row in acquisition["downloads"]}
    expected.update(acquisition["protected_file_hashes"])
    files = []
    for path in sorted([*DATA_ROOT.glob("MEG/FIF/**/*.fif"), *DATA_ROOT.glob("MEG/logs/*.mat")]):
        relative = path.relative_to(DATA_ROOT).as_posix()
        digest = sha256_file(path)
        files.append({"file": relative, "sha256": digest, "matches_acquisition_record": expected.get(relative) == digest})
    return {"files": files, "fif_files": sum(f["file"].endswith(".fif") for f in files),
            "mat_files": sum(f["file"].endswith(".mat") for f in files),
            "all_match": len(files) == 8 and all(f["matches_acquisition_record"] for f in files)}


def pinned_env_unchanged() -> dict:
    before = (OVERLAY / "pinned_env_freeze_before.txt").read_text(encoding="utf-8")
    now = subprocess.run([str(PINNED_PYTHON), "-m", "pip", "freeze", "--all"], capture_output=True, text=True).stdout
    return {"pip_freeze_identical_to_pre_phase6_snapshot": before == now, "packages": len(before.splitlines())}


# ---- figures -------------------------------------------------------------------

def figure_clusters(audit: dict, clusters: dict) -> None:
    rows = clusters["clusters"]
    pairs = sorted((c["max_within_cluster_cosine"] for c in rows if c["size_sentence_texts"] > 1), reverse=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    ax.bar(range(len(pairs)), pairs, color="#4C72B0")
    ax.axhline(0.5, color="#C44E52", linestyle="--", label="official threshold (cosine > 0.5)")
    ax.set_xlabel("multi-sentence cluster (sorted)")
    ax.set_ylabel("TF-IDF cosine of the pair")
    ax.set_title(f"{len(pairs)} official clusters merge a list1 and a list2 text")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[1]
    parts = ("train", "val", "test")
    singles = [sum(1 for c in rows if c["official_split"] == p and c["size_sentence_texts"] == 1) for p in parts]
    doubles = [sum(1 for c in rows if c["official_split"] == p and c["size_sentence_texts"] > 1) for p in parts]
    ax.bar(parts, singles, color="#55A868", label="single-text cluster")
    ax.bar(parts, doubles, bottom=singles, color="#8172B2", label="list1+list2 pair cluster")
    for i, (s, d) in enumerate(zip(singles, doubles)):
        ax.text(i, s + d + 0.8, f"{s + d}", ha="center", fontsize=9)
    ax.set_ylabel("official clusters")
    ax.set_title("Official 80/10/10: whole clusters per partition")
    ax.legend(fontsize=8)

    ax = axes[2]
    names = ("D_clean", "E_clean")
    kept = [audit["cross_session_clean"][n]["composition"]["test"]["sentence_records"] for n in names]
    dropped = [audit["cross_session_clean"][n]["test_records_excluded_for_cluster_integrity"] for n in names]
    ax.bar(names, kept, color="#4C72B0", label="test sentences kept")
    ax.bar(names, dropped, bottom=kept, color="#DD8452", hatch="//", label="excluded: cluster also on train side")
    ax.set_ylabel("session-2 sentences in the literal test block")
    ax.set_title("Literal D/E would split 32 clusters each")
    ax.legend(fontsize=8)
    fig.suptitle("Phase 7: official TF-IDF clusters on S22 (128 texts, 96 clusters)")
    fig.tight_layout()
    fig.savefig(FIG["clusters"], dpi=130)
    plt.close(fig)


def figure_composition(manifests: dict) -> None:
    summary = manifests["summary"]
    blocks = ["session1/block1/list1", "session1/block2/list2", "session2/block1/list2", "session2/block2/list1"]
    colors = dict(zip(blocks, ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]))
    labels, data = [], []
    for split in SPLITS:
        for part in ("train", "val", "test"):
            labels.append(f"{split} / {part}")
            comp = summary[split][part]["by_session_block_list"]
            data.append([comp.get(b, {}).get("keystrokes", 0) for b in blocks])
    data = np.array(data)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    left = np.zeros(len(labels))
    for j, block in enumerate(blocks):
        ax.barh(labels, data[:, j], left=left, color=colors[block], label=block)
        left += data[:, j]
    for i, total in enumerate(left):
        ax.text(total + 40, i, f"{int(total)} keystrokes", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("keystrokes")
    ax.set_xlim(0, left.max() * 1.18)
    ax.set_title("Phase 7 clean splits: keystrokes per partition by session/block/list")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG["composition"], dpi=130)
    plt.close(fig)


def figure_preflight(preflight: dict, parameters: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if preflight["status"] == "GPU preflight executed":
        m = preflight["measured"]
        names = ["weights", "train peak allocated", "train peak reserved", "val batch peak"]
        values = [m["weights_allocated_gib"], m["train_peak_allocated_gib"], m["train_peak_reserved_gib"],
                  m["val_batch_peak_allocated_gib"]]
        ax.bar(names, values, color="#4C72B0")
        ax.axhline(m["device_total_memory_gib"], color="#C44E52", linestyle="--", label=m["device"])
        ax.set_ylabel("GiB (measured)")
        ax.legend()
        ax.set_title("Phase 7 GPU preflight (measured)")
    else:
        gib = parameters * 4 / 1024 ** 3
        names = ["weights", "gradients", "AdamW exp_avg", "AdamW exp_avg_sq"]
        bottom = 0.0
        for name, color in zip(names, ["#4C72B0", "#55A868", "#8172B2", "#937860"]):
            ax.bar(["fp32 parameter state\n(arithmetic, not measured)"], [gib], bottom=bottom, color=color,
                   label=f"{name}: {gib:.2f} GiB")
            bottom += gib
        ax.set_ylabel("GiB")
        ax.set_ylim(0, bottom * 2.0)
        ax.legend(loc="upper left", fontsize=8)
        ax.text(0.98, 0.95, "GPU PREFLIGHT NOT EXECUTED\nno CUDA device on this machine",
                transform=ax.transAxes, ha="right", va="top", fontsize=12, color="#C44E52", fontweight="bold")
        ax.text(0.98, 0.70, f"{parameters:,} parameters x 4 bytes x 4 copies = {bottom:.2f} GiB\n"
                            "lower bound: excludes activations, CUDA context,\nDDP buckets and allocator overhead",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.set_title("Phase 7 GPU preflight: not executed (no measurement shown)")
    fig.tight_layout()
    fig.savefig(FIG["preflight"], dpi=130)
    plt.close(fig)


# ---- main ----------------------------------------------------------------------

def main() -> None:
    audit = read(RESULTS / "phase7_official_split_audit.json")
    manifests = read(RESULTS / "phase7_clean_split_manifests.json")
    clusters = read(RESULTS / "phase7_official_clusters.json")
    preflight = read(RESULTS / "phase7_gpu_preflight.json")
    control = read(RESULTS / "phase7_control_harness.json")
    anchor = read(RESULTS / "phase7_text_anchor.json")
    signal_dry = read(DRY / "signal_dependence_dry_run.json")
    dry = {f"{s}_{cond}": read(DRY / f"{s}_{cond}_seed33/run_summary.json")
           for s in SPLITS for cond in ("real", "control")}

    dry_checks = {
        name: {
            "status": run["status"],
            "dataset_hashes_verified": run["dataset"]["all_verified"],
            "raw_files_hashed": sum(1 for f in run["dataset"]["files"] if f["kind"] == "raw" and f["verified"]),
            "split_disjoint": run["split"]["integrity"]["disjoint"],
            "loaders_match_manifest": all(v["matches_manifest"] for v in run["loaders"].values()),
            "loaders": {p: {k: v[k] for k in ("keystroke_segments", "sentences", "batch_size", "batches")}
                        for p, v in run["loaders"].items()},
            "parameters_total": run["parameters"]["total"],
            "parameters_exact": run["parameters"]["total"] == EXPECTED_PARAMETERS,
            "experiment_run_source_matches_audit": run["provenance"]["experiment_run_source_matches_audit"],
            "official_source_clean": run["provenance"]["official_source_clean"],
            "trainer": run["trainer"],
            "deviations_from_official": run["deviations_from_official"],
            "control_relabelling": run.get("control", {}).get("relabelling"),
            "evaluation_path_exercised": run["dry_run_evaluation_path_check"]["finite"],
        } for name, run in dry.items()}
    dry_ok = all(v["dataset_hashes_verified"] and v["raw_files_hashed"] == 8 and v["split_disjoint"]
                 and v["loaders_match_manifest"] and v["parameters_exact"] and v["experiment_run_source_matches_audit"]
                 and v["official_source_clean"] and v["evaluation_path_exercised"]
                 and v["trainer"]["max_epochs"] == 300 and v["trainer"]["gradient_clip_val"] is None
                 and v["trainer"]["early_stopping"] == [{"monitor": "val_CER", "patience": 30, "mode": "min"}]
                 for v in dry_checks.values())

    run_matrix = []
    for split in SPLITS:
        for seed in SEEDS:
            for condition in ("real", "control"):
                out = f"results/runs/phase7/{split}/{condition}_seed{seed}"
                run_matrix.append({
                    "split": split, "seed": seed, "condition": condition, "output_dir": out,
                    "command": (f"python scripts/phase7_official_v1_train.py --split {split} --seed {seed}"
                                f"{' --control' if condition == 'control' else ''} --output-dir {out}"),
                    "primary": split == "official_v1_clean",
                })
    signal_commands = [f"python scripts/phase7_signal_dependence.py --run-dir {r['output_dir']}"
                       for r in run_matrix if r["condition"] == "real"]
    harness = {
        "task": "Phase 7 Tasks 6, 8, 10, 11: exact official Brain2Qwerty v1 GPU training harness",
        "official_revision": OFFICIAL_REVISION,
        "training_executed": False,
        "why_not_executed": "No CUDA GPU on this machine; Phase 7 forbids local training (Task 14).",
        "files": {f: sha256_file(ROOT / f) for f in HARNESS_FILES},
        "what_is_official": [
            "experiment_config() at 5f98896 (model, loss, optimizer, scheduler, epochs, patience, batch sizes, extractors)",
            "Data.build: MegExtractor, LabelEncoder targets, subject ids, ChannelPositions2D, SegmentDataset, "
            "SentenceGroupedDistributedSampler",
            "Experiment._build_modules, BrainModule, materialize_lazy_params, Experiment._trainer_setup "
            "(EarlyStopping val_CER patience 30, ModelCheckpoint best/last, CSVLogger, DDP when devices > 1)",
            "trainer.fit then trainer.test on the final weights, as Experiment.run",
        ],
        "what_differs": [
            "the official splitter transform is replaced by Phase7ManifestSplit, which assigns train/val/test from "
            "the formal manifests (for official_v1_clean these equal the official splitter's own assignment)",
            "control only: Phase7TrainTargetDerangement relabels training keystroke targets",
            "Windows only: the study event cache is disabled (illegal filenames); not needed on Linux",
            "after training, rank 0 additionally scores the test partition in one process (official test batching) "
            "for the Phase 7 metrics; the official trainer.test output is kept alongside",
        ],
        "inputs": {
            "dataset_manifest": "data/manifests/s22_official_v1_dataset.json (8 raw files + metadata + stored events, sha256)",
            "split_manifest": "--split {official_v1_clean, D_clean, E_clean} or --split-manifest <files>",
            "seed": "--seed (33, 123, 777)",
            "output_dir": "--output-dir",
            "control": "--control",
        },
        "logged_per_run": [
            "git commit and dirty flag", "official revision, vendor tree hash, Experiment.run source hash",
            "all installed package versions and mismatches against the official requirements.lock",
            "exact parameter count by module", "full official training configuration and realised devices/global batch",
            "split identity: manifest file hashes and per-partition UID hashes", "dataset hashes (verified)",
            "runtime", "GPU names, memory, capability, torch CUDA build, cuDNN, nvidia-smi driver",
            "control derangement map and relabelling statistics (control runs)",
        ],
        "gpu_environment": {
            "install": "pip install -r vendor/brain2qwerty/requirements.lock  (Linux x86-64, CUDA 12.4, torch 2.6.0)",
            "pythonpath": "vendor/brain2qwerty:src",
            "data": "data/raw/spanishbcbl_s22 (verified against the dataset manifest; nothing is downloaded)",
            "memory_lower_bound": "9.29 GiB fp32 parameter + gradient + AdamW state per GPU, before activations",
        },
        "run_matrix": run_matrix,
        "runs": len(run_matrix),
        "order": ("preflight first; then official_v1_clean (primary) real and control for each seed, then D_clean "
                  "and E_clean; then signal dependence on each real run"),
        "signal_dependence": {
            "script": "scripts/phase7_signal_dependence.py",
            "model": "real runs only (control runs refused); final weights (last.ckpt); no retraining",
            "conditions": {"A": "original", "B": "zero_meg", "C": "temporal_permutation (fixed derangement of 25 "
                           "samples, seed 2026)", "D": "channel_permutation (fixed derangement of 306 channels, seed "
                           "2026; positions unchanged)"},
            "measures": ["character F1", "keystroke accuracy", "CER", "WER", "prediction agreement with A"],
            "commands": signal_commands,
            "dry_run": {"status": signal_dry["status"], "conditions": signal_dry["conditions"]},
            "interpretation_rule": signal_dry["interpretation_rule"],
        },
        "metrics": {
            "primary": ["per-keystroke accuracy", "character F1", "mean sentence CER", "mean sentence WER"],
            "secondary": ["substitutions", "deletions", "insertions", "decoded length ratio", "confusion matrix",
                          "per-character recall", "predicted and target sequence lengths (characters and keystrokes)"],
            "weights_scored": "final weights (official) as primary; best-val checkpoint also scored and labelled",
            "comparability": ("v1 emits exactly one label per keystroke (oracle segmentation), so its CER is not "
                              "comparable with CTC CER from earlier phases; compare v1 with its own matched control"),
        },
        "known_protocol_facts": [
            "Experiment.devices defaults to 8: on 8 GPUs the global batch is 8 x 64 keystrokes; with fewer GPUs the "
            "per-device batch stays 64 and the realised value is logged",
            "batches are consecutive keystrokes in sentence order, so a sentence can straddle two batches and reach "
            "the transformer in pieces (official behaviour)",
            "the official val_CER/test_CER metric averages per batch, not per sentence",
            "under DDP, SentenceGroupedDistributedSampler pads ranks with repeated samples and the official prediction "
            "callback saves rank 0 only; the Phase 7 single-process scoring avoids both",
        ],
        "dry_runs": dry_checks,
        "dry_runs_all_pass": dry_ok,
    }
    OUT_HARNESS.write_text(json.dumps(harness, indent=2, default=str), encoding="utf-8")

    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_clusters(audit, clusters)
    figure_composition(manifests)
    figure_preflight(preflight, EXPECTED_PARAMETERS)

    tests = run_tests()
    raw = raw_file_hashes()
    modified = [line for line in git("status", "--porcelain").splitlines() if not line.startswith("??")]
    integrity = {
        "raw_fif_and_mat_hashes_unchanged": raw["all_match"],
        "raw_files": raw,
        **{f"{name}_unchanged": not git("status", "--porcelain", "--", *path.split()) for name, path in PROTECTED.items()},
        "tracked_files_modified": modified,
        "existing_results_unchanged": not any(" results/" in line or line[3:].startswith("results/") for line in modified),
        "pinned_environment": pinned_env_unchanged(),
        "no_dataset_download": True,
        "evidence_selector_implemented": False,
        "llm_used": False,
        "local_training_performed": False,
        "note": "computed before the EXPERIMENT_LOG.md entry for this phase was appended",
    }

    split_ok = all(v is True for k, v in audit["official_splitter"]["exact_reproduction"].items()
                   if isinstance(v, bool)) and audit["leakage_checks"]["official_split_clean"]
    preflight_executed = preflight["status"] == "GPU preflight executed"
    harness_ready = dry_ok and control["all_pass"] and (not preflight_executed
                                                        or preflight["measured"]["fits_with_headroom"])
    if not split_ok:
        classification = ("C", "Official split could not be reproduced exactly")
    elif not harness_ready:
        classification = ("B", "Split reproduced but GPU harness blocked")
    else:
        classification = ("A", "Clean official split successfully reproduced and GPU harness ready")
    stats = audit["cluster_statistics"]
    master = {
        "task": "Phase 7: official paraphrase-disjoint splits and the exact official v1 GPU harness",
        "classification": {"code": classification[0], "label": classification[1],
                           "qualifier": ("'ready' = built, dry-run verified end to end on CPU without optimisation, "
                                         "control verified; the GPU preflight was not executed (no CUDA device), so "
                                         "CUDA memory and runtime are unmeasured" if not preflight_executed else
                                         "GPU preflight executed")},
        "no_decoding_conclusion": "Phase 7 prepares leakage-safe evaluation; it makes no decoding claim.",
        "splitter": audit["official_splitter"],
        "scope_note": audit["scope_note"],
        "cluster_statistics": stats,
        "official_split": {p: {k: audit["official_split"]["composition"][p][k] for k in
                               ("unique_sentence_groups", "sentence_records", "keystrokes", "tfidf_clusters",
                                "by_session_block_list")} for p in ("train", "val", "test")},
        "official_split_fractions": audit["official_split"]["achieved_fraction"],
        "cross_session_clean": {n: {"literal_feasible": r["literal_definition"]["feasible_without_breaking_clusters"],
                                    "literal_shared_clusters": r["clusters_shared_by_literal_train_and_test"],
                                    "test_records_excluded": r["test_records_excluded_for_cluster_integrity"],
                                    "partitions": {p: {k: r["composition"][p][k] for k in
                                                       ("sentence_records", "unique_sentence_groups", "keystrokes",
                                                        "tfidf_clusters")} for p in ("train", "val", "test")},
                                    "classification": r["classification"]}
                                for n, r in audit["cross_session_clean"].items()},
        "leakage_checks": audit["leakage_checks"],
        "gpu_preflight": {"status": preflight["status"], "reason": preflight.get("reason")},
        "control_harness_all_pass": control["all_pass"],
        "dry_runs_all_pass": dry_ok,
        "text_anchor": {n: {k: r[k] for k in ("character_f1", "mean_sentence_cer", "mean_sentence_wer", "length_ratio",
                                              "test_sentences", "test_keystrokes")}
                        for n, r in anchor["splits"].items()},
        "artifacts": {
            "results": [p.relative_to(ROOT).as_posix() for p in (
                RESULTS / "phase7_official_split_audit.json", RESULTS / "phase7_clean_split_manifests.json",
                RESULTS / "phase7_gpu_preflight.json", OUT_HARNESS, RESULTS / "phase7_control_harness.json",
                RESULTS / "phase7_text_anchor.json", OUT_MASTER, RESULTS / "phase7_official_clusters.json")],
            "manifests": [f"data/manifests/{n}" for n in manifests["manifests"]] + [
                "data/manifests/s22_official_v1_dataset.json"],
            "figures": [p.relative_to(ROOT).as_posix() for p in FIG.values()],
            "dry_runs": sorted(p.relative_to(ROOT).as_posix() for p in DRY.rglob("*.json")),
            "not_created": {"training-result figures": "no training was executed (Task 13)"},
        },
        "tests": tests,
        "integrity": integrity,
        "reproducibility": {
            "git_commit_at_report": git("rev-parse", "HEAD"),
            "pinned_environment": str(PINNED_PYTHON),
            "overlay": str(OVERLAY / "site"),
            "commands": [
                'PYTHONPATH="<overlay>/site;vendor/brain2qwerty;src" <pinned>/python scripts/phase7_official_split.py',
                "PYTHONPATH=src python scripts/phase7_dataset_manifest.py",
                "<pinned> scripts/phase7_official_v1_train.py --split <S> --seed 33 [--control] --dry-run --output-dir results/runs/phase7/dry_run/<S>_<cond>_seed33",
                "<pinned> scripts/phase7_control_harness_check.py",
                "<pinned> scripts/phase7_gpu_preflight.py",
                "<pinned> scripts/phase7_signal_dependence.py --dry-run --output results/runs/phase7/dry_run/signal_dependence_dry_run.json",
                "PYTHONPATH=src python scripts/phase7_text_anchor.py",
                "PYTHONPATH=src python scripts/phase7_master_report.py",
            ],
        },
    }
    OUT_MASTER.write_text(json.dumps(master, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"classification": master["classification"], "tests": tests, "dry_runs_all_pass": dry_ok,
                      "control": control["all_pass"],
                      "integrity": {k: v for k, v in integrity.items() if isinstance(v, bool)},
                      "pinned_env": integrity["pinned_environment"]["pip_freeze_identical_to_pre_phase6_snapshot"],
                      "modified": modified}, indent=2))


if __name__ == "__main__":
    main()
