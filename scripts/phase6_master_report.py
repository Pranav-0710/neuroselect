"""Phase 6: assemble the official-v1 audit, outcome, reproducibility and integrity record.

Reads the measured Phase 6 artifacts (smoke test, text anchor, leakage audit),
records the official model audit from the vendored source, states which parts
could and could not be executed and why, runs the full NeuroSelect test suite,
re-hashes the raw data, and checks integrity. Trains nothing.
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

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures/debug"
VENDOR = ROOT / "vendor/brain2qwerty/brain2qwerty_v1"
DATA_ROOT = ROOT / "data/raw/spanishbcbl_s22"
OVERLAY = Path.home() / "Envs/phase6-official-overlay"
PINNED_PYTHON = Path.home() / "Envs/neuroselect-brain2qwerty-v1/Scripts/python.exe"
OFFICIAL_REVISION = "5f9889621d0df391c5aab37c996683d308e6e926"

SMOKE = RESULTS / "phase6_official_v1_smoke_test.json"
ANCHOR = RESULTS / "phase6_text_anchor.json"
LEAKAGE = RESULTS / "phase6_leakage_audit.json"
CTC_5Z = RESULTS / "s22_expanded_ctc_baseline.json"
CONTROL_5Z = RESULTS / "s22_expanded_no_signal_control.json"
TENSORS = RESULTS / "s22_expanded_event_tensors.json"
ACQUISITION = RESULTS / "s22_acquisition.json"

OUT = {
    "audit": RESULTS / "phase6_official_v1_model_audit.json",
    "baseline": RESULTS / "phase6_official_v1_baseline.json",
    "control": RESULTS / "phase6_official_v1_no_signal_control.json",
    "dependence": RESULTS / "phase6_signal_dependence.json",
    "master": RESULTS / "phase6_master_report.json",
}
FIG = {
    "real_vs_control": FIGURES / "phase6_real_vs_control.png",
    "outputs": FIGURES / "phase6_outputs.png",
    "feasibility": FIGURES / "phase6_feasibility.png",
}
NOT_CREATED = {
    "results/figures/debug/phase6_signal_ablation.png": (
        "Not created: the signal ablation needs a trained official model and none could be trained. "
        "Ablating an untrained network would measure only that random weights depend on their input."
    ),
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


def official_dict(path: Path, name: str) -> dict:
    """Evaluate a module-level dict literal from the official config source."""
    source = path.read_text(encoding="utf-8")
    match = re.search(rf"^{name} = (\{{.*?^\}})", source, re.S | re.M)
    if match is None:
        raise RuntimeError(f"{name} not found in {path}")
    return eval(match.group(1), {"__builtins__": {}}, {"None": None, "True": True, "False": False})


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


def pinned_env_unchanged() -> dict:
    before = (OVERLAY / "pinned_env_freeze_before.txt").read_text(encoding="utf-8")
    now = subprocess.run([str(PINNED_PYTHON), "-m", "pip", "freeze", "--all"],
                         capture_output=True, text=True).stdout
    return {"packages": len(before.splitlines()), "pip_freeze_identical_to_pre_phase6_snapshot": before == now,
            "packages_list": before.splitlines()}


def raw_file_hashes() -> dict:
    acquisition = read(ACQUISITION)
    expected = {row["remote_path"]: row["actual_sha256"] for row in acquisition["downloads"]}
    expected.update(acquisition["protected_file_hashes"])
    files = []
    for path in sorted([*DATA_ROOT.glob("MEG/FIF/**/*.fif"), *DATA_ROOT.glob("MEG/logs/*.mat")]):
        relative = path.relative_to(DATA_ROOT).as_posix()
        digest = sha256_file(path)
        files.append({"file": relative, "sha256": digest, "matches_acquisition_record": expected.get(relative) == digest})
    return {"files": files, "all_match": all(f["matches_acquisition_record"] for f in files)}


def main() -> None:
    smoke, anchor, leakage = read(SMOKE), read(ANCHOR), read(LEAKAGE)
    ctc5z, control5z = read(CTC_5Z), read(CONTROL_5Z)
    feasibility = smoke["feasibility"]
    encoder = official_dict(VENDOR / "config/model_config.py", "ENCODER")
    transformer = official_dict(VENDOR / "config/model_config.py", "TRANSFORMER")
    xp_source = (VENDOR / "config/xp_config.py").read_text(encoding="utf-8")

    # ---- Part 1: official v1 model audit ------------------------------
    audit = {
        "task": "Phase 6 Part 1: official Brain2Qwerty v1 model audit",
        "official_revision": OFFICIAL_REVISION,
        "sources_inspected": [
            "vendor/brain2qwerty/brain2qwerty_v1/config/model_config.py",
            "vendor/brain2qwerty/brain2qwerty_v1/config/xp_config.py",
            "vendor/brain2qwerty/brain2qwerty_v1/main.py",
            "vendor/brain2qwerty/brain2qwerty_v1/pl_module.py",
            "vendor/brain2qwerty/brain2qwerty_v1/utils.py",
            "vendor/brain2qwerty/brain2qwerty_v1/metrics.py",
            "vendor/brain2qwerty/brain2qwerty_v1/transforms.py",
            "neuraltrain 0.2.2: models/simpleconv.py, models/common.py, models/transformer.py",
            "neuralset 0.2.2: extractors/neuro.py (ChannelPositions, MegExtractor), dataloader.py",
            "x-transformers 2.4.9 (realized module introspected)",
        ],
        "official_encoder_config": encoder,
        "official_transformer_config": transformer,
        "formulation": ("Per-keystroke classification, not sequence transduction. Every keystroke window is encoded "
                        "independently, keystrokes are grouped by sentence for a transformer, and each keystroke is "
                        "classified into one of 29 characters. No CTC, no blank."),
        "input": {
            "neural_tensor": "(B, 306, 25): one keystroke window [-0.2, +0.3) s at 50 Hz, channels-first",
            "batch_size": {"train": 64, "val": 2048, "test": 2048},
            "batch_unit": "keystrokes, not sentences",
            "realized_and_verified": smoke["smoke_subset"]["batch_parity"],
        },
        "channel_positions": {
            "extractor": "ChannelPositions2D (official subclass re-enabling 2D layouts for MEG)",
            "source": "mne.find_layout(raw.info) 2D sensor layout",
            "normalization": "min-max to [0, 1] per dimension; channels without a layout position set to -0.1 and masked",
            "shape": "(B, 306, 2)",
            "use": "Fourier-embedded (2D, 32 frequencies per dimension -> 2048 features) and used as keys of the "
                   "per-subject spatial attention in the ChannelMerger",
            "s22_realized": {"masked_channels": smoke["smoke_subset"]["channel_positions_masked_invalid"],
                             "identical_across_blocks": smoke["smoke_subset"]["channel_positions_identical_across_all_keystrokes"]},
        },
        "subject_conditioning": {
            "subject_id": "LabelEncoder over the Meg events' subject field, shape (B, 1); 0 for S22",
            "channel_merger": "per_subject=True: heads (n_subjects=200, 270, 2048) selected by subject id",
            "subject_layers": ("subject_layers_config={} in the official config parses to an enabled SubjectLayers "
                               "with defaults: a per-subject 512->2048 linear map, weights (200, 512, 2048) + bias "
                               "(200, 2048)"),
            "subject_slots": 200,
            "per_subject_table_parameters": feasibility["per_subject_table_parameters"],
        },
        "encoder": {
            "class": "neuraltrain SimpleConvTimeAgg",
            "stages": [
                "input dropout 0.2 is applied to the first conv layer's input",
                "ChannelMerger: 306 sensors -> 270 virtual channels via Fourier-position spatial attention "
                "(spatial dropout radius 0.2; usage_penalty 1.0 is computed but BrainModule never adds it to the loss)",
                "initial_linear: Conv1d 270 -> 512, kernel 1",
                "SubjectLayers: per-subject 512 -> 2048",
                "ConvSequence: 8 x Conv1d 2048 -> 2048, kernel 3, dilations 1,2,4,1,2,4,1,2 (growth 2, period 3), "
                "BatchNorm, GELU (gelu=True overrides relu_leakiness), dropout 0.5, residual skip with LayerScale 0.1, "
                "no activation after the last layer",
                "time aggregation: Bahdanau attention (hidden 256) over the 25 samples -> one 2048-d vector",
            ],
            "hidden": 2048,
            "output": "(B, 2048), one embedding per keystroke",
        },
        "sentence_model": {
            "class": "x-transformers Encoder via neuraltrain TransformerEncoder",
            "realized": smoke["realized_transformer"],
            "dimension": 2048,
            "feed_forward_multiplier": 4,
            "unspecified_settings_take_class_defaults": ("the official config sets only alibi_pos_bias, depth and heads, "
                                                         "so use_scalenorm=True, rotary_pos_emb=True, attn_dropout=0.1, "
                                                         "ff_mult=4 and scale_residual=True apply"),
            "grouping": ("BrainModule._transformer_forward groups the keystrokes of a batch by sentence_UID, pads, "
                         "masks and runs the encoder. Grouping is within a batch only."),
        },
        "target_and_loss": {
            "target": "official LabelEncoder over Keystroke.button with BUTTON_MAPPING: 29 classes "
                      "(letters, space, <special>, <number>; rare symbols fold into <special>)",
            "relation_to_neuroselect_vocab": "official class = NeuroSelect SpanishBCBL id - 1 (no blank)",
            "head": "Linear(2048, 29)",
            "loss": "CrossEntropyLoss over keystrokes (mean over the batch)",
        },
        "decoding": {
            "procedure": "argmax per keystroke -> CHAR_INDEX; exactly one character per keystroke",
            "consequence": ("The decoder is given the keystroke segmentation: output length always equals the number "
                            "of keystrokes. Its CER is therefore not comparable with CTC CER, which must infer length."),
            "language_model": "optional KenLM beam search is a separate post-processing script, not part of training",
        },
        "train_validation_test": {
            "splitter": "Brain2QwertyV1Splitter: 80/10/10 by keystrokes over 19 pooled participants, TF-IDF cosine > 0.5 "
                        "sentence clusters kept in one split, cluster order shuffled with seed 1",
            "sampler": "SentenceGroupedDistributedSampler without shuffling: fixed sentence order every epoch; with "
                       "64-keystroke batches, sentences straddling a batch boundary reach the transformer in two parts",
            "early_stopping": "EarlyStopping on val_CER, patience 30",
            "checkpointing": "best val_CER checkpoint saved, but trainer.test runs on the final in-memory weights",
            "official_cer_metric": "Levenshtein over each batch concatenated into one string, normalized per batch "
                                   "(not per sentence, despite its docstring)",
        },
        "optimization": {
            "optimizer": "AdamW, lr 5e-5, weight decay 1e-4",
            "scheduler": "OneCycleLR, max_lr 5e-5, pct_start 0.1, stepped per batch",
            "gradient_clipping": None,
            "epochs": 300, "seed": 33, "devices": "8 GPUs by default; CPU fallback when CUDA is absent",
            "source_check": {"n_epochs_300": '"n_epochs": 300' in xp_source, "patience_30": '"patience": 30' in xp_source},
        },
        "parameters_measured": smoke["parameters"],
        "corrections_to_earlier_audit": {
            "earlier_artifact": "results/official_v1_style_baseline.json (commit c13a6e8)",
            "rotary_positional_embedding": "earlier: false. Measured: RotaryEmbedding is active alongside ALiBi.",
            "normalization": "earlier: x-transformers default (LayerNorm). Measured: ScaleNorm.",
            "parameter_count": f"earlier: not obtained. Measured: {feasibility['exact_parameters']:,}.",
            "subject_layers": f"earlier audit did not record them. Measured: {smoke['parameters']['subject_layers']:,} parameters.",
        },
        "published_benchmark_reproduced": False,
    }

    # ---- Part 4: baseline -----------------------------------------------
    official_hours = feasibility["estimated_hours_per_run"]
    baseline = {
        "task": "Phase 6 Part 4: official v1 baseline on S22 splits C, D, E",
        "execution_status": "NOT EXECUTED",
        "reason": "The exact official architecture cannot be trained on this machine, at any epoch count.",
        "blockers": {
            "memory": {
                "training_static_memory_gib": feasibility["training_static_memory_gb"],
                "estimated_training_process_gib": feasibility["estimated_training_process_commit_gb"],
                "system_available_commit_gib_with_model_loaded": feasibility["system_available_commit_gb_with_model_loaded"],
                "system_available_physical_gib_with_model_loaded": feasibility["system_available_physical_gb_with_model_loaded"],
                "system_total_physical_gib": feasibility["system_total_physical_gb"],
                "explanation": (f"AdamW training holds four fp32 copies of all {feasibility['exact_parameters'] / 1e6:.1f}M parameters (weights, gradients, two "
                                "moment buffers). Reducing epochs does not reduce this. Even with every other "
                                "application closed it would need most of the 15.6 GB of RAM."),
            },
            "compute": {
                "basis": feasibility["basis"],
                "forward_seconds_per_keystroke": smoke["forward_check"]["forward_seconds_per_keystroke"],
                "estimated_seconds_per_epoch": feasibility["estimated_seconds_per_epoch"],
                "estimated_hours_per_run": official_hours,
                "epochs_fitting_2_hours": feasibility["epochs_fitting_2_hours"],
                "twelve_run_total_hours_at_100_epochs": float(
                    3 * sum(official_hours["diagnostic_100_epochs"].values())
                    + 3 * official_hours["diagnostic_100_epochs"]["C"]),
            },
            "gpu_only_operations": False,
            "gpu_note": "No GPU-only operation was found; the model runs on CPU (forward verified). The limits are memory and time.",
        },
        "lightweight_diagnostic_assessment": (
            f"The brief allows the same architecture with fewer epochs. A 2-hour run fits "
            f"{feasibility['epochs_fitting_2_hours']['C']} epochs on C "
            f"({feasibility['epochs_fitting_2_hours']['C'] * feasibility['optimizer_steps_per_epoch']['C']} optimizer "
            f"steps), {feasibility['epochs_fitting_2_hours']['D']} on D and {feasibility['epochs_fitting_2_hours']['E']} "
            f"on E, too few to train a {feasibility['exact_parameters'] / 1e6:.1f}M-parameter model at the official "
            "5e-5 learning rate, and memory prevents even that. Not run."),
        "not_done_deliberately": [
            "No architecture change: n_subjects=1 would shrink the per-subject tables while keeping the single-subject "
            "function, but it alters the official model and still exceeds available memory.",
            "No mixed precision, no gradient checkpointing, no reduced hidden size.",
        ],
        "exact_official_parameters": audit["optimization"] | {"batch_size": 64, "val_test_batch_size": 2048},
        "planned_protocol_if_executable": {
            "splits": {"C": "train s1/b1 + s2/b2 (list1) -> test s1/b2 + s2/b1 (list2)",
                       "D": "train s1/b1 (list1) -> test s2/b1 (list2)",
                       "E": "train s1/b2 (list2) -> test s2/b2 (list1)"},
            "seeds": [33, 123, 777],
            "inputs": "validated official-v1 slab, proven identical to the official SegmentDataset batch",
            "validation": "none: the splits have no validation side, so EarlyStopping and best-checkpointing are omitted "
                          "and the final weights are evaluated, which is also what the official trainer.test uses",
        },
        "results": None,
        "executed_instead": "results/phase6_official_v1_smoke_test.json (forward-only smoke test, PASS)",
    }

    control = {
        "task": "Phase 6 Part 5: no-valid-signal control with the official v1 architecture",
        "execution_status": "NOT EXECUTED",
        "reason": "Depends on training the official v1 model, which is not executable here (see the baseline artifact).",
        "planned_specification": {
            "signals": "unchanged", "sequence_lengths": "unchanged",
            "targets_permuted": "training sentences only, derangement, shuffle seed 2026, same permutation for seeds 33/123/777",
            "evaluation_targets": "correctly paired",
            "design_note": ("Derange at the stimulus-group level: in split C a UID-level derangement maps 3 of 128 "
                            "training sentences onto their own text from the other session (Phase 5Z)."),
        },
        "results": None,
    }

    dependence = {
        "task": "Phase 6 Part 7: test-time signal-dependence of the trained real-label model",
        "execution_status": "NOT EXECUTED",
        "reason": ("Requires a trained official model and none could be trained. Ablating the untrained network was not "
                   "done: it would show only that random weights respond to their input."),
        "planned_conditions": {
            "A": "original signal",
            "B": "temporal shuffle within each event (fixed seed)",
            "C": "zeroed MEG",
            "D": "one fixed random permutation of the 306 channels, positions unchanged",
        },
        "planned_measures": ["character F1", "CER", "decoded length", "agreement with condition-A predictions"],
        "formulation_note": "Decoded length is fixed by the keystroke count in v1, so only character identity can change.",
        "results": None,
    }

    # ---- reproducibility, tests, integrity ------------------------------
    pinned = pinned_env_unchanged()
    raw = raw_file_hashes()
    tensors = read(TENSORS)
    tests = run_tests()
    manifests = {path.name: sha256_file(path) for path in sorted((ROOT / "data/manifests").glob("split_[CDE]_*.json"))}
    tracked_changes = git("status", "--porcelain", "--", "results", "vendor", "tests", "src",
                          "scripts/prepare_real_subset.py").splitlines()
    modified_tracked = [line for line in tracked_changes if not line.startswith("??")]
    integrity = {
        "official_source_unchanged": not git("status", "--porcelain", "--", "vendor/brain2qwerty"),
        "historical_preprocessing_unchanged": not git("status", "--porcelain", "--", "scripts/prepare_real_subset.py"),
        "official_v1_preprocessing_unchanged": not git("status", "--porcelain", "--", "src/neuroselect/official_v1_preprocessing.py"),
        "tests_unmodified": not git("status", "--porcelain", "--", "tests"),
        "previous_results_unmodified": not [l for l in modified_tracked if "results/" in l],
        "modified_tracked_files": modified_tracked,
        "raw_fif_and_mat_unchanged": raw["all_match"],
        "event_clean_artifacts": {
            name: {"sha256": sha256_file(DATA_ROOT / name),
                   "modified": os.path.getmtime(DATA_ROOT / name)}
            for name in ("events_clean.pkl", "events_clean_all_blocks.pkl", "events_raw.pkl")
        },
        "event_clean_artifacts_unchanged_note": ("modification times predate Phase 6, and the smoke test rebuilt the "
                                                 "events through the official study and found them identical"),
        "pinned_official_environment_unchanged": pinned["pip_freeze_identical_to_pre_phase6_snapshot"],
        "dataset_downloads": False,
        "software_downloads": ("Eight wheels were downloaded into a separate overlay because the pinned environment "
                               "lacked the official lock's sentence-transformer and metric dependencies"),
        "evidence_selector": False,
        "llm": False,
        "architecture_modified": False,
    }
    reproducibility = {
        "git_commit": git("rev-parse", "HEAD"),
        "official_brain2qwerty_revision": OFFICIAL_REVISION,
        "pinned_environment": {"python": str(PINNED_PYTHON), "packages": pinned["packages_list"],
                               "unchanged": pinned["pip_freeze_identical_to_pre_phase6_snapshot"]},
        "overlay": {
            "path": str(OVERLAY / "site"),
            "why": "official requirements.lock pins these; the pinned environment did not contain them",
            "packages_from_official_lock": {"x-transformers": "2.4.9", "einops": "0.8.1", "einx": "0.3.0",
                                            "loguru": "0.7.3", "frozendict": "2.4.6", "Levenshtein": "0.27.1",
                                            "RapidFuzz": "3.13.0"},
            "windows_shim_not_in_lock": {"win32-setctime": "1.2.0", "reason": "loguru's Windows-only import"},
            "install": "pip install --no-deps --target <overlay>/site from locally hashed wheels",
            "wheel_sha256": {path.name: sha256_file(path) for path in sorted((OVERLAY / "wheels").glob("*.whl"))},
        },
        "windows_deviation": ("Official study event cache disabled: exca names its cache files with the timeline as JSON, "
                              "which Windows rejects. Events are rebuilt uncached and verified identical to the stored "
                              "extraction."),
        "raw_file_sha256": raw["files"],
        "event_tensor_slab": {"file": "data/processed/s22_official_v1/events.npy",
                              "per_block_sha256": {b["block"]: b["tensor_sha256"] for b in tensors["blocks"]},
                              "index_sha256": sha256_file(ROOT / "data/processed/s22_official_v1/index.jsonl")},
        "preprocessing": {"variant": "official_v1_compatible",
                          "module_sha256": sha256_file(ROOT / "src/neuroselect/official_v1_preprocessing.py")},
        "splits": manifests,
        "seeds_planned": [33, 123, 777], "permutation_seed_planned": 2026,
        "epochs": {"official": 300, "run": 0},
        "optimizer": audit["optimization"],
        "batch_size": {"train": 64, "test": 2048},
        "model_settings": {"encoder": encoder, "transformer": transformer},
        "runtime_seconds": {"smoke_test": smoke["seconds"]},
        "cpu": smoke["environment"]["cpu"],
        "torch_threads": smoke["environment"]["torch_threads"],
        "cuda_available": smoke["environment"]["cuda_available"],
    }

    simplified_c = ctc5z["splits"]["C"]["evaluation_error_decomposition"]
    control_c = control5z["control"]["evaluation_error_decomposition"]
    full = official_hours["official_300_epochs"].values()
    diagnostic = official_hours["diagnostic_100_epochs"].values()
    fits = feasibility["epochs_fitting_2_hours"].values()
    total_days = baseline["blockers"]["compute"]["twelve_run_total_hours_at_100_epochs"] / 24
    master = {
        "task": "Phase 6: official Brain2Qwerty v1 keystroke-level baseline on expanded S22",
        "interpretation": "D",
        "interpretation_text": "Official baseline cannot be executed reliably",
        "interpretation_basis": [
            f"The exact official model needs about {feasibility['estimated_training_process_commit_gb']:.0f} GiB to "
            f"train ({feasibility['exact_parameters'] / 1e6:.1f}M parameters x 4 fp32 copies) against "
            f"{feasibility['system_available_commit_gb_with_model_loaded']:.1f} GiB of available commit on this "
            f"{feasibility['system_total_physical_gb']:.1f} GiB laptop.",
            f"Compute alone puts the official schedule at {min(full):.0f}-{max(full):.0f} hours per run and the "
            f"100-epoch diagnostic at {min(diagnostic):.0f}-{max(diagnostic):.0f} hours per run "
            f"(12 runs, about {total_days:.0f} days); 2 hours buys {min(fits)}-{max(fits)} epochs.",
            "No reduced-epoch run of the same architecture fits in memory, and changing the architecture was not allowed.",
        ],
        "what_was_established": {
            "official_model_runs_on_cpu": True,
            "inputs_identical_to_official_dataloader": smoke["smoke_subset"]["all_inputs_identical_to_official_segment_dataset"],
            "finite_logits_loss_and_decoding": smoke["forward_check"]["logits_finite"] and smoke["forward_check"]["loss_finite"],
            "official_formulation": audit["formulation"],
        },
        "major_findings": {
            "paraphrase_overlap_in_list_splits": {
                "finding": ("32 of the 64 list2 sentences are a list1 sentence with one inserted modifier (TF-IDF "
                            "cosine up to 0.875). The official splitter's paraphrase rule (cosine > 0.5) would keep "
                            "these pairs together. Splits A-E are exact-text disjoint but not paraphrase-disjoint."),
                "pairs_above_official_threshold": {s: v["official_paraphrase_rule"]["pairs_above_threshold"]
                                                   for s, v in leakage["splits"].items()},
                "example": leakage["splits"]["C"]["official_paraphrase_rule"]["examples_above_threshold"][0],
                "consequence": ("Can only inflate held-out scores, so the Phase 5Z null result stands; any future "
                                "positive result on these splits would be confounded."),
            },
            "official_environment_incomplete": "the pinned environment lacked x-transformers, einops, einx, loguru, "
                                               "frozendict, Levenshtein and RapidFuzz from the official lock",
            "oracle_segmentation": audit["decoding"]["consequence"],
            "per_subject_tables": f"{feasibility['per_subject_table_parameters']:,} of {feasibility['exact_parameters']:,} "
                                  "parameters are 200-slot per-subject tables; S22 reaches one slot",
        },
        "reference_numbers_from_earlier_phases": {
            "phase5Z_simplified_ctc_C_character_f1": simplified_c["character_f1"],
            "phase5Z_ctc_control_C_character_f1": control_c["character_f1"],
            "phase6_text_anchor_character_f1": {s: a["character_f1"] for s, a in anchor["splits"].items()},
        },
        "parts": {
            "1_audit": str(OUT["audit"].relative_to(ROOT)),
            "2_smoke_test": "PASS: " + str(SMOKE.relative_to(ROOT)),
            "3_splits": "C, D, E defined; F/G excluded",
            "4_baseline": "NOT EXECUTED (memory and compute)",
            "5_control": "NOT EXECUTED (depends on 4)",
            "6_real_vs_control": "NOT AVAILABLE (depends on 4 and 5)",
            "7_signal_dependence": "NOT EXECUTED (needs a trained model)",
            "8_text_anchor": str(ANCHOR.relative_to(ROOT)),
            "9_leakage": "exact checks PASS; official paraphrase rule FAILS (32 pairs per split)",
        },
        "artifacts": {**{k: str(v.relative_to(ROOT)) for k, v in OUT.items()},
                      "smoke_test": str(SMOKE.relative_to(ROOT)),
                      "text_anchor": str(ANCHOR.relative_to(ROOT)),
                      "leakage": str(LEAKAGE.relative_to(ROOT)),
                      "figures": [str(p.relative_to(ROOT)) for p in FIG.values()],
                      "not_created": NOT_CREATED},
        "reproducibility": reproducibility,
        "tests": tests,
        "integrity": integrity,
        "claims_not_made": ["published benchmark reproduced", "mind reading", "clinical readiness",
                            "state of the art", "subject-independent decoding"],
    }

    for key, payload in (("audit", audit), ("baseline", baseline), ("control", control),
                         ("dependence", dependence), ("master", master)):
        OUT[key].write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # ---- figures --------------------------------------------------------
    FIGURES.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 5.2))
    labels = ["simplified CTC\nreal (5Z)", "simplified CTC\ncontrol (5Z)", "text anchor\n(no model)",
              "official v1\nreal", "official v1\ncontrol"]
    values = [simplified_c["character_f1"], control_c["character_f1"], anchor["splits"]["C"]["character_f1"], 0, 0]
    colours = ["#3a6ea5", "#c8553d", "#2e7d32", "white", "white"]
    bars = axis.bar(labels, values, color=colours, edgecolor="black")
    # Missing columns, not values: span the whole axis so no height can be read as a score.
    for bar in bars[3:]:
        bar.set_height(0.42)
        bar.set_hatch("//")
        bar.set_edgecolor("#cccccc")
        bar.set_linewidth(0.6)
    for index, value in enumerate(values[:3]):
        axis.text(index, value + 0.005, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    for index in (3, 4):
        axis.text(index, 0.18, f"NOT EXECUTED\nneeds ~{feasibility['estimated_training_process_commit_gb']:.0f} GiB\nto train",
                  ha="center", va="center", fontsize=9, color="#555555")
    axis.set_ylim(0, 0.42)
    axis.set_ylabel("held-out character F1, split C")
    axis.set_title("Split C character F1: the official v1 model could not be trained on this hardware")
    figure.tight_layout()
    figure.savefig(FIG["real_vs_control"], dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    available = feasibility["system_available_commit_gb_with_model_loaded"]
    needed = feasibility["estimated_training_process_commit_gb"]
    axes[0].bar(["needed to train\n(official v1)", "available commit\n(model loaded)", "total RAM"],
                [needed, available, feasibility["system_total_physical_gb"]],
                color=["#c8553d", "#3a6ea5", "#bbbbbb"])
    for index, value in enumerate([needed, available, feasibility["system_total_physical_gb"]]):
        axes[0].text(index, value + 0.2, f"{value:.1f}", ha="center", fontsize=10)
    axes[0].set_ylabel("GiB")
    axes[0].set_title(f"Memory: training needs 4 fp32 copies of {feasibility['exact_parameters'] / 1e6:.1f}M parameters")
    splits = ["C", "D", "E"]
    width = 0.35
    positions = np.arange(len(splits))
    for offset, (schedule, colour) in zip((-width / 2, width / 2), (("official_300_epochs", "#7a4988"),
                                                                    ("diagnostic_100_epochs", "#e0a458"))):
        heights = [official_hours[schedule][s] for s in splits]
        axes[1].bar(positions + offset, heights, width, label=schedule.replace("_", " "), color=colour)
        for x, value in zip(positions + offset, heights):
            axes[1].text(x, value * 1.08, f"{value:.0f} h", ha="center", fontsize=9)
    axes[1].axhline(2, color="black", linestyle="--", label="~2 h budget per run")
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions, splits)
    axes[1].set_ylabel("estimated hours per run (log scale)")
    axes[1].set_title("Compute: estimated from the measured forward cost")
    axes[1].legend(fontsize=8)
    figure.suptitle("Phase 6 feasibility of the exact official Brain2Qwerty v1 model on this CPU laptop")
    figure.tight_layout()
    figure.savefig(FIG["feasibility"], dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(13, 7))
    blocks = [
        ("Smoke test: UNTRAINED official v1 decodes (forward check only, not a result)",
         [f"{e['block']}\n  target : {e['target']}\n  decoded: {e['decoded']}"
          for e in smoke["forward_check"]["decoded_examples_untrained"][:3]]),
        ("Text anchor (split C): length-matched training sentence, no model",
         [f"target : {e['target']}\n  anchor : {e['anchor']}  (CER {e['cer']:.2f})"
          for e in anchor["splits"]["C"]["examples"][:3]]),
        ("Official paraphrase rule: list1 (train) vs list2 (test) pairs above cosine 0.5",
         [f"{e['cosine']:.3f}  train: {e['train']}\n        test : {e['test']}"
          for e in leakage["splits"]["C"]["official_paraphrase_rule"]["examples_above_threshold"][:4]]),
    ]
    for axis, (title, lines) in zip(axes, blocks):
        axis.axis("off")
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.text(0.0, 0.95, "\n".join(lines), family="monospace", fontsize=9, va="top", transform=axis.transAxes)
    figure.tight_layout()
    figure.savefig(FIG["outputs"], dpi=160)
    plt.close(figure)

    print(json.dumps({"interpretation": master["interpretation"], "tests": tests,
                      "integrity": {k: v for k, v in integrity.items() if isinstance(v, bool)}}, indent=2))


if __name__ == "__main__":
    main()
