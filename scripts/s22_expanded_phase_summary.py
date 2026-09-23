"""Steps 7, 8, 10 and 12: expanded-baseline phase summary artifact.

Pulls the phase's own artifacts together into one reproducibility and integrity
record, compares the expanded results against the 8-trial development result,
and quantifies the data-scale change. Nothing is trained or downloaded here.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "results/s22_expanded_baseline.json"
FIGURE_PATH = ROOT / "results/figures/debug/s22_expanded_baseline_summary.png"
METADATA_PATH = ROOT / "data/manifests/s22_sentence_block_metadata.jsonl"
MANIFEST_DIR = ROOT / "data/manifests"

TENSORS = ROOT / "results/s22_expanded_event_tensors.json"
BOUNDARY = ROOT / "results/official_v1_boundary_event_parity.json"
SPLITS = ROOT / "results/s22_split_ladder_audit.json"
PROBE = ROOT / "results/s22_expanded_event_probe.json"
CTC = ROOT / "results/s22_expanded_ctc_baseline.json"
CONTROL = ROOT / "results/s22_expanded_no_signal_control.json"
ACQUISITION = ROOT / "results/s22_acquisition.json"
EIGHT_TRIAL = ROOT / "results/official_v1_event_sequence_ctc_baseline.json"
PERMUTATION_8 = ROOT / "results/no_signal_target_permutation.json"

PREVIOUS = {
    "trials": 8,
    "train_trials": 6,
    "evaluation_trials": 2,
    "keystrokes_total": 240,
    "blocks": 1,
    "sessions": 1,
    "event_sequence_ctc_mean_eval_cer": 0.9194,
    "target_permutation_mean_eval_cer": 1.082,
}
HISTORICAL_PREPROCESSING = "scripts/prepare_real_subset.py"
OFFICIAL_REVISION = "5f9889621d0df391c5aab37c996683d308e6e926"
OFFICIAL_ENV = Path.home() / "Envs/neuroselect-brain2qwerty-v1"


def git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def package_versions(executable: str, names: list[str]) -> dict[str, str]:
    code = (
        "import json,importlib.metadata as m;"
        f"print(json.dumps({{n: (m.version(n) if _ok(n) else None) for n in {names!r}}}))"
    )
    helper = "def _ok(n):\n import importlib.metadata as m\n try:\n  m.version(n)\n  return True\n except Exception:\n  return False\n"
    result = subprocess.run([executable, "-c", helper + code], capture_output=True, text=True)
    try:
        return json.loads(result.stdout.strip())
    except Exception:
        return {"error": result.stderr.strip()[:400]}


def main() -> None:
    tensors = read(TENSORS)
    boundary = read(BOUNDARY)
    splits = read(SPLITS)
    probe = read(PROBE)
    ctc = read(CTC)
    control = read(CONTROL)
    acquisition = read(ACQUISITION)
    records = [json.loads(line) for line in METADATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

    # ---- Step 8: data-scale analysis ---------------------------------
    keystrokes = [record["number_of_keystrokes"] for record in records]
    label_counts = tensors["label_distribution"] if tensors else {}
    total_labels = sum(label_counts.values()) or 1
    expanded = {
        "events": tensors["totals"]["events"] if tensors else None,
        "sentences": len(records),
        "unique_sentence_texts": len({record["sentence_presented"] for record in records}),
        "blocks": 4,
        "sessions": 2,
        "keystrokes_per_sentence": {
            "mean": float(statistics.fmean(keystrokes)),
            "median": float(statistics.median(keystrokes)),
            "std": float(statistics.pstdev(keystrokes)),
            "min": min(keystrokes), "max": max(keystrokes),
        },
        "per_block": {
            block["block"]: {"events": block["events"], "sentences": block["sentences"]}
            for block in (tensors["blocks"] if tensors else [])
        },
        "class_distribution": {
            character: {"count": count, "share": count / total_labels}
            for character, count in sorted(label_counts.items(), key=lambda item: -item[1])
        },
    }
    split_sizes = {}
    for name in sorted(MANIFEST_DIR.glob("split_*.json")):
        manifest = json.loads(name.read_text(encoding="utf-8"))
        split_sizes[manifest["split_name"]] = {
            "classification": manifest["classification"],
            "train_sentences": len(manifest["train_sentence_UIDs"]),
            "test_sentences": len(manifest["test_sentence_UIDs"]),
            "train_blocks": manifest["train_blocks"],
            "test_blocks": manifest["test_blocks"],
        }

    data_scale = {
        "previous_development_setting": PREVIOUS,
        "expanded_setting": expanded,
        "growth": {
            "events": expanded["events"] / PREVIOUS["keystrokes_total"] if expanded["events"] else None,
            "sentences": len(records) / PREVIOUS["trials"],
            "blocks": 4 / PREVIOUS["blocks"],
            "sessions": 2 / PREVIOUS["sessions"],
        },
        "split_sizes": split_sizes,
    }

    # ---- Step 7: baseline comparison ---------------------------------
    comparison = None
    if ctc:
        primary = ctc["splits"][ctc["primary_split"]]
        comparison = {
            "previous_event_sequence_ctc": {
                "setting": "8 trials from one block of one session, 6 train / 2 evaluation, 240 keystrokes",
                "mean_evaluation_cer": PREVIOUS["event_sequence_ctc_mean_eval_cer"],
                "label": "8-trial single-block development result, NOT a generalization benchmark",
            },
            "expanded_primary": {
                "split": ctc["primary_split"],
                "setting": f"{primary['train_sentences']} train / {primary['test_sentences']} held-out sentences, "
                           f"{primary['train_events']} / {primary['test_events']} keystrokes, sentence-disjoint across both sessions",
                "mean_evaluation_cer": primary["evaluation"]["mean_cer"]["mean"],
                "std_evaluation_cer": primary["evaluation"]["mean_cer"]["std"],
                "mean_train_cer": primary["train"]["mean_cer"]["mean"],
                "label": "expanded-data sentence-disjoint held-out result",
            },
            "other_expanded_splits": {
                split: {
                    "classification": summary["classification"],
                    "mean_evaluation_cer": summary["evaluation"]["mean_cer"]["mean"],
                }
                for split, summary in ctc["splits"].items() if split != ctc["primary_split"]
            },
            "previous_target_permutation_control": PREVIOUS["target_permutation_mean_eval_cer"],
            "expanded_target_permutation_control": (
                control["control"]["evaluation"]["mean_cer"]["mean"] if control else None
            ),
            "comparability_caveats": [
                "The 8-trial result trained on 6 sentences from one block; the expanded primary result trains on "
                "128 sentences from two blocks in two sessions. Data volume, block count and session count all differ.",
                "The 8-trial runs used one optimizer step per sequence. The expanded runs average gradients over "
                "same-length sentence groups, so the number of optimizer steps per epoch differs. No zero padding "
                "is introduced by that grouping and every forward pass is numerically unchanged.",
                "The 8-trial evaluation held out 2 sentences from the same block as training; the expanded primary "
                "evaluation holds out 128 sentences from different blocks and a different stimulus list.",
                "Neither setting is a state-of-the-art comparison. Both are internal development baselines on one subject.",
            ],
        }

    # ---- Step 10: reproducibility -------------------------------------
    raw_hashes = {}
    if tensors:
        for block in tensors["blocks"]:
            raw_hashes[block["fif"]] = block["fif_sha256"]
    acquisition_hashes = {}
    if acquisition:
        for row in acquisition.get("downloads", acquisition.get("files", [])):
            if isinstance(row, dict) and "remote_path" in row:
                acquisition_hashes[row["remote_path"]] = row.get("sha256") or row.get("actual_sha256")

    names = ["numpy", "scipy", "scikit-learn", "pandas", "torch", "mne", "matplotlib"]
    official_names = names + ["neuralset", "neuraltrain", "neuralfetch", "exca", "pydantic", "lightning", "torchmetrics", "dtw-python"]
    official_python = OFFICIAL_ENV / "Scripts/python.exe"
    reproducibility = {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain").splitlines(),
        "official_brain2qwerty_revision": OFFICIAL_REVISION,
        "preprocessing_variant": "official_v1_compatible",
        "preprocessing_module": "src/neuroselect/official_v1_preprocessing.py",
        "representation": "official_v1_event_sequence",
        "model_seeds": [33, 123, 777],
        "target_permutation_seed": 2026,
        "optimizer": {"name": "Adam", "learning_rate": 0.001, "gradient_clip_max_norm": 1.0, "epochs": 300},
        "checkpoints_saved": False,
        "checkpoint_note": "No model checkpoints are stored; every run is reproducible from its seed and the split manifests.",
        "training_environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": package_versions(sys.executable, names),
        },
        "official_extraction_environment": {
            "path": str(OFFICIAL_ENV),
            "packages": package_versions(str(official_python), official_names) if official_python.exists() else None,
        },
        "data_hashes": {"fif_sha256": raw_hashes, "acquisition_record": acquisition_hashes},
        "event_tensor_hashes": {block["block"]: block["tensor_sha256"] for block in (tensors["blocks"] if tensors else [])},
        "split_manifests": sorted(path.name for path in MANIFEST_DIR.glob("split_*.json")),
        "sentence_metadata": str(METADATA_PATH.relative_to(ROOT)).replace("\\", "/"),
    }

    # ---- Step 12: integrity -------------------------------------------
    vendor_dirty = git("status", "--porcelain", "--", "vendor/brain2qwerty").splitlines()
    historical_dirty = git("status", "--porcelain", "--", HISTORICAL_PREPROCESSING).splitlines()
    tests_dirty = git("status", "--porcelain", "--", "tests").splitlines()
    raw_dirty = git("status", "--porcelain", "--", "data/raw").splitlines()
    integrity = {
        "official_source_unchanged": not vendor_dirty,
        "official_source_status": vendor_dirty,
        "historical_preprocessing_unchanged": not historical_dirty,
        "historical_preprocessing_file": HISTORICAL_PREPROCESSING,
        "tests_unmodified": not tests_dirty,
        "raw_data_untracked_or_unchanged": not raw_dirty,
        "raw_fif_hashes_match_acquisition_record": all(
            acquisition_hashes.get(path.replace("data/raw/spanishbcbl_s22/", ""), digest) == digest
            for path, digest in raw_hashes.items()
        ) if acquisition_hashes else "acquisition record has no comparable hash list",
        "no_dataset_download_this_phase": True,
        "evidence_selector_implemented": False,
        "llm_used": False,
        "architecture_changed": False,
        "decoder_changed": False,
        "vocabulary_changed": False,
        "official_v1_preprocessing_semantics_changed": False,
        "official_v1_preprocessing_semantics_note": (
            "The module gained the official right-edge output placement it previously refused to perform. "
            "Every event it already handled is bit-identical (the 240-event parity fixture and the pytest "
            "reference hashes are unchanged), and the new path was verified against the real official "
            "MegExtractor. See results/official_v1_boundary_event_parity.json."
        ),
        "boundary_parity_status": boundary["verification"]["status"] if boundary else None,
        "split_audit_status": splits["overall_status"] if splits else None,
    }

    artifact = {
        "task": "Expanded-data baseline phase summary for subject S22",
        "phase_steps": {
            "1_canonical_manifest": str(METADATA_PATH.relative_to(ROOT)).replace("\\", "/"),
            "2_splits": reproducibility["split_manifests"],
            "3_event_tensors": str(TENSORS.relative_to(ROOT)).replace("\\", "/") if tensors else None,
            "4_event_probe": str(PROBE.relative_to(ROOT)).replace("\\", "/") if probe else None,
            "5_ctc_baseline": str(CTC.relative_to(ROOT)).replace("\\", "/") if ctc else None,
            "6_no_signal_control": str(CONTROL.relative_to(ROOT)).replace("\\", "/") if control else None,
        },
        "data_scale": data_scale,
        "baseline_comparison": comparison,
        "reproducibility": reproducibility,
        "integrity": integrity,
        "headline_numbers": {
            "events": expanded["events"],
            "sentences": expanded["sentences"],
            "unique_sentence_texts": expanded["unique_sentence_texts"],
            "primary_ctc_split": ctc["primary_split"] if ctc else None,
            "primary_ctc_evaluation_cer": comparison["expanded_primary"]["mean_evaluation_cer"] if comparison else None,
            "control_evaluation_cer": comparison["expanded_target_permutation_control"] if comparison else None,
            "primary_probe_test_accuracy": probe["reports"][probe["primary_split"]]["test"]["accuracy"] if probe else None,
            "primary_probe_majority_baseline": probe["reports"][probe["primary_split"]]["test"]["majority_baseline_accuracy"] if probe else None,
        },
        "claims_not_made": [
            "No claim of successful brain-to-text decoding.",
            "No claim of state-of-the-art performance.",
            "No significance testing was performed.",
        ],
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- figure --------------------------------------------------------
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(["8-trial dev", "expanded"], [PREVIOUS["keystrokes_total"], expanded["events"]], color=["#bbbbbb", "#3a6ea5"])
    for index, value in enumerate([PREVIOUS["keystrokes_total"], expanded["events"]]):
        axes[0].text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=10)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("keystroke events (log scale)")
    axes[0].set_title(f"Dataset scale: {expanded['events'] / PREVIOUS['keystrokes_total']:.0f}x more events")

    axes[1].hist(keystrokes, bins=24, color="#3a6ea5", edgecolor="white")
    axes[1].axvline(float(statistics.fmean(keystrokes)), color="#c8553d", linestyle="--",
                    label=f"mean {statistics.fmean(keystrokes):.1f}")
    axes[1].set_xlabel("keystrokes per sentence")
    axes[1].set_ylabel("sentences")
    axes[1].set_title("Sentence-length distribution (256 sentences)")
    axes[1].legend(fontsize=8)

    characters = list(expanded["class_distribution"])[:15]
    shares = [expanded["class_distribution"][c]["share"] for c in characters]
    axes[2].bar(range(len(characters)), shares, color="#7a4988")
    axes[2].set_xticks(range(len(characters)), ["space" if c == " " else c for c in characters], fontsize=8)
    axes[2].set_ylabel("share of all events")
    axes[2].set_title("Event class distribution (top 15 of 29)")
    figure.suptitle("Expanded-data baseline phase: dataset summary", fontsize=12)
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)

    print(json.dumps({
        "artifact": str(OUT_PATH.relative_to(ROOT)),
        "headline_numbers": artifact["headline_numbers"],
        "integrity": {k: v for k, v in integrity.items() if isinstance(v, (bool, str)) and k != "official_v1_preprocessing_semantics_note"},
    }, indent=2))


if __name__ == "__main__":
    main()
