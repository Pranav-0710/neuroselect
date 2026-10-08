"""Phase 7: exact official Brain2Qwerty v1 harness on the formal clean splits.

Everything model- and optimisation-side is the official code at revision
5f98896, unmodified and unparameterised beyond the official config:
``experiment_config()`` builds the ``Experiment``; ``Data.build`` builds the
loaders (official MegExtractor, LabelEncoder, subject ids, channel positions,
SentenceGroupedDistributedSampler); ``Experiment._build_modules``,
``BrainModule``, ``materialize_lazy_params`` and ``Experiment._trainer_setup``
build the model and the Lightning trainer (AdamW 5e-5, wd 1e-4, OneCycleLR,
300 epochs, EarlyStopping on val_CER with patience 30, ModelCheckpoint, no
clipping). ``run_official`` is a line-for-line mirror of ``Experiment.run``
that keeps the loaders so the test partition can be scored afterwards; the
source hash of ``Experiment.run`` is checked against the audited one.

Two events transforms replace the official splitter, and only it:
* ``Phase7ManifestSplit`` assigns train/val/test from a formal split manifest
  (for the official split these equal the official splitter's own output);
* ``Phase7TrainTargetDerangement`` (control only) relabels the ``button`` field
  of training keystrokes with a deranged sentence's labels. MEG is untouched.

Environment: the official requirements.lock (Linux, CUDA 12.4, torch 2.6.0)
plus ``src`` and ``vendor/brain2qwerty`` on PYTHONPATH. Locally (Windows, CPU)
the pinned environment plus the Phase 6 overlay runs everything except
training.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import subprocess
import sys
import time
import typing as tp
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("phase6_harness", ROOT / "scripts/phase6_official_v1_harness.py")
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)  # sets the official env vars and imports the official modules
_spec = importlib.util.spec_from_file_location("phase7_common", ROOT / "scripts/phase7_common.py")
c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c)

np, torch, ns, pd = h.np, h.torch, h.ns, h.pd
import lightning.pytorch as pl  # noqa: E402
from neuralset.events.study import EventsTransform  # noqa: E402

from brain2qwerty_v1.config.xp_config import experiment_config  # noqa: E402
from brain2qwerty_v1.main import Experiment  # noqa: E402

DATASET_MANIFEST = ROOT / "data/manifests/s22_official_v1_dataset.json"
OFFICIAL_SPLIT_FILES = [ROOT / f"data/manifests/official_v1_clean_{p}.json" for p in c.PARTITIONS]
SPLIT_CHOICES = {
    "official_v1_clean": OFFICIAL_SPLIT_FILES,
    "D_clean": [ROOT / "data/manifests/s22_D_clean.json"],
    "E_clean": [ROOT / "data/manifests/s22_E_clean.json"],
}
SEEDS = (33, 123, 777)
LOCK_FILE = ROOT / "vendor/brain2qwerty/requirements.lock"
# sha256 of inspect.getsource(Experiment.run) at revision 5f98896, audited in Phase 7.
AUDITED_RUN_SOURCE_SHA256 = "50cf428f68c6c13b27ec60bc23af868f2172d2dab5838e2b79756bda89f4904f"
ABLATION_SEED = 2026


# ---- events transforms that replace the official splitter -----------------

class Phase7ManifestSplit(EventsTransform):
    """Train/val/test from a formal manifest, written like the official splitter.

    Only keystroke rows receive a split (as in the official splitter, whose
    sentence-text map leaves non-keystroke rows unassigned); sentences outside
    every partition stay unassigned and are never loaded.
    """

    partitions: dict[str, list[str]]

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        uid = events["sentence_UID"].astype(str)
        keystroke = events["type"] == "Keystroke"
        events["split"] = pd.Series(np.nan, index=events.index, dtype=object)
        for name, uids in self.partitions.items():
            events.loc[keystroke & uid.isin(set(uids)), "split"] = name
        return events


class Phase7TrainTargetDerangement(EventsTransform):
    """No-valid-signal control: training keystrokes take a deranged sentence's labels.

    For each training sentence, the ``button`` values of its keystrokes (time
    order) are replaced by the donor sentence's original button values, fitted
    to the keystroke count by ``phase7_common.adapt_labels``. Only rows already
    assigned to ``train`` are touched; MEG, timing and every evaluation target
    stay as recorded.
    """

    donor_of_uid: dict[str, str]
    last_stats: tp.ClassVar[dict] = {}

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        keystrokes = events[events["type"] == "Keystroke"].sort_values("start", kind="stable")
        rows_of = {uid: list(group.index) for uid, group in keystrokes.groupby(keystrokes["sentence_UID"].astype(str))}
        original = {uid: [str(b) for b in events.loc[rows, "button"]] for uid, rows in rows_of.items()}
        changed = total = adapted = 0
        for uid, donor in self.donor_of_uid.items():
            rows = rows_of[uid]
            if set(events.loc[rows, "split"]) != {"train"}:
                raise RuntimeError(f"Control relabelling reached a non-training sentence: {uid}")
            new = c.adapt_labels(original[donor], len(rows))
            changed += sum(a != b for a, b in zip(original[uid], new))
            total += len(rows)
            adapted += int(len(original[donor]) != len(rows))
            events.loc[rows, "button"] = new
        type(self).last_stats = {
            "training_sentences_relabelled": len(self.donor_of_uid),
            "training_keystrokes_relabelled": total,
            "keystroke_labels_changed": changed,
            "fraction_of_training_labels_changed": changed / max(total, 1),
            "sentences_needing_length_adaptation": adapted,
        }
        return events


# ---- provenance -------------------------------------------------------------

def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as error:  # pragma: no cover - descriptive only
        return f"unavailable: {error}"


def package_versions() -> dict:
    installed = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            installed[name.lower().replace("_", "-")] = dist.version
    locked = {}
    for line in LOCK_FILE.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            locked[name.lower().replace("_", "-")] = version.strip()
    key = ("torch", "lightning", "pytorch-lightning", "neuralset", "neuraltrain", "x-transformers", "numpy",
           "scikit-learn", "mne", "pandas", "torchmetrics", "exca", "levenshtein", "einops")
    return {
        "key_packages": {k: installed.get(k) for k in key},
        "lock_file": LOCK_FILE.relative_to(ROOT).as_posix(),
        "lock_mismatches": {k: {"locked": v, "installed": installed.get(k)} for k, v in sorted(locked.items())
                            if installed.get(k) != v},
        "all_installed": dict(sorted(installed.items())),
    }


def gpu_description() -> dict:
    info = {
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            info["devices"].append({"index": index, "name": props.name,
                                    "total_memory_gib": props.total_memory / 1024 ** 3,
                                    "capability": f"{props.major}.{props.minor}"})
    try:
        query = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=30)
        info["nvidia_smi"] = query.stdout.strip() or query.stderr.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        info["nvidia_smi"] = f"not available: {type(error).__name__}"
    return info


def run_source_hash() -> str:
    return hashlib.sha256(inspect.getsource(Experiment.run).encode("utf-8")).hexdigest()


def provenance() -> dict:
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_worktree_dirty": bool(git("status", "--porcelain")),
        "official_revision": h.OFFICIAL_REVISION,
        "official_source_clean": not git("status", "--porcelain", "--", "vendor/brain2qwerty"),
        "official_tree": git("rev-parse", "HEAD:vendor/brain2qwerty"),
        "experiment_run_source_sha256": run_source_hash(),
        "experiment_run_source_matches_audit": run_source_hash() == AUDITED_RUN_SOURCE_SHA256,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(),
        "gpu": gpu_description(),
    }


def verify_dataset(manifest_path: Path, hash_raw: bool = True) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = []
    for entry in manifest["files"]:
        if entry["kind"] == "raw" and not hash_raw:
            checks.append({**entry, "verified": None, "note": "raw hashing skipped by flag"})
            continue
        path = ROOT / entry["path"]
        actual = c.sha256_file(path) if path.exists() else None
        checks.append({"path": entry["path"], "kind": entry["kind"], "sha256": entry["sha256"],
                       "actual_sha256": actual, "verified": actual == entry["sha256"]})
    return {
        "dataset_manifest": manifest_path.relative_to(ROOT).as_posix(),
        "dataset_manifest_sha256": c.sha256_file(manifest_path),
        "files": checks,
        "all_verified": all(row["verified"] in (True, None) for row in checks),
        "expected_counts": manifest["counts"],
    }


# ---- experiment construction ----------------------------------------------

def build_experiment(split: dict, seed: int, output_dir: Path, control: dict | None = None,
                     devices: int | None = None, local_dry_run: bool = False) -> tuple[dict, Experiment, list[str]]:
    """The official config with only the splitter replaced (and the control relabel appended)."""
    cfg = experiment_config()
    deviations = []
    cfg["seed"] = int(seed)
    cfg["output_dir"] = str(output_dir)
    if devices is not None:
        cfg["devices"] = int(devices)
        deviations.append(f"devices set to {devices} (official default: Experiment.devices = 8, capped by GPUs present)")
    names = [t["name"] for t in cfg["data"]["transforms"]]
    if names != ["SpanishBCBLPreprocessing", "Brain2QwertyV1Splitter"]:
        raise RuntimeError(f"Unexpected official transforms {names}")
    cfg["data"]["transforms"][1] = {
        "name": "Phase7ManifestSplit",
        "partitions": {p: [r["sentence_UID"] for r in split["partitions"][p]] for p in c.PARTITIONS},
    }
    if control is not None:
        cfg["data"]["transforms"].append({"name": "Phase7TrainTargetDerangement",
                                          "donor_of_uid": control["donor_of_uid"]})
    if os.name == "nt":
        cfg["data"]["study"].pop("infra", None)
        cfg["data"]["study"].pop("infra_timelines", None)
        deviations.append("Windows: official study event cache disabled (exca writes JSON in filenames, illegal on "
                          "Windows); events verified identical to the stored extraction")
    if local_dry_run:
        cfg["data"].update(num_workers=0, persistent_workers=False, pin_memory=False)
        deviations.append("local dry run: dataloader workers 0, no pinned memory (throughput settings only)")
    return cfg, Experiment(**cfg), deviations


def build_loaders(xp: Experiment) -> dict:
    """First half of Experiment.run: seed, then the official Data.build."""
    pl.seed_everything(xp.seed, workers=True)
    Path(xp.output_dir).mkdir(parents=True, exist_ok=True)
    return xp.data.build()


def build_module(xp: Experiment, loaders: dict):
    """Experiment.run's module construction, verbatim."""
    brain, transformer = xp._build_modules(loaders["train"])
    module = h.BrainModule(model=brain, transformer=transformer, loss=xp.loss.build(),
                           metrics={"CER": h.CER()}, optimizer=xp.optimizer)
    h.materialize_lazy_params(module, loaders["train"])
    return module


def loader_audit(loaders: dict, split: dict) -> dict:
    out = {}
    for part in c.PARTITIONS:
        segments = loaders[part].dataset.segments
        uids = {str(s.trigger.extra["sentence_UID"]) for s in segments}
        expected = split["partitions"][part]
        out[part] = {
            "keystroke_segments": len(segments),
            "expected_keystrokes": int(sum(r["keystrokes"] for r in expected)),
            "sentences": len(uids),
            "expected_sentences": len(expected),
            "matches_manifest": len(segments) == sum(r["keystrokes"] for r in expected)
            and uids == {r["sentence_UID"] for r in expected},
            "batch_size": loaders[part].batch_size,
            "batches": len(loaders[part]),
        }
    return out


# ---- evaluation ---------------------------------------------------------------

def sentence_order(segments: list) -> list[list[int]]:
    """Sentence groups in segment order, as SentenceGroupedDistributedSampler builds them (one process)."""
    groups: dict[str, list[int]] = {}
    for index, segment in enumerate(segments):
        groups.setdefault(str(segment.trigger.extra.get("sentence_UID", index)), []).append(index)
    return list(groups.values())


def neuro_ablation(condition: str, n_channels: int, n_times: int):
    """Fixed test-time input ablations (seed 2026); 'original' returns None."""
    rng = np.random.default_rng(ABLATION_SEED)

    def derangement(n: int) -> np.ndarray:
        while True:
            perm = rng.permutation(n)
            if not (perm == np.arange(n)).any():
                return perm

    if condition == "original":
        return None
    if condition == "zero_meg":
        return lambda x: torch.zeros_like(x)
    if condition == "temporal_permutation":
        perm = torch.as_tensor(derangement(n_times))
        return lambda x: x[:, :, perm.to(x.device)]
    if condition == "channel_permutation":
        perm = torch.as_tensor(derangement(n_channels))
        return lambda x: x[:, perm.to(x.device), :]
    raise ValueError(condition)


@torch.no_grad()
def evaluate(module, dataset, batch_size: int, ablation=None, max_sentences: int | None = None) -> dict:
    """Single-process scoring in official test batching (sentence order, fixed batch size)."""
    module.eval()
    device = next(module.parameters()).device
    groups = sentence_order(dataset.segments)
    if max_sentences is not None:
        groups = groups[:max_sentences]
    order = [i for group in groups for i in group]
    predicted, target = {}, {}
    started = time.time()
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        batch = dataset.collate_fn([dataset[i] for i in indices]).to(device)
        if ablation is not None:
            batch.data["neuro"] = ablation(batch.data["neuro"])
        logits = module._transformer_forward(batch, module.forward(batch))
        for i, p, t in zip(indices, logits.argmax(dim=1).cpu().tolist(),
                           batch.data["feature"].squeeze(1).cpu().tolist()):
            predicted[i], target[i] = int(p), int(t)
    sentences = []
    for group in groups:
        group = sorted(group, key=lambda i: float(dataset.segments[i].trigger.start))
        sentences.append({"sentence_UID": str(dataset.segments[group[0]].trigger.extra["sentence_UID"]),
                          "target": [target[i] for i in group], "predicted": [predicted[i] for i in group]})
    result = c.metrics(sentences, h.CHAR_INDEX, h.NUM_CLASSES)
    result["seconds"] = time.time() - started
    result["keystroke_order"] = order
    result["predicted_in_order"] = [predicted[i] for i in order]
    return result


def run_official(xp: Experiment, loaders: dict, module) -> tuple[object, list]:
    """Second half of Experiment.run (training branch), verbatim."""
    trainer = xp._trainer_setup()
    trainer.fit(module, loaders["train"], loaders["val"])
    official_test = trainer.test(module, dataloaders=loaders["test"]) if "test" in loaders else []
    return trainer, official_test


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)
