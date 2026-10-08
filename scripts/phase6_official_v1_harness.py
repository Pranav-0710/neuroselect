"""Phase 6 harness: the official Brain2Qwerty v1 model on the validated S22 tensors.

Model-side everything is official and unmodified: the configuration comes from
``brain2qwerty_v1.config.xp_config``; the encoder and sentence transformer are
built from it by neuraltrain; targets, subject ids and channel positions come
from the official extractors; batches follow the official sampler; training and
test steps are ``BrainModule``'s own.

Only the neural input is sourced differently: rows of the validated official-v1
event-tensor slab, which equal the official ``MegExtractor`` output bit for bit
(``results/official_v1_boundary_event_parity.json``,
``results/s22_event_tensor_env_parity.json``). The Phase 6 smoke test re-checks
that against batches built by the official ``SegmentDataset``.

Run with the pinned environment plus the Phase 6 overlay, which supplies the
official-lock packages the pinned environment lacks (x-transformers 2.4.9,
einops, einx, loguru, frozendict, Levenshtein, RapidFuzz):

    PYTHONPATH="<overlay>/site;vendor/brain2qwerty;src" <pinned-env>/python script.py
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import re
from ctypes import wintypes
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/raw/spanishbcbl_s22"
TENSOR_PATH = ROOT / "data/processed/s22_official_v1/events.npy"
INDEX_PATH = ROOT / "data/processed/s22_official_v1/index.jsonl"
MANIFEST_DIR = ROOT / "data/manifests"
METADATA_PATH = MANIFEST_DIR / "s22_sentence_block_metadata.jsonl"
STORED_EVENTS = DATA_ROOT / "events_clean_all_blocks.pkl"
OFFICIAL_CACHE = Path.home() / "Envs" / "phase6-official-cache"
OVERLAY = Path.home() / "Envs" / "phase6-official-overlay"

# Must be set before brain2qwerty_v1.config.xp_config is imported.
os.environ.setdefault("BRAIN2QWERTY_STUDIES", str(DATA_ROOT))
os.environ.setdefault("BRAIN2QWERTY_CACHE", str(OFFICIAL_CACHE))
os.environ.setdefault("BRAIN2QWERTY_RESULTS", str(ROOT / "results/runs/phase6/official_results"))

import neuralset as ns  # noqa: E402
import pandas as pd  # noqa: E402
import studies  # noqa: E402,F401  (registers Pinet2024Meg)
import torch  # noqa: E402
from neuralset.dataloader import Batch  # noqa: E402

from brain2qwerty_v1 import transforms as _official_transforms  # noqa: E402,F401
from brain2qwerty_v1.config.xp_config import experiment_config  # noqa: E402
from brain2qwerty_v1.main import Experiment  # noqa: E402
from brain2qwerty_v1.metrics import CER  # noqa: E402
from brain2qwerty_v1.pl_module import BrainModule  # noqa: E402
from brain2qwerty_v1.utils import (  # noqa: E402
    CHAR_INDEX,
    NUM_CLASSES,
    ChannelPositions2D,
    SentenceGroupedDistributedSampler,
    materialize_lazy_params,
)

OFFICIAL_REVISION = "5f9889621d0df391c5aab37c996683d308e6e926"
SPLIT_FILES = {
    "C": "split_C_sentence_disjoint.json",
    "D": "split_D_cross_session_sentence_disjoint.json",
    "E": "split_E_cross_session_sentence_disjoint.json",
}
TIMELINE_PATTERN = re.compile(r"session=(?P<session>[^,]+),.*task=(?P<task>[^,]+)")


# ---- configuration -------------------------------------------------------

def official_experiment() -> tuple[dict, Experiment]:
    """The official configuration, parsed by the official Experiment model."""
    cfg = experiment_config()
    return cfg, Experiment(**cfg)


# ---- events and extractors ------------------------------------------------

def build_events(xp: Experiment) -> pd.DataFrame:
    """The official ``Data.build_events`` path, minus the official splitter.

    The splitter is the second configured transform; Phase 6 assigns its own
    sentence-disjoint splits instead, so only SpanishBCBLPreprocessing runs.

    The study is built with the official name and path but without the official
    exca event cache: on Windows that cache writes files whose names embed the
    timeline as JSON (quotes and colons are illegal in Windows filenames). The
    cache changes speed, not events; ``check_against_stored_events`` verifies it.
    """
    official_study = experiment_config()["data"]["study"]
    study = ns.events.Study(name=official_study["name"], path=official_study["path"])
    events = study.run()
    preprocessing = [t for t in xp.data.transforms if type(t).__name__ == "SpanishBCBLPreprocessing"]
    if len(preprocessing) != 1:
        raise RuntimeError("Expected exactly one SpanishBCBLPreprocessing transform in the official config")
    events = preprocessing[0]._run(events)
    return ns.events.standardize_events(events)


def check_against_stored_events(events: pd.DataFrame) -> dict:
    """The rebuilt keystrokes must equal the stored 5X extraction exactly."""
    stored = pd.read_pickle(STORED_EVENTS)
    fresh = events[events["type"] == "Keystroke"]
    old = stored[stored["type"] == "Keystroke"]
    fresh_keys = sorted(zip(fresh["timeline"], fresh["start"].round(6), fresh["button"].astype(str)))
    old_keys = sorted(zip(old["timeline"], old["start"].round(6), old["button"].astype(str)))
    return {
        "rebuilt_keystrokes": len(fresh_keys),
        "stored_keystrokes": len(old_keys),
        "identical_timeline_start_button": fresh_keys == old_keys,
    }


def assign_split(events: pd.DataFrame, split: str) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((MANIFEST_DIR / SPLIT_FILES[split]).read_text(encoding="utf-8"))
    events = events.copy()
    uid = events["sentence_UID"].astype(str)
    events["split"] = None
    events.loc[uid.isin(manifest["train_sentence_UIDs"]), "split"] = "train"
    events.loc[uid.isin(manifest["test_sentence_UIDs"]), "split"] = "test"
    return events, manifest


def prepare_extractors(xp: Experiment, events: pd.DataFrame) -> dict:
    """Exactly the extractor setup of the official ``Data.build``."""
    neuro, feature = xp.data.neuro, xp.data.feature
    neuro.prepare(events)
    feature.prepare(events)
    subject_id = ns.extractors.LabelEncoder(event_types="Meg", event_field="subject")
    subject_id.prepare(events)
    channel_positions = ChannelPositions2D(neuro=neuro)
    channel_positions.prepare(events)
    return {"neuro": neuro, "feature": feature, "subject_id": subject_id,
            "channel_positions": channel_positions}


def keystroke_segments(xp: Experiment, events: pd.DataFrame, split: str) -> list:
    mask = (events.split == split) & (events.type == "Keystroke")
    return ns.segments.list_segments(events, mask, start=xp.data.start, duration=xp.data.duration)


# ---- slab-backed tensors -------------------------------------------------

def slab_index() -> dict[tuple[str, str, float], int]:
    lookup = {}
    for line in INDEX_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            lookup[(str(row["session"]), str(row["block"]), round(float(row["timestamp_seconds"]), 6))] = row["row"]
    return lookup


def segment_key(segment) -> tuple[str, str, float]:
    match = TIMELINE_PATTERN.search(str(segment.trigger.timeline))
    if match is None:
        raise ValueError(f"Unparseable timeline {segment.trigger.timeline!r}")
    return match["session"], match["task"], round(float(segment.trigger.start), 6)


class KeystrokeTensors:
    """All per-keystroke inputs for a list of official segments.

    ``neuro`` comes from the slab; ``feature``, ``subject_id`` and
    ``channel_positions`` are produced by the official extractors and collated
    by the official ``SegmentDataset``.
    """

    def __init__(self, segments: list, extractors: dict, slab: np.ndarray, lookup: dict):
        light = {name: extractors[name] for name in ("feature", "subject_id", "channel_positions")}
        dataset = ns.SegmentDataset(extractors=light, segments=segments, remove_incomplete_segments=True)
        if len(dataset.segments) != len(segments):
            raise RuntimeError(f"Official dataset dropped {len(segments) - len(dataset.segments)} segments")
        official = dataset.load_all()
        rows = [lookup[segment_key(segment)] for segment in segments]
        neuro = np.ascontiguousarray(np.asarray(slab[rows]).transpose(0, 2, 1))  # (N, 306, 25)
        self.segments = list(segments)
        self.rows = rows
        self.data = {"neuro": torch.from_numpy(neuro), **{k: v for k, v in official.data.items()}}

    def __len__(self) -> int:
        return len(self.segments)

    def batch(self, indices: list[int], neuro: torch.Tensor | None = None) -> Batch:
        index = torch.as_tensor(indices, dtype=torch.long)
        data = {name: tensor[index] for name, tensor in self.data.items()}
        if neuro is not None:
            data["neuro"] = neuro[index]
        return Batch(data=data, segments=[self.segments[i] for i in indices])

    def loader(self, batch_size: int, neuro: torch.Tensor | None = None) -> torch.utils.data.DataLoader:
        """Official batching: SentenceGroupedDistributedSampler (no shuffle), fixed batch size."""
        sampler = SentenceGroupedDistributedSampler(self.segments)
        return torch.utils.data.DataLoader(
            list(range(len(self))), batch_size=batch_size, sampler=sampler,
            collate_fn=lambda indices: self.batch(list(indices), neuro), num_workers=0,
        )


# ---- model ----------------------------------------------------------------

def build_module(xp: Experiment, n_in_channels: int) -> BrainModule:
    """Mirrors ``Experiment._build_modules`` + ``BrainModule`` construction."""
    hidden = xp.brain_model_config.hidden
    brain = xp.brain_model_config.build(n_in_channels=n_in_channels, n_outputs=hidden)
    transformer = xp.transformer_config.build(dim=hidden)
    return BrainModule(model=brain, transformer=transformer, loss=xp.loss.build(),
                       metrics={"CER": CER()}, optimizer=xp.optimizer)


def parameter_counts(module: BrainModule) -> dict:
    def count(part) -> int:
        # Lazy layers (Bahdanau attention) hold UninitializedParameter until the
        # official materialize_lazy_params dummy forward; count only real ones.
        return int(sum(p.numel() for p in part.parameters()
                       if not isinstance(p, torch.nn.parameter.UninitializedParameter)))
    model = module.model
    uninitialized = sum(1 for p in module.parameters()
                        if isinstance(p, torch.nn.parameter.UninitializedParameter))
    return {
        "uninitialized_lazy_parameters": uninitialized,
        "channel_merger": count(model.merger) if model.merger is not None else 0,
        "channel_merger_heads_shape": list(model.merger.heads.shape) if model.merger is not None else None,
        "initial_linear": count(model.initial_linear) if model.initial_linear is not None else 0,
        "subject_layers": count(model.subject_layers) if model.subject_layers is not None else 0,
        "subject_layers_weight_shape": (list(model.subject_layers.weights.shape)
                                        if model.subject_layers is not None
                                        and hasattr(model.subject_layers, "weights") else None),
        "conv_encoder": count(model.encoder),
        "time_aggregation": count(model.time_agg_out) if model.time_agg_out is not None else 0,
        "sentence_transformer": count(module.transformer),
        "output_linear": count(module.linear),
        "total": count(module),
    }


# ---- memory (Windows; the pinned environment has no psutil) ---------------

class _Counters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def memory() -> dict:
    """Process working set / private commit and system available memory, in GB."""
    counters = _Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    gb = 1024 ** 3
    return {
        "process_working_set_gb": counters.WorkingSetSize / gb,
        "process_peak_working_set_gb": counters.PeakWorkingSetSize / gb,
        "process_private_commit_gb": counters.PagefileUsage / gb,
        "process_peak_private_commit_gb": counters.PeakPagefileUsage / gb,
        "system_available_physical_gb": status.ullAvailPhys / gb,
        "system_total_physical_gb": status.ullTotalPhys / gb,
        "system_available_commit_gb": status.ullAvailPageFile / gb,
        "system_commit_limit_gb": status.ullTotalPageFile / gb,
    }


# ---- decoding and metrics -------------------------------------------------

def load_aggregator():
    """Reuse the 5Z alignment so every character F1 in the project is computed one way."""
    spec = importlib.util.spec_from_file_location("s22_expanded_ctc_aggregate",
                                                  ROOT / "scripts/s22_expanded_ctc_aggregate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def labels_to_text(labels) -> str:
    return "".join(CHAR_INDEX.get(int(label), "") for label in labels)


@torch.no_grad()
def predict(module: BrainModule, tensors: KeystrokeTensors, batch_size: int,
            neuro: torch.Tensor | None = None) -> dict:
    """Per-keystroke predictions in official test batching, keyed by tensor index."""
    module.eval()
    predicted = np.empty(len(tensors), dtype=np.int64)
    losses = np.empty(len(tensors), dtype=np.float64)
    order = list(SentenceGroupedDistributedSampler(tensors.segments))
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        batch = tensors.batch(indices, neuro)
        logits = module._transformer_forward(batch, module.forward(batch))
        target = batch.data["feature"].squeeze(1)
        predicted[indices] = logits.argmax(dim=1).numpy()
        losses[indices] = torch.nn.functional.cross_entropy(logits, target, reduction="none").numpy()
    return {"predicted": predicted, "loss": losses}


def sentence_rows(tensors: KeystrokeTensors, predicted: np.ndarray) -> list[dict]:
    """Group keystroke predictions into sentences in keystroke-time order."""
    from neuroselect.metrics import cer, wer

    aggregator = load_aggregator()
    targets = tensors.data["feature"].squeeze(1).numpy()
    by_sentence: dict[str, list[int]] = {}
    for index, segment in enumerate(tensors.segments):
        by_sentence.setdefault(str(segment.trigger.extra["sentence_UID"]), []).append(index)
    rows = []
    for uid, indices in by_sentence.items():
        indices = sorted(indices, key=lambda i: float(tensors.segments[i].trigger.start))
        target = labels_to_text(targets[indices])
        decoded = labels_to_text(predicted[indices])
        s, d, ins, hits = aggregator.edit_components(target, decoded)
        session, task, _ = segment_key(tensors.segments[indices[0]])
        rows.append({
            "sentence_UID": uid,
            "block": f"session{session}/{task}",
            "keystrokes": len(indices),
            "target": target,
            "decoded": decoded,
            "keystroke_accuracy": float(np.mean(targets[indices] == predicted[indices])),
            "cer": cer(target, decoded),
            "wer": wer(target, decoded),
            "substitutions": s, "deletions": d, "insertions": ins, "hits": hits,
            "target_length": len(target),
            "decoded_length": len(decoded),
            "decoded_target_ratio": len(decoded) / max(len(target), 1),
        })
    return rows


def summarize(rows: list[dict], aggregator=None) -> dict:
    aggregator = aggregator or load_aggregator()
    totals = {key: int(sum(row[key] for row in rows)) for key in ("substitutions", "deletions", "insertions", "hits")}
    totals["reference_characters"] = int(sum(row["target_length"] for row in rows))
    totals["hypothesis_characters"] = int(sum(row["decoded_length"] for row in rows))
    keystrokes = sum(row["keystrokes"] for row in rows)
    return {
        "sentences": len(rows),
        "keystrokes": keystrokes,
        "keystroke_accuracy": float(sum(row["keystroke_accuracy"] * row["keystrokes"] for row in rows) / max(keystrokes, 1)),
        "mean_sentence_cer": float(np.mean([row["cer"] for row in rows])),
        "mean_sentence_wer": float(np.mean([row["wer"] for row in rows])),
        "mean_decoded_target_ratio": float(np.mean([row["decoded_target_ratio"] for row in rows])),
        "counts": totals,
        **aggregator.rates(totals),
    }
