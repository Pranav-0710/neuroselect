"""Phase 7 shared, dependency-light logic (numpy only; no official imports).

Used by the official-v1 GPU harness (pinned/GPU environment) and by the Phase 7
tests and reports (workspace environment):

* reading the formal split manifests into partitions and checking them;
* the matched no-valid-signal control: a deterministic stimulus-level
  derangement of training targets (seed 2026) and the per-keystroke label
  adaptation it needs;
* the Phase 7 metric set for one-character-per-keystroke decoders.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARTITIONS = ("train", "val", "test")
CONTROL_SEED = 2026
CONTROL_WINDOW = 8
SPACE_TOKEN = "<space>"


# ---- split manifests -------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(paths: list[Path]) -> dict:
    """Partitions from either the three official_v1_clean_<p>.json files or one s22_<X>_clean.json."""
    partitions: dict[str, list[dict]] = {}
    names = set()
    for path in paths:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        names.add(manifest["split_name"])
        if "partition" in manifest:
            partitions[manifest["partition"]] = manifest["records"]
        else:
            for name, part in manifest["partitions"].items():
                partitions[name] = part["records"]
    if len(names) != 1:
        raise ValueError(f"Split manifests from different splits: {sorted(names)}")
    missing = [p for p in PARTITIONS if p not in partitions]
    if missing:
        raise ValueError(f"Split is missing partitions {missing}")
    return {
        "split_name": names.pop(),
        "partitions": partitions,
        "manifest_files": {Path(p).as_posix(): sha256_file(Path(p)) for p in paths},
        "partition_uid_sha256": {
            p: hashlib.sha256("\n".join(sorted(r["sentence_UID"] for r in partitions[p])).encode()).hexdigest()
            for p in PARTITIONS},
    }


def check_split(partitions: dict[str, list[dict]]) -> dict:
    """Disjointness of UIDs, sentence groups, TF-IDF clusters and texts across partitions."""
    fields = ("sentence_UID", "unique_sentence_group_id", "tfidf_cluster_id", "sentence_presented", "sentence_typed")
    result = {}
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1:]:
            result[f"{a}_vs_{b}"] = {
                field: len({r[field] for r in partitions[a]} & {r[field] for r in partitions[b]}) for field in fields}
    result["disjoint"] = all(v == 0 for pair in result.values() if isinstance(pair, dict) for v in pair.values())
    return result


# ---- matched no-valid-signal control ----------------------------------------

def stimulus_derangement(train_records: list[dict], seed: int = CONTROL_SEED, window: int = CONTROL_WINDOW) -> dict:
    """Deterministic stimulus-level derangement of the training targets.

    Each training sentence group g receives the target text of another training
    group sigma(g): never itself, never a group in the same official TF-IDF
    cluster (so no paraphrase supplies a near-correct target). Groups are ordered
    by presented-text length (ties broken by a seeded key) and deranged within
    consecutive windows of ``window`` groups, so donor and recipient lengths stay
    close and few per-keystroke labels need adaptation. Both occurrences of a
    group take the donor group's occurrence from the same session when the
    training side has one. Depends only on the training records and the seed,
    never on the model seed.
    """
    rng = np.random.default_rng(seed)
    groups: dict[str, dict] = {}
    for record in train_records:
        entry = groups.setdefault(record["unique_sentence_group_id"], {
            "text": record["sentence_presented"], "cluster": record["tfidf_cluster_id"], "records": []})
        entry["records"].append(record)
    ids = sorted(groups)
    if len(ids) < 2:
        raise ValueError("A derangement needs at least two training groups")
    tie = dict(zip(ids, rng.random(len(ids))))
    ordered = sorted(ids, key=lambda g: (len(groups[g]["text"]), tie[g]))
    windows = [ordered[i:i + window] for i in range(0, len(ordered), window)]
    if len(windows) > 1 and len(windows[-1]) < max(2, window // 2):
        windows[-2].extend(windows.pop())

    sigma, attempts = {}, {}
    for index, members in enumerate(windows):
        for attempt in range(1, 100_001):
            permuted = [members[i] for i in rng.permutation(len(members))]
            if all(src != dst and groups[src]["cluster"] != groups[dst]["cluster"]
                   for src, dst in zip(members, permuted)):
                sigma.update(zip(members, permuted))
                attempts[index] = attempt
                break
        else:
            raise RuntimeError(f"No valid derangement for window {index}: {members}")

    donor_of_uid = {}
    for g, entry in groups.items():
        donors = sorted(groups[sigma[g]]["records"], key=lambda r: r["sentence_UID"])
        for record in entry["records"]:
            same_session = [d for d in donors if d["session"] == record["session"]]
            donor_of_uid[record["sentence_UID"]] = (same_session or donors)[0]["sentence_UID"]

    by_uid = {r["sentence_UID"]: r for r in train_records}
    length_gaps = [by_uid[d]["keystrokes"] - by_uid[r]["keystrokes"] for r, d in donor_of_uid.items()]
    return {
        "seed": seed,
        "window": window,
        "group_map": {g: sigma[g] for g in ids},
        "donor_of_uid": donor_of_uid,
        "windows": len(windows),
        "attempts_per_window": attempts,
        "fixed_points": sum(1 for g in ids if sigma[g] == g),
        "same_cluster_pairs": sum(1 for g in ids if groups[g]["cluster"] == groups[sigma[g]]["cluster"]),
        "identical_target_text_pairs": sum(1 for r, d in donor_of_uid.items()
                                           if by_uid[r]["sentence_typed"] == by_uid[d]["sentence_typed"]),
        "keystroke_length_gap": {"mean_abs": float(np.mean(np.abs(length_gaps))), "max_abs": int(np.max(np.abs(length_gaps))),
                                 "exact_length_matches": int(sum(1 for gap in length_gaps if gap == 0))},
    }


def adapt_labels(donor_tokens: list[str], length: int) -> list[str]:
    """Donor button tokens fitted to the recipient's keystroke count.

    One label per keystroke is fixed by the official formulation, so a longer
    donor is truncated and a shorter donor is continued by repeating it after a
    space token (``donor <space> donor ...``).
    """
    if length <= len(donor_tokens):
        return list(donor_tokens[:length])
    out: list[str] = []
    while len(out) < length:
        if out:
            out.append(SPACE_TOKEN)
        out.extend(donor_tokens)
    return out[:length]


# ---- metrics -------------------------------------------------------------

def _load_aggregator():
    spec = importlib.util.spec_from_file_location("s22_expanded_ctc_aggregate",
                                                  ROOT / "scripts/s22_expanded_ctc_aggregate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(sentences: list[dict], char_index: dict[int, str], num_classes: int) -> dict:
    """Phase 7 metric set for a one-character-per-keystroke decoder.

    ``sentences``: [{"sentence_UID", "target": [label...], "predicted": [label...]}]
    with labels in keystroke-time order. Decoding is the official one: each label
    maps through CHAR_INDEX (unknown labels decode to the empty string).
    """
    from neuroselect.metrics import cer, wer

    aggregator = _load_aggregator()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    totals = dict.fromkeys(("substitutions", "deletions", "insertions", "hits",
                            "reference_characters", "hypothesis_characters"), 0)
    rows = []
    for sentence in sentences:
        target = np.asarray(sentence["target"], dtype=np.int64)
        predicted = np.asarray(sentence["predicted"], dtype=np.int64)
        if target.shape != predicted.shape:
            raise ValueError("One prediction per keystroke is required")
        np.add.at(confusion, (target, predicted), 1)
        target_text = "".join(char_index.get(int(t), "") for t in target)
        decoded = "".join(char_index.get(int(p), "") for p in predicted)
        s, d, i, hits = aggregator.edit_components(target_text, decoded)
        for key, value in (("substitutions", s), ("deletions", d), ("insertions", i), ("hits", hits),
                           ("reference_characters", len(target_text)), ("hypothesis_characters", len(decoded))):
            totals[key] += value
        rows.append({
            "sentence_UID": sentence["sentence_UID"],
            "keystrokes": int(len(target)),
            "target": target_text,
            "decoded": decoded,
            "keystroke_accuracy": float((target == predicted).mean()),
            "cer": cer(target_text, decoded),
            "wer": wer(target_text, decoded),
            "target_length": len(target_text),
            "predicted_length": len(decoded),
        })
    keystrokes = int(confusion.sum())
    support = confusion.sum(axis=1)
    return {
        "sentences": len(rows),
        "keystrokes": keystrokes,
        "keystroke_accuracy": float(np.trace(confusion) / max(keystrokes, 1)),
        "mean_sentence_cer": float(np.mean([r["cer"] for r in rows])),
        "mean_sentence_wer": float(np.mean([r["wer"] for r in rows])),
        "counts": totals,
        **aggregator.rates(totals),
        "target_sequence_length_characters": totals["reference_characters"],
        "predicted_sequence_length_characters": totals["hypothesis_characters"],
        "target_sequence_length_keystrokes": keystrokes,
        "predicted_sequence_length_keystrokes": keystrokes,
        "decoding_note": ("one label per keystroke (oracle keystroke segmentation); characters can be fewer than "
                          "keystrokes only through labels absent from CHAR_INDEX"),
        "confusion_matrix": confusion.tolist(),
        "confusion_axes": "rows = target label, columns = predicted label (official label ids)",
        "per_character_recall": {
            char_index.get(label, f"<{label}>"): (float(confusion[label, label] / support[label]) if support[label] else None)
            for label in range(num_classes)},
        "per_character_support": {char_index.get(label, f"<{label}>"): int(support[label]) for label in range(num_classes)},
        "rows": rows,
    }


def agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of keystrokes with identical predicted labels."""
    a, b = np.asarray(a), np.asarray(b)
    return float((a == b).mean()) if a.size else float("nan")
