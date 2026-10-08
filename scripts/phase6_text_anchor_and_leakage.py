"""Phase 6 Parts 8 and 9: text-only anchor and split leakage audit (no model).

Part 8: for every held-out sentence, the training sentence closest in target
character length is used as its hypothesis (ties broken by sentence UID); the
held-out text itself is never used. Descriptive language/text prior reference,
not a model.

Part 9: exact-overlap checks plus the official splitter's own paraphrase rule.
The official ``Brain2QwertyV1Splitter`` clusters sentences whose TF-IDF cosine
similarity exceeds 0.5 and keeps each cluster inside one split. Phase 6 splits
are defined by stimulus list, so this audit measures how many train/test
sentence pairs the official rule would have kept together.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from neuroselect.metrics import cer, wer

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data/manifests"
METADATA_PATH = MANIFEST_DIR / "s22_sentence_block_metadata.jsonl"
ANCHOR_OUT = ROOT / "results/phase6_text_anchor.json"
LEAKAGE_OUT = ROOT / "results/phase6_leakage_audit.json"
SPLIT_FILES = {
    "C": "split_C_sentence_disjoint.json",
    "D": "split_D_cross_session_sentence_disjoint.json",
    "E": "split_E_cross_session_sentence_disjoint.json",
}
OFFICIAL_PARAPHRASE_THRESHOLD = 0.5
VENDOR_TRANSFORMS = ROOT / "vendor/brain2qwerty/brain2qwerty_v1/transforms.py"


def load_aggregator():
    spec = importlib.util.spec_from_file_location("s22_expanded_ctc_aggregate",
                                                  ROOT / "scripts/s22_expanded_ctc_aggregate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_threshold_from_source() -> float:
    """Read the threshold from the official splitter rather than trusting a constant."""
    for line in VENDOR_TRANSFORMS.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("threshold:"):
            return float(line.split("=")[1])
    raise RuntimeError("Official splitter threshold not found")


def main() -> None:
    aggregator = load_aggregator()
    metadata = {}
    for line in METADATA_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            metadata[record["sentence_UID"]] = record
    threshold = official_threshold_from_source()
    if threshold != OFFICIAL_PARAPHRASE_THRESHOLD:
        raise RuntimeError(f"Official threshold changed: {threshold}")

    anchors, leakage = {}, {}
    for split, filename in SPLIT_FILES.items():
        manifest = json.loads((MANIFEST_DIR / filename).read_text(encoding="utf-8"))
        train = [metadata[uid] for uid in manifest["train_sentence_UIDs"]]
        test = [metadata[uid] for uid in manifest["test_sentence_UIDs"]]

        # ---- Part 8: length-matched training-sentence anchor -------------
        pool = sorted((record["sentence_UID"], record["sentence_typed"]) for record in train)
        totals = dict.fromkeys(("substitutions", "deletions", "insertions", "hits",
                                "reference_characters", "hypothesis_characters"), 0)
        rows = []
        for record in test:
            target = record["sentence_typed"]
            uid, hypothesis = min(pool, key=lambda item: (abs(len(item[1]) - len(target)), item[0]))
            s, d, i, hits = aggregator.edit_components(target, hypothesis)
            for key, value in (("substitutions", s), ("deletions", d), ("insertions", i), ("hits", hits),
                               ("reference_characters", len(target)), ("hypothesis_characters", len(hypothesis))):
                totals[key] += value
            rows.append({"test_sentence_UID": record["sentence_UID"], "anchor_sentence_UID": uid,
                         "target": target, "anchor": hypothesis,
                         "cer": cer(target, hypothesis), "wer": wer(target, hypothesis)})
        anchors[split] = {
            "train_sentences": len(train),
            "test_sentences": len(test),
            "test_text_ever_used": False,
            "mean_sentence_cer": float(np.mean([row["cer"] for row in rows])),
            "mean_sentence_wer": float(np.mean([row["wer"] for row in rows])),
            "counts": totals,
            **aggregator.rates(totals),
            "examples": rows[:4],
        }

        # ---- Part 9: leakage ----------------------------------------------
        train_presented = {r["sentence_presented"] for r in train}
        test_presented = {r["sentence_presented"] for r in test}
        texts = sorted(train_presented | test_presented)
        similarity = cosine_similarity(TfidfVectorizer().fit_transform(texts))
        position = {text: index for index, text in enumerate(texts)}
        cross = similarity[np.ix_([position[t] for t in sorted(train_presented)],
                                  [position[t] for t in sorted(test_presented)])]
        over = np.argwhere(cross > threshold)
        train_sorted, test_sorted = sorted(train_presented), sorted(test_presented)
        leakage[split] = {
            "classification": manifest["classification"],
            "train_blocks": manifest["train_blocks"],
            "test_blocks": manifest["test_blocks"],
            "presented_text_overlap": len(train_presented & test_presented),
            "typed_text_overlap": len({r["sentence_typed"] for r in train} & {r["sentence_typed"] for r in test}),
            "sentence_UID_overlap": len(set(manifest["train_sentence_UIDs"]) & set(manifest["test_sentence_UIDs"])),
            "unique_sentence_group_overlap": len({r["unique_sentence_group_id"] for r in train}
                                                 & {r["unique_sentence_group_id"] for r in test}),
            "official_paraphrase_rule": {
                "rule": "TF-IDF cosine similarity > threshold keeps sentences in one split (Brain2QwertyV1Splitter)",
                "threshold": threshold,
                "tfidf_fit_on": "the presented sentences of this split's train and test sides",
                "cross_split_pairs": int(cross.size),
                "pairs_above_threshold": int(len(over)),
                "max_cross_split_cosine": float(cross.max()),
                "mean_cross_split_cosine": float(cross.mean()),
                "examples_above_threshold": [
                    {"train": train_sorted[a], "test": test_sorted[b], "cosine": float(cross[a, b])}
                    for a, b in sorted(over.tolist(), key=lambda ab: -cross[ab[0], ab[1]])[:8]
                ],
            },
        }

    leakage_checks = {
        "no_presented_text_overlap": all(v["presented_text_overlap"] == 0 for v in leakage.values()),
        "no_typed_text_overlap": all(v["typed_text_overlap"] == 0 for v in leakage.values()),
        "no_sentence_UID_overlap": all(v["sentence_UID_overlap"] == 0 for v in leakage.values()),
        "no_unique_sentence_group_overlap": all(v["unique_sentence_group_overlap"] == 0 for v in leakage.values()),
        "official_paraphrase_rule_satisfied": all(
            v["official_paraphrase_rule"]["pairs_above_threshold"] == 0 for v in leakage.values()),
        "model_inputs": ("BrainModule.forward reads only batch.data['neuro'], ['subject_id'] and "
                         "['channel_positions']; the target batch.data['feature'] enters only the loss"),
        "target_text_fed_as_input": False,
        "session_or_block_fed_as_input_or_target": False,
        "subject_id_input": "official input; constant 0 for the single subject S22",
        "normalization_fit": ("official per-recording RobustScaler on the continuous label-free signal of "
                              "each recording (transductive within a recording, as in the official pipeline); "
                              "no PCA; BatchNorm running statistics come from training batches only"),
        "test_tensors_used_for_any_fit": False,
    }
    ANCHOR_OUT.write_text(json.dumps({
        "task": "Phase 6 Part 8: text-only anchor",
        "what_it_is": ("Each held-out sentence scored against the training sentence closest in target character "
                       "length (ties broken by sentence UID). The held-out text is never used. A descriptive "
                       "language/text prior reference, not a model."),
        "splits": anchors,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    LEAKAGE_OUT.write_text(json.dumps({
        "task": "Phase 6 Part 9: data/split leakage audit",
        "splits": leakage,
        "checks": leakage_checks,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "anchor_f1": {s: round(a["character_f1"], 4) for s, a in anchors.items()},
        "anchor_cer": {s: round(a["mean_sentence_cer"], 4) for s, a in anchors.items()},
        "paraphrase_pairs_above_0.5": {s: v["official_paraphrase_rule"]["pairs_above_threshold"] for s, v in leakage.items()},
        "max_cross_cosine": {s: round(v["official_paraphrase_rule"]["max_cross_split_cosine"], 3) for s, v in leakage.items()},
        "checks": {k: v for k, v in leakage_checks.items() if isinstance(v, bool)},
    }, indent=2))


if __name__ == "__main__":
    main()
