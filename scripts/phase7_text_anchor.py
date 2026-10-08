"""Phase 7 Task 12: length-matched training-sentence text anchor on the clean test partitions.

Same anchor as Phases 5Z and 6: every held-out sentence is scored against the
training-partition sentence closest in target character length (ties broken by
sentence UID). The held-out text is never used and no model is involved.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from neuroselect.metrics import cer, wer

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("phase7_common", ROOT / "scripts/phase7_common.py")
c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c)

OUT = ROOT / "results/phase7_text_anchor.json"
SPLITS = {
    "official_v1_clean": [ROOT / f"data/manifests/official_v1_clean_{p}.json" for p in c.PARTITIONS],
    "D_clean": [ROOT / "data/manifests/s22_D_clean.json"],
    "E_clean": [ROOT / "data/manifests/s22_E_clean.json"],
}


def main() -> None:
    aggregator = c._load_aggregator()
    results = {}
    for name, files in SPLITS.items():
        split = c.load_split(files)
        train, test = split["partitions"]["train"], split["partitions"]["test"]
        pool = sorted((r["sentence_UID"], r["sentence_typed"]) for r in train)
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
                         "target": target, "anchor": hypothesis, "cer": cer(target, hypothesis),
                         "wer": wer(target, hypothesis)})
        results[name] = {
            "train_sentences_in_pool": len(train),
            "test_sentences": len(test),
            "test_keystrokes": int(sum(r["keystrokes"] for r in test)),
            "test_text_ever_used": False,
            "mean_sentence_cer": float(np.mean([r["cer"] for r in rows])),
            "mean_sentence_wer": float(np.mean([r["wer"] for r in rows])),
            "counts": totals,
            **aggregator.rates(totals),
            "rows": rows,
        }
    OUT.write_text(json.dumps({
        "task": "Phase 7 Task 12: text-only anchor on the clean test partitions",
        "what_it_is": ("Each held-out sentence scored against the training-partition sentence closest in target "
                       "character length (ties by sentence UID). No model; the held-out text is never used. A "
                       "descriptive text-prior reference."),
        "comparability_note": ("The anchor emits a whole training sentence, so its length can differ from the target; "
                               "official v1 emits exactly one label per keystroke. Compare character F1, and read CER "
                               "with the length ratio."),
        "splits": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({n: {"f1": round(r["character_f1"], 4), "mean_cer": round(r["mean_sentence_cer"], 4),
                          "length_ratio": round(r["length_ratio"], 3), "n": r["test_sentences"]}
                      for n, r in results.items()}, indent=1))


if __name__ == "__main__":
    main()
