"""5Y: canonical S22 sentence metadata and split-ladder design (metadata only).

Reads the official cleaned events, writes a pandas-independent sentence
manifest, derives list membership from exact sentence text (not filenames),
defines splits A-G, and audits them for leakage. No preprocessing, no
training, no model of any kind.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = ROOT / "data/raw/spanishbcbl_s22/events_clean_all_blocks.pkl"
METADATA_PATH = ROOT / "data/manifests/s22_sentence_block_metadata.jsonl"
MANIFEST_DIR = ROOT / "data/manifests"
OUT_PATH = ROOT / "results/s22_split_ladder_audit.json"
FIGURE_PATH = ROOT / "results/figures/debug/s22_split_ladder.png"

# Block identity as recorded in the official events (session, task).
BLOCKS = [("1", "block1"), ("1", "block2"), ("2", "block1"), ("2", "block2")]
# Log filenames, used ONLY to cross-check the text-derived list assignment.
LOG_FILENAME_LISTS = {
    ("1", "block1"): "list1", ("1", "block2"): "list2",
    ("2", "block1"): "list2", ("2", "block2"): "list1",
}

SPLITS = {
    "A_same_session": {
        "file": "split_A_same_session.json",
        "purpose": "Same-session block transfer with sentence-disjoint test data (session 1).",
        "train": [("1", "block1")], "test": [("1", "block2")],
        "sentence_disjoint_claim": True,
    },
    "B_same_session": {
        "file": "split_B_same_session.json",
        "purpose": "Same-session block transfer with sentence-disjoint test data (session 2).",
        "train": [("2", "block2")], "test": [("2", "block1")],
        "sentence_disjoint_claim": True,
    },
    "C_sentence_disjoint": {
        "file": "split_C_sentence_disjoint.json",
        "purpose": "Sentence-disjoint generalization pooled across all available sessions and blocks.",
        "train": [("1", "block1"), ("2", "block2")], "test": [("1", "block2"), ("2", "block1")],
        "sentence_disjoint_claim": True,
    },
    "D_cross_session_sentence_disjoint": {
        "file": "split_D_cross_session_sentence_disjoint.json",
        "purpose": "Cross-session and sentence-disjoint transfer (session 1 list1 -> session 2 list2).",
        "train": [("1", "block1")], "test": [("2", "block1")],
        "sentence_disjoint_claim": True,
    },
    "E_cross_session_sentence_disjoint": {
        "file": "split_E_cross_session_sentence_disjoint.json",
        "purpose": "Cross-session and sentence-disjoint transfer in the opposite direction (session 1 list2 -> session 2 list1).",
        "train": [("1", "block2")], "test": [("2", "block2")],
        "sentence_disjoint_claim": True,
    },
    "F_cross_session_sentence_overlap": {
        "file": "split_F_cross_session_sentence_overlap.json",
        "purpose": "CROSS-SESSION WITH SENTENCE OVERLAP (list1 session 1 -> list1 session 2). Adaptation only; never a sentence-generalization benchmark.",
        "train": [("1", "block1")], "test": [("2", "block2")],
        "sentence_disjoint_claim": False,
    },
    "G_cross_session_sentence_overlap": {
        "file": "split_G_cross_session_sentence_overlap.json",
        "purpose": "CROSS-SESSION WITH SENTENCE OVERLAP (list2 session 1 -> list2 session 2). Adaptation only; never a sentence-generalization benchmark.",
        "train": [("1", "block2")], "test": [("2", "block1")],
        "sentence_disjoint_claim": False,
    },
}


def block_label(session: str, task: str) -> str:
    return f"session{session}/{task}"


def describe(values: list[float]) -> dict:
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main() -> None:
    events = pd.read_pickle(EVENTS_PATH)
    sentences = events[events["type"] == "Sentence"].copy()
    if "is_percep" in sentences.columns:
        sentences = sentences[sentences["is_percep"] == False]  # noqa: E712
    keystrokes = events[events["type"] == "Keystroke"]
    all_words = events[events["type"] == "Word"]
    # Word events include the RSVP presentation words; only is_percep == False
    # words were actually typed. Keystroke events are production-only already.
    words = all_words[all_words["is_percep"] == False]  # noqa: E712
    perceptual_words = all_words[all_words["is_percep"] == True]  # noqa: E712

    # Group identity uses the exact presented stimulus text. Typed text is NOT
    # usable as a group key: typing errors differ between sessions.
    presented_texts = sorted(set(sentences["text"].astype(str)))
    group_ids = {text: f"G{index:03d}" for index, text in enumerate(presented_texts)}

    key = ["session", "task", "trial_id"]
    keystroke_counts = keystrokes.groupby(key).size().to_dict()
    word_counts = words.groupby(key).size().to_dict()
    perceptual_word_counts = perceptual_words.groupby(key).size().to_dict()
    keystroke_first = keystrokes.groupby(key)["start"].min().to_dict()
    keystroke_last = keystrokes.groupby(key)["stop"].max().to_dict()

    # List membership derived from exact sentence-text set equality between blocks.
    block_text_sets = {}
    for session, task in BLOCKS:
        frame = sentences[(sentences["session"].astype(str) == session) & (sentences["task"] == task)]
        block_text_sets[(session, task)] = frozenset(frame["text"].astype(str))
    derived_lists: dict[tuple[str, str], str] = {}
    classes: list[frozenset[str]] = []
    for block in BLOCKS:
        texts = block_text_sets[block]
        for index, existing in enumerate(classes):
            if existing == texts:
                derived_lists[block] = f"list{index + 1}"
                break
        else:
            classes.append(texts)
            derived_lists[block] = f"list{len(classes)}"
    list_assignment_matches_filenames = derived_lists == LOG_FILENAME_LISTS

    records = []
    for _, row in sentences.iterrows():
        session = str(row["session"])
        task = str(row["task"])
        trial = row["trial_id"]
        index = (session, task, trial)
        text = str(row["text"])
        records.append({
            "sentence_UID": str(row["sentence_UID"]),
            "subject": "S22",
            "session": session,
            "block": task,
            "list_id": derived_lists[(session, task)],
            "sentence_typed": str(row["sentence_typed"]),
            "sentence_presented": text,
            "unique_sentence_group_id": group_ids[text],
            "trial_id": int(trial),
            "number_of_keystrokes": int(keystroke_counts.get(index, 0)),
            "number_of_words": int(word_counts.get(index, 0)),
            "number_of_perceptual_words": int(perceptual_word_counts.get(index, 0)),
            "first_event_time": float(keystroke_first.get(index, row["start"])),
            "last_event_time": float(keystroke_last.get(index, row["stop"])),
            "sentence_start": float(row["start"]),
            "sentence_stop": float(row["stop"]),
            "target_characters": len(str(row["sentence_typed"])),
        })
    records.sort(key=lambda r: (r["session"], r["block"], r["trial_id"]))
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")

    by_uid = {r["sentence_UID"]: r for r in records}
    presented_counts: dict[str, int] = {}
    typed_counts: dict[str, int] = {}
    for record in records:
        presented_counts[record["sentence_presented"]] = presented_counts.get(record["sentence_presented"], 0) + 1
        typed_counts[record["sentence_typed"]] = typed_counts.get(record["sentence_typed"], 0) + 1
    sentence_structure = {
        "total_sentence_records": len(records),
        "unique_presented_texts": len(presented_counts),
        "presented_appearing_once": sum(1 for c in presented_counts.values() if c == 1),
        "presented_appearing_twice": sum(1 for c in presented_counts.values() if c == 2),
        "presented_appearing_more_than_twice": sum(1 for c in presented_counts.values() if c > 2),
        "unique_typed_texts": len(typed_counts),
        "typed_appearing_once": sum(1 for c in typed_counts.values() if c == 1),
        "typed_appearing_twice": sum(1 for c in typed_counts.values() if c == 2),
        "typed_appearing_more_than_twice": sum(1 for c in typed_counts.values() if c > 2),
        "grouping_key": "exact presented stimulus text (unnormalized)",
        "grouping_note": "Typed text cannot group sentences across sessions: the same stimulus is typed with different errors, so typed strings do not pair up.",
        "expectation_met": len(records) == 256 and len(presented_counts) == 128
        and all(c == 2 for c in presented_counts.values()),
    }

    block_statistics = {}
    for session, task in BLOCKS:
        rows = [r for r in records if r["session"] == session and r["block"] == task]
        block_statistics[block_label(session, task)] = {
            "session": session,
            "block": task,
            "list_id_derived_from_text": derived_lists[(session, task)],
            "list_id_from_log_filename": LOG_FILENAME_LISTS[(session, task)],
            "sentence_count": len(rows),
            "unique_presented_texts": len({r["sentence_presented"] for r in rows}),
            "unique_sentence_groups": len({r["unique_sentence_group_id"] for r in rows}),
            "total_keystrokes": sum(r["number_of_keystrokes"] for r in rows),
            "total_words": sum(r["number_of_words"] for r in rows),
            "total_perceptual_words": sum(r["number_of_perceptual_words"] for r in rows),
            "trial_id_range": [min(r["trial_id"] for r in rows), max(r["trial_id"] for r in rows)],
            "target_characters": describe([r["target_characters"] for r in rows]),
        }

    list_groups: dict[str, list[frozenset[str]]] = {}
    for block, list_id in derived_lists.items():
        list_groups.setdefault(list_id, []).append(block_text_sets[block])
    block_structure_checks = {
        "every_block_has_64_production_sentences": all(b["sentence_count"] == 64 for b in block_statistics.values()),
        "list1_blocks_share_identical_text_sets": len(set(list_groups["list1"])) == 1,
        "list2_blocks_share_identical_text_sets": len(set(list_groups["list2"])) == 1,
        "list1_list2_text_overlap": len(list_groups["list1"][0] & list_groups["list2"][0]),
        "same_list_cross_session_shared_texts": len(list_groups["list1"][0] & list_groups["list1"][1]),
        "all_four_blocks_present": len(block_statistics) == 4,
        "derived_list_assignment_matches_log_filenames": list_assignment_matches_filenames,
    }

    def side(blocks: list[tuple[str, str]]) -> list[dict]:
        return [r for r in records if (r["session"], r["block"]) in {(s, t) for s, t in blocks}]

    split_report = {}
    for name, spec in SPLITS.items():
        train, test = side(spec["train"]), side(spec["test"])
        train_texts = {r["sentence_presented"] for r in train}
        test_texts = {r["sentence_presented"] for r in test}
        train_typed = {r["sentence_typed"] for r in train}
        test_typed = {r["sentence_typed"] for r in test}
        train_groups = {r["unique_sentence_group_id"] for r in train}
        test_groups = {r["unique_sentence_group_id"] for r in test}
        train_uids = [r["sentence_UID"] for r in train]
        test_uids = [r["sentence_UID"] for r in test]
        train_trials = {(r["session"], r["block"], r["trial_id"]) for r in train}
        test_trials = {(r["session"], r["block"], r["trial_id"]) for r in test}
        text_overlap = len(train_texts & test_texts)
        group_overlap = len(train_groups & test_groups)
        leakage = {
            "exact_presented_text_overlap": text_overlap,
            "typed_target_string_overlap": len(train_typed & test_typed),
            "unique_sentence_group_overlap": group_overlap,
            "sentence_UID_overlap": len(set(train_uids) & set(test_uids)),
            "block_scoped_trial_overlap": len(train_trials & test_trials),
            "raw_trial_id_overlap": len({r["trial_id"] for r in train} & {r["trial_id"] for r in test}),
            "raw_trial_id_note": "trial_id is block-scoped (2-65 in every block); identical numbers refer to different recordings, so the block-scoped check is the meaningful one.",
            "record_overlap": len(set(train_uids) & set(test_uids)),
            "sentence_disjoint": text_overlap == 0 and group_overlap == 0,
        }
        statistics_block = {
            "train_sentence_count": len(train),
            "test_sentence_count": len(test),
            "train_unique_sentences": len(train_texts),
            "test_unique_sentences": len(test_texts),
            "train_keystrokes": sum(r["number_of_keystrokes"] for r in train),
            "test_keystrokes": sum(r["number_of_keystrokes"] for r in test),
            "train_sessions": sorted({r["session"] for r in train}),
            "test_sessions": sorted({r["session"] for r in test}),
            "train_blocks": sorted({block_label(r["session"], r["block"]) for r in train}),
            "test_blocks": sorted({block_label(r["session"], r["block"]) for r in test}),
            "train_lists": sorted({r["list_id"] for r in train}),
            "test_lists": sorted({r["list_id"] for r in test}),
            "mean_train_keystrokes_per_sentence": float(statistics.fmean([r["number_of_keystrokes"] for r in train])),
            "mean_test_keystrokes_per_sentence": float(statistics.fmean([r["number_of_keystrokes"] for r in test])),
            "train_target_characters": describe([r["target_characters"] for r in train]),
            "test_target_characters": describe([r["target_characters"] for r in test]),
        }
        expected = spec["sentence_disjoint_claim"]
        split_report[name] = {
            "purpose": spec["purpose"],
            "classification": "SENTENCE-DISJOINT" if expected else "CROSS-SESSION WITH SENTENCE OVERLAP",
            "sentence_disjoint_claimed": expected,
            "statistics": statistics_block,
            "leakage": leakage,
            "cross_session": sorted({r["session"] for r in train}) != sorted({r["session"] for r in test}),
            "cross_block": sorted({r["block"] for r in train}) != sorted({r["block"] for r in test}),
            "cross_list": sorted({r["list_id"] for r in train}) != sorted({r["list_id"] for r in test}),
            "status": "PASS" if leakage["sentence_disjoint"] == expected else "FAIL",
        }
        manifest = {
            "split_name": name,
            "purpose": spec["purpose"],
            "classification": split_report[name]["classification"],
            "train_sentence_UIDs": train_uids,
            "test_sentence_UIDs": test_uids,
            "train_blocks": statistics_block["train_blocks"],
            "test_blocks": statistics_block["test_blocks"],
            "train_sessions": statistics_block["train_sessions"],
            "test_sessions": statistics_block["test_sessions"],
            "train_lists": statistics_block["train_lists"],
            "test_lists": statistics_block["test_lists"],
            "text_overlap_count": text_overlap,
            "unique_sentence_group_overlap_count": group_overlap,
            "typed_target_string_overlap_count": leakage["typed_target_string_overlap"],
            "source_metadata": str(METADATA_PATH.relative_to(ROOT)).replace("\\", "/"),
            "training_performed": False,
        }
        (MANIFEST_DIR / spec["file"]).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    verification = {
        "all_256_production_records_included": len(records) == 256,
        "128_unique_presented_texts": len(presented_counts) == 128,
        "each_unique_sentence_appears_twice": all(c == 2 for c in presented_counts.values()),
        "list1_list2_overlap_zero": block_structure_checks["list1_list2_text_overlap"] == 0,
        "same_list_cross_session_overlap_64": block_structure_checks["same_list_cross_session_shared_texts"] == 64,
        "all_four_blocks_represented": block_structure_checks["all_four_blocks_present"],
        "split_A_leakage_audit_passes": split_report["A_same_session"]["status"] == "PASS",
        "split_B_leakage_audit_passes": split_report["B_same_session"]["status"] == "PASS",
        "split_C_leakage_audit_passes": split_report["C_sentence_disjoint"]["status"] == "PASS",
        "split_D_leakage_audit_passes": split_report["D_cross_session_sentence_disjoint"]["status"] == "PASS",
        "split_E_leakage_audit_passes": split_report["E_cross_session_sentence_disjoint"]["status"] == "PASS",
        "F_and_G_marked_sentence_overlapping": all(
            split_report[name]["classification"] == "CROSS-SESSION WITH SENTENCE OVERLAP"
            for name in ("F_cross_session_sentence_overlap", "G_cross_session_sentence_overlap")
        ),
        "no_model_training": True,
        "no_classifier_training": True,
        "no_preprocessing_change": True,
        "no_raw_data_modification": True,
        "no_downloads": True,
    }

    artifact = {
        "task": "5Y S22 split-ladder design and validation (metadata only)",
        "events_source": str(EVENTS_PATH.relative_to(ROOT)).replace("\\", "/"),
        "sentence_metadata": str(METADATA_PATH.relative_to(ROOT)).replace("\\", "/"),
        "dataset_counts": {
            "total_events": int(len(events)),
            "keystrokes": int(len(keystrokes)),
            "produced_words": int(len(words)),
            "perceptual_words": int(len(perceptual_words)),
            "production_sentences": len(records),
            "blocks": len(BLOCKS),
            "sessions": 2,
        },
        "sentence_structure": sentence_structure,
        "block_statistics": block_statistics,
        "block_structure_checks": block_structure_checks,
        "list_membership_method": "Derived from exact sentence-text set equality between blocks, then cross-checked against log filenames.",
        "splits": split_report,
        "verification": verification,
        "overall_status": "PASS" if all(v for k, v in verification.items()) and all(s["status"] == "PASS" for s in split_report.values()) else "FAIL",
        "figures": [str(FIGURE_PATH.relative_to(ROOT)).replace("\\", "/")],
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- figure -------------------------------------------------------
    figure = plt.figure(figsize=(14, 8.5))
    grid = figure.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=0.35)
    top = figure.add_subplot(grid[0])
    colors = {"list1": "#3a6ea5", "list2": "#c8553d"}
    # Columns are lists and rows are sessions, so vertically aligned blocks are
    # exactly the ones that share sentences.
    column_of = {"list1": 0, "list2": 1}
    for (session, task), list_id in derived_lists.items():
        x, y = column_of[list_id], 1 if session == "1" else 0
        stats = block_statistics[block_label(session, task)]
        top.add_patch(mpatches.FancyBboxPatch((x * 3.2, y * 1.5), 2.6, 1.0, boxstyle="round,pad=0.04",
                                              facecolor=colors[list_id], alpha=0.85, edgecolor="black"))
        top.text(x * 3.2 + 1.3, y * 1.5 + 0.72, f"session {session} / {task}", ha="center", color="white", fontsize=11, fontweight="bold")
        top.text(x * 3.2 + 1.3, y * 1.5 + 0.45, f"{list_id}  ({stats['sentence_count']} sentences)", ha="center", color="white", fontsize=10)
        top.text(x * 3.2 + 1.3, y * 1.5 + 0.2, f"{stats['total_keystrokes']} keystrokes", ha="center", color="white", fontsize=9)
    for centre, list_id in ((1.3, "list1"), (4.5, "list2")):
        top.annotate("", xy=(centre, 1.5), xytext=(centre, 1.0), arrowprops=dict(arrowstyle="<->", color=colors[list_id], lw=2.5))
        top.text(centre + 0.15, 1.23, f"{list_id}: same 64 sentences\n(cross-session repeat)", color=colors[list_id], fontsize=9.5, fontweight="bold", va="center")
    top.annotate("", xy=(1.3, 2.72), xytext=(4.5, 2.72), arrowprops=dict(arrowstyle="<->", color="black", lw=2))
    top.text(2.9, 2.8, "list1 vs list2: 0 shared sentences", ha="center", fontsize=10.5, fontweight="bold")
    top.set_xlim(-0.3, 7.2)
    top.set_ylim(-0.2, 3.1)
    top.axis("off")
    top.set_title("S22 structure: list1 and list2 repeat across sessions; list1 and list2 share no sentences", fontsize=12)

    bottom = figure.add_subplot(grid[1])
    names = list(SPLITS)
    block_order = [("1", "block1"), ("1", "block2"), ("2", "block1"), ("2", "block2")]
    for row, name in enumerate(names):
        spec = SPLITS[name]
        for column, block in enumerate(block_order):
            role = "train" if block in spec["train"] else ("test" if block in spec["test"] else None)
            face = {"train": "#3a6ea5", "test": "#e0a458", None: "#f0f0f0"}[role]
            bottom.add_patch(mpatches.Rectangle((column, -row), 0.96, 0.9, facecolor=face, edgecolor="grey"))
            if role:
                bottom.text(column + 0.48, -row + 0.45, role.upper(), ha="center", va="center",
                            color="white" if role == "train" else "black", fontsize=9, fontweight="bold")
        report = split_report[name]
        label = "sentence-disjoint" if report["leakage"]["sentence_disjoint"] else "SENTENCE OVERLAP"
        colour = "#2e7d32" if report["leakage"]["sentence_disjoint"] else "#c62828"
        bottom.text(4.15, -row + 0.45, f"{name}   [{label}: text overlap {report['leakage']['exact_presented_text_overlap']}]",
                    va="center", fontsize=9.5, color=colour)
    bottom.set_xticks([c + 0.48 for c in range(4)], [f"s{s}/{t}\n{derived_lists[(s, t)]}" for s, t in block_order], fontsize=9)
    bottom.set_yticks([])
    bottom.set_xlim(-0.1, 9.6)
    bottom.set_ylim(-len(names) + 0.05, 1.0)
    bottom.set_title("Split ladder A-G", fontsize=12)
    for spine in bottom.spines.values():
        spine.set_visible(False)
    figure.savefig(FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(json.dumps({
        "overall_status": artifact["overall_status"],
        "sentence_structure": {k: v for k, v in sentence_structure.items() if not isinstance(v, str)},
        "block_structure_checks": block_structure_checks,
        "splits": {name: {"status": report["status"], "text_overlap": report["leakage"]["exact_presented_text_overlap"],
                          "typed_overlap": report["leakage"]["typed_target_string_overlap"],
                          "train": report["statistics"]["train_blocks"], "test": report["statistics"]["test_blocks"]}
                   for name, report in split_report.items()},
        "verification": verification,
    }, indent=2))


if __name__ == "__main__":
    main()
