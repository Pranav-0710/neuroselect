"""Phase 7 Tasks 1-5: the exact official Brain2Qwerty v1 splitter on S22.

Runs the unmodified official ``Brain2QwertyV1Splitter`` (revision 5f98896,
configured as in ``xp_config``: seed 1, threshold 0.5, ratios 80/10/10) on the
events produced by the official study + ``SpanishBCBLPreprocessing``, i.e. the
official ``Data.build_events`` order (study -> preprocessing -> splitter).

The official clusters are a local variable of ``_run``. They are captured, not
re-derived, by temporarily substituting the ``random`` module seen by the
official ``transforms`` module with a recorder that forwards every call to the
real ``random`` and keeps a copy of the list handed to ``shuffle``; the source
file is not touched. An independent verbatim re-implementation is then run and
must reproduce the captured clusters and the official assignment exactly.

From the official clusters it builds:
  * the official 80/10/10 split mapped onto all 256 sentence records;
  * D-clean (s1 list1 -> s2 list2) and E-clean (s1 list2 -> s2 list1), with
    every official cluster kept on one side, a validation carve-out from the
    training clusters, and an audit of whether the literal definition is
    feasible.

Run in the pinned official environment plus the Phase 6 overlay:

    PYTHONPATH="<overlay>/site;vendor/brain2qwerty;src" <pinned-env>/python scripts/phase7_official_split.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("phase6_harness", ROOT / "scripts/phase6_official_v1_harness.py")
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)

np, ns = h.np, h.ns
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

MANIFEST_DIR = ROOT / "data/manifests"
AUDIT_OUT = ROOT / "results/phase7_official_split_audit.json"
MANIFESTS_OUT = ROOT / "results/phase7_clean_split_manifests.json"
CLUSTERS_OUT = ROOT / "results/phase7_official_clusters.json"
PARTITIONS = ("train", "val", "test")
OFFICIAL_FILES = {p: MANIFEST_DIR / f"official_v1_clean_{p}.json" for p in PARTITIONS}
CROSS_SESSION = {
    "D_clean": {"train": ("1", "block1"), "test": ("2", "block1"),
                "definition": "TRAIN = session 1 list1 (s1/block1); TEST = session 2 list2 (s2/block1)"},
    "E_clean": {"train": ("1", "block2"), "test": ("2", "block2"),
                "definition": "TRAIN = session 1 list2 (s1/block2); TEST = session 2 list1 (s2/block2)"},
}
# Validation carve-out for D/E-clean: the official train:val proportion (80:10),
# allocated over the training side's official clusters with the official seed
# and the official first-fit rule. A research choice, not part of the official code.
CARVE_OUT_RATIOS = (8 / 9, 1 / 9)
OFFICIAL_THRESHOLD = 0.5  # asserted equal to the configured official splitter below


# ---- capture of the official clusters ------------------------------------

class _RecordingRandom:
    """Forwards everything to ``random``; records the list passed to ``shuffle``."""

    def __init__(self, real):
        self._real = real
        self.calls: list[str] = []
        self.before_shuffle: list[list[int]] | None = None
        self.after_shuffle: list[list[int]] | None = None

    def __getattr__(self, name):
        self.calls.append(name)
        return getattr(self._real, name)

    def shuffle(self, items):
        self.calls.append("shuffle")
        self.before_shuffle = [list(c) for c in items]
        self._real.shuffle(items)
        self.after_shuffle = [list(c) for c in items]


def run_official_splitter(splitter, events):
    recorder = _RecordingRandom(random)
    module = h._official_transforms
    if module.random is not random:
        raise RuntimeError("Official transforms module does not use the stdlib random module")
    module.random = recorder
    try:
        output = splitter.run(events.copy())
    finally:
        module.random = random
    if recorder.after_shuffle is None:
        raise RuntimeError("Official splitter did not shuffle; clusters not captured")
    return output, recorder


def reimplemented_split(events, ratios, seed, threshold):
    """Verbatim re-statement of Brain2QwertyV1Splitter._run, used only as a check."""
    buttons = events[events["type"] == "Keystroke"]
    unique_sentences = buttons["sentence"].unique()
    random.seed(seed)
    sim = cosine_similarity(TfidfVectorizer().fit_transform(unique_sentences))
    clusters, visited = [], set()
    for i in range(sim.shape[0]):
        if i in visited:
            continue
        cluster, expanded = {i}, True
        while expanded:
            expanded = False
            for idx in list(cluster):
                for j in range(sim.shape[1]):
                    if j not in cluster and sim[idx, j] > threshold:
                        cluster.add(j)
                        expanded = True
        visited.update(cluster)
        clusters.append(list(cluster))
    random.shuffle(clusters)
    total = len(buttons)
    sizes = {"train": int(ratios[0] * total), "val": int(ratios[1] * total),
             "test": total - int(ratios[0] * total) - int(ratios[1] * total)}
    current = dict.fromkeys(PARTITIONS, 0)
    assignment, trace = {}, []
    for cluster in clusters:
        sentences = [unique_sentences[idx] for idx in cluster]
        size = len(buttons[buttons["sentence"].isin(sentences)])
        assigned = "test"
        for split in PARTITIONS:
            if current[split] + size <= sizes[split]:
                current[split] += size
                assigned = split
                break
        trace.append({"cluster_members": len(cluster), "keystrokes": int(size), "assigned": assigned,
                      "running_totals": dict(current)})
        for sentence in sentences:
            assignment[sentence] = assigned
    return list(unique_sentences), sim, clusters, assignment, sizes, current, trace


# ---- helpers ---------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, payload) -> str:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")
    return sha256_bytes(text.encode("utf-8"))


def record_view(record: dict) -> dict:
    return {key: record[key] for key in (
        "sentence_UID", "unique_sentence_group_id", "tfidf_cluster_id", "session", "block", "list_id",
        "sentence_presented", "sentence_typed", "keystrokes")}


def composition(records: list[dict]) -> dict:
    return {
        "sentence_records": len(records),
        "unique_sentence_groups": len({r["unique_sentence_group_id"] for r in records}),
        "tfidf_clusters": len({r["tfidf_cluster_id"] for r in records}),
        "keystrokes": int(sum(r["keystrokes"] for r in records)),
        "by_session_block_list": {
            f"session{s}/{b}/{l}": {"sentence_records": n,
                                    "keystrokes": int(sum(r["keystrokes"] for r in records
                                                          if (r["session"], r["block"], r["list_id"]) == (s, b, l)))}
            for (s, b, l), n in sorted(Counter((r["session"], r["block"], r["list_id"]) for r in records).items())
        },
    }


def overlap(a: list[dict], b: list[dict], sim, position) -> dict:
    """All leakage checks between two partitions."""
    def keys(records, field):
        return {r[field] for r in records}

    a_text, b_text = sorted(keys(a, "sentence_presented")), sorted(keys(b, "sentence_presented"))
    if a_text and b_text:
        cross = sim[np.ix_([position[t] for t in a_text], [position[t] for t in b_text])]
        max_cosine, above = float(cross.max()), int((cross > OFFICIAL_THRESHOLD).sum())
        worst = np.unravel_index(int(cross.argmax()), cross.shape)
        worst_pair = {"a": a_text[worst[0]], "b": b_text[worst[1]], "cosine": float(cross[worst])}
    else:
        max_cosine, above, worst_pair = None, 0, None
    return {
        "presented_text_overlap": len(keys(a, "sentence_presented") & keys(b, "sentence_presented")),
        "typed_text_overlap": len(keys(a, "sentence_typed") & keys(b, "sentence_typed")),
        "sentence_UID_overlap": len(keys(a, "sentence_UID") & keys(b, "sentence_UID")),
        "unique_sentence_group_overlap": len(keys(a, "unique_sentence_group_id") & keys(b, "unique_sentence_group_id")),
        "tfidf_cluster_overlap": len(keys(a, "tfidf_cluster_id") & keys(b, "tfidf_cluster_id")),
        "paraphrase_pairs_above_official_threshold": above,
        "max_cross_partition_tfidf_cosine": max_cosine,
        "most_similar_cross_partition_pair": worst_pair,
    }


def overlap_is_clean(result: dict) -> bool:
    return all(result[k] == 0 for k in ("presented_text_overlap", "typed_text_overlap", "sentence_UID_overlap",
                                         "unique_sentence_group_overlap", "tfidf_cluster_overlap",
                                         "paraphrase_pairs_above_official_threshold"))


def twin_check(partitions: dict[str, list[dict]], universe: list[dict]) -> dict:
    """No group may have an occurrence in one partition and another occurrence in a different one."""
    where: dict[str, set] = {}
    for name, records in partitions.items():
        for r in records:
            where.setdefault(r["unique_sentence_group_id"], set()).add(name)
    split_groups = sorted(g for g, names in where.items() if len(names) > 1)
    # Occurrences of evaluated groups that sit outside every partition (unused) are reported, not leaks.
    used = {r["sentence_UID"] for records in partitions.values() for r in records}
    unused_twins = sorted(r["sentence_UID"] for r in universe
                          if r["unique_sentence_group_id"] in where and r["sentence_UID"] not in used)
    return {"groups_split_across_partitions": split_groups,
            "no_group_split_across_partitions": not split_groups,
            "unused_occurrences_of_used_groups": unused_twins}


# ---- main ------------------------------------------------------------------

def main() -> None:
    cfg, xp = h.official_experiment()
    transforms = list(xp.data.transforms)
    names = [type(t).__name__ for t in transforms]
    if names != ["SpanishBCBLPreprocessing", "Brain2QwertyV1Splitter"]:
        raise RuntimeError(f"Unexpected official transforms {names}")
    preprocessing, splitter = transforms
    if splitter.threshold != OFFICIAL_THRESHOLD:
        raise RuntimeError(f"Official threshold is {splitter.threshold}")

    official_study = cfg["data"]["study"]
    events = ns.events.Study(name=official_study["name"], path=official_study["path"]).run()
    events = preprocessing.run(events)
    stored = h.check_against_stored_events(events)
    if not stored["identical_timeline_start_button"]:
        raise RuntimeError("Rebuilt official events differ from the stored extraction")

    split_events, recorder = run_official_splitter(splitter, events)
    unique_sentences, sim, clusters, assignment, sizes, achieved, trace = reimplemented_split(
        events.copy(), splitter.splitting_ratios, splitter.seed, splitter.threshold)
    keystroke_rows = split_events[split_events["type"] == "Keystroke"]
    official_assignment = dict(zip(keystroke_rows["sentence"], keystroke_rows["split"]))
    exact = {
        "captured_clusters_equal_reimplementation": recorder.after_shuffle == clusters,
        "official_assignment_equals_reimplementation": official_assignment == assignment,
        "every_keystroke_assigned": bool(keystroke_rows["split"].notna().all()),
        "random_module_calls_seen_by_official_code": recorder.calls,
    }
    if not (exact["captured_clusters_equal_reimplementation"] and exact["official_assignment_equals_reimplementation"]
            and exact["every_keystroke_assigned"]):
        raise RuntimeError(f"Official splitter not reproduced exactly: {exact}")

    # ---- clusters in official allocation order -----------------------------
    position = {text: i for i, text in enumerate(unique_sentences)}
    cluster_of_text = {}
    for order, members in enumerate(clusters):
        for idx in members:
            cluster_of_text[unique_sentences[idx]] = f"TC{order:03d}"

    # ---- the 256 sentence records ------------------------------------------
    metadata = [json.loads(line) for line in h.METADATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    event_sentence = keystroke_rows.groupby(keystroke_rows["sentence_UID"].astype(str))["sentence"].first().to_dict()
    event_keystrokes = keystroke_rows.groupby(keystroke_rows["sentence_UID"].astype(str)).size().to_dict()
    records = []
    for row in metadata:
        uid = row["sentence_UID"]
        if event_sentence.get(uid) != row["sentence_presented"]:
            raise RuntimeError(f"Official sentence text differs from metadata for {uid}")
        if int(event_keystrokes[uid]) != int(row["number_of_keystrokes"]):
            raise RuntimeError(f"Keystroke count differs for {uid}")
        records.append({**row, "keystrokes": int(event_keystrokes[uid]),
                        "tfidf_cluster_id": cluster_of_text[row["sentence_presented"]],
                        "official_split": assignment[row["sentence_presented"]]})
    if len(records) != 256 or len(event_sentence) != 256:
        raise RuntimeError("Expected 256 sentence records")
    group_texts = {}
    for r in records:
        group_texts.setdefault(r["unique_sentence_group_id"], set()).add(r["sentence_presented"])
    if len(group_texts) != 128 or any(len(t) != 1 for t in group_texts.values()):
        raise RuntimeError("Expected 128 unique sentence groups, one presented text each")

    lists_of_text = {}
    for r in records:
        lists_of_text.setdefault(r["sentence_presented"], set()).add(r["list_id"])
    cluster_rows = []
    for order, members in enumerate(clusters):
        texts = [unique_sentences[i] for i in members]
        member_records = [r for r in records if r["sentence_presented"] in texts]
        lists = sorted({l for t in texts for l in lists_of_text[t]})
        inner = sim[np.ix_(members, members)]
        cluster_rows.append({
            "tfidf_cluster_id": f"TC{order:03d}",
            "official_allocation_order": order,
            "size_sentence_texts": len(members),
            "sentence_texts": texts,
            "unique_sentence_groups": sorted({r["unique_sentence_group_id"] for r in member_records}),
            "lists": lists,
            "cross_list": len(lists) > 1,
            "sentence_records": len(member_records),
            "keystrokes": int(sum(r["keystrokes"] for r in member_records)),
            "official_split": assignment[texts[0]],
            "max_within_cluster_cosine": float(inner[~np.eye(len(members), dtype=bool)].max()) if len(members) > 1 else None,
            "allocation_trace": trace[order],
        })
    multi = [c for c in cluster_rows if c["size_sentence_texts"] > 1]
    off_diagonal = sim[~np.eye(len(unique_sentences), dtype=bool)]
    list2_with_list1_partner = sum(
        1 for t in unique_sentences if "list2" in lists_of_text[t]
        and any(sim[position[t], position[u]] > splitter.threshold for u in unique_sentences if "list1" in lists_of_text[u]))
    cluster_stats = {
        "unique_sentence_texts": len(unique_sentences),
        "clusters": len(clusters),
        "cluster_size_histogram": {str(k): v for k, v in sorted(Counter(len(c) for c in clusters).items())},
        "singleton_clusters": sum(1 for c in clusters if len(c) == 1),
        "multi_sentence_clusters": len(multi),
        "sentence_texts_in_multi_sentence_clusters": sum(c["size_sentence_texts"] for c in multi),
        "cross_list_clusters": sum(1 for c in cluster_rows if c["cross_list"]),
        "within_list1_clusters_multi": sum(1 for c in multi if c["lists"] == ["list1"]),
        "within_list2_clusters_multi": sum(1 for c in multi if c["lists"] == ["list2"]),
        "list2_texts_with_a_list1_partner_above_threshold": list2_with_list1_partner,
        "pairs_above_threshold": int((np.triu(sim, 1) > splitter.threshold).sum()),
        "max_off_diagonal_cosine": float(off_diagonal.max()),
        "largest_cluster_size": max(len(c) for c in clusters),
        "transitive_merges": sum(1 for c in multi if c["size_sentence_texts"] > 2),
    }

    # ---- official 80/10/10 --------------------------------------------------
    official = {p: [r for r in records if r["official_split"] == p] for p in PARTITIONS}
    official_overlaps = {f"{a}_vs_{b}": overlap(official[a], official[b], sim, position)
                         for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    official_twins = twin_check(official, records)
    total_keystrokes = sum(r["keystrokes"] for r in records)
    official_summary = {
        "target_keystrokes": sizes,
        "achieved_keystrokes": {p: int(sum(r["keystrokes"] for r in official[p])) for p in PARTITIONS},
        "achieved_fraction": {p: sum(r["keystrokes"] for r in official[p]) / total_keystrokes for p in PARTITIONS},
        "running_totals_after_allocation": achieved,
        "composition": {p: composition(official[p]) for p in PARTITIONS},
        "overlaps": official_overlaps,
        "twin_check": official_twins,
        "all_clean": all(overlap_is_clean(v) for v in official_overlaps.values())
        and official_twins["no_group_split_across_partitions"],
    }
    official_hashes = {}
    for p in PARTITIONS:
        others = [q for q in PARTITIONS if q != p]
        payload = {
            "split_name": "official_v1_clean",
            "partition": p,
            "subject": "S22",
            "official_revision": h.OFFICIAL_REVISION,
            "construction": ("Exact official Brain2QwertyV1Splitter (seed 1, threshold 0.5, ratios 0.8/0.1/0.1 in "
                             "keystrokes) on the S22 events of the official study + SpanishBCBLPreprocessing; "
                             "sentence texts mapped back to all sentence records"),
            "scope_note": ("The published official run applies the splitter to all participants' events; only S22 "
                           "is available here, so the TF-IDF fit, clusters and allocation are over S22's 128 texts"),
            "sentence_groups": sorted({r["unique_sentence_group_id"] for r in official[p]}),
            "tfidf_clusters": sorted({r["tfidf_cluster_id"] for r in official[p]}),
            "sentence_UIDs": [r["sentence_UID"] for r in official[p]],
            "records": [record_view(r) for r in official[p]],
            **composition(official[p]),
            "overlap_checks": {f"vs_{q}": overlap(official[p], official[q], sim, position) for q in others},
            "twin_check": official_twins,
            "sentence_disjoint": all(overlap_is_clean(overlap(official[p], official[q], sim, position)) for q in others),
            "training_performed": False,
        }
        official_hashes[OFFICIAL_FILES[p].name] = write_json(OFFICIAL_FILES[p], payload)

    # ---- D-clean / E-clean ----------------------------------------------------
    cross_results, cross_hashes = {}, {}
    cluster_order = {c["tfidf_cluster_id"]: c["official_allocation_order"] for c in cluster_rows}
    for name, spec in CROSS_SESSION.items():
        train_side = [r for r in records if (r["session"], r["block"]) == spec["train"]]
        test_side = [r for r in records if (r["session"], r["block"]) == spec["test"]]
        train_clusters = {r["tfidf_cluster_id"] for r in train_side}
        test_clean = [r for r in test_side if r["tfidf_cluster_id"] not in train_clusters]
        excluded = [r for r in test_side if r["tfidf_cluster_id"] in train_clusters]
        shared_clusters = sorted({r["tfidf_cluster_id"] for r in excluded})

        # validation carve-out over the training side's official clusters
        side_clusters = sorted(train_clusters, key=cluster_order.get)
        random.seed(splitter.seed)
        random.shuffle(side_clusters)
        side_total = sum(r["keystrokes"] for r in train_side)
        side_sizes = {"train": int(CARVE_OUT_RATIOS[0] * side_total), "val": int(CARVE_OUT_RATIOS[1] * side_total)}
        side_current = {"train": 0, "val": 0}
        side_assigned, overflow = {}, []
        for cid in side_clusters:
            size = sum(r["keystrokes"] for r in train_side if r["tfidf_cluster_id"] == cid)
            assigned = None
            for part in ("train", "val"):
                if side_current[part] + size <= side_sizes[part]:
                    side_current[part] += size
                    assigned = part
                    break
            if assigned is None:  # fits neither: stays on the training side
                assigned = "train"
                side_current["train"] += size
                overflow.append(cid)
            side_assigned[cid] = assigned
        parts = {
            "train": [r for r in train_side if side_assigned[r["tfidf_cluster_id"]] == "train"],
            "val": [r for r in train_side if side_assigned[r["tfidf_cluster_id"]] == "val"],
            "test": test_clean,
        }
        overlaps = {f"{a}_vs_{b}": overlap(parts[a], parts[b], sim, position)
                    for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
        twins = twin_check(parts, records)
        literal_overlap = overlap(train_side, test_side, sim, position)
        clean = all(overlap_is_clean(v) for v in overlaps.values()) and twins["no_group_split_across_partitions"]
        result = {
            "definition": spec["definition"],
            "literal_definition": {
                "train_records": len(train_side),
                "test_records": len(test_side),
                "overlap_checks": literal_overlap,
                "cluster_integrity_preserved": literal_overlap["tfidf_cluster_overlap"] == 0,
                "feasible_without_breaking_clusters": literal_overlap["tfidf_cluster_overlap"] == 0,
            },
            "resolution": ("Every official cluster with a member on the training side stays entirely on the training "
                           "side; test-side sentences in such clusters are excluded from the test partition. Nothing "
                           "is moved across sessions or blocks."),
            "test_records_excluded_for_cluster_integrity": len(excluded),
            "clusters_shared_by_literal_train_and_test": len(shared_clusters),
            "excluded_test_records": [{**record_view(r),
                                       "train_side_cluster_members": sorted(
                                           t["sentence_presented"] for t in train_side
                                           if t["tfidf_cluster_id"] == r["tfidf_cluster_id"])}
                                      for r in excluded],
            "validation_carve_out": {
                "rule": ("official seed (1) shuffle of the training side's official clusters, then the official "
                         "first-fit allocation at the official train:val proportion 80:10; a cluster fitting neither "
                         "stays in train. A research choice for early stopping, not official code."),
                "target_keystrokes": side_sizes,
                "achieved_keystrokes": {p: int(sum(r["keystrokes"] for r in parts[p])) for p in ("train", "val")},
                "overflow_clusters_kept_in_train": overflow,
            },
            "composition": {p: composition(parts[p]) for p in PARTITIONS},
            "overlaps": overlaps,
            "twin_check": twins,
            "all_clean": clean,
            "classification": "CLUSTER-DISJOINT (official TF-IDF clusters)" if clean else "NOT CLUSTER-DISJOINT",
        }
        cross_results[name] = result
        path = MANIFEST_DIR / f"s22_{name}.json"
        cross_hashes[path.name] = write_json(path, {
            "split_name": name,
            "subject": "S22",
            "official_revision": h.OFFICIAL_REVISION,
            "definition": spec["definition"],
            "classification": result["classification"],
            "literal_definition_feasible": result["literal_definition"]["feasible_without_breaking_clusters"],
            "resolution": result["resolution"],
            "partitions": {p: {
                "sentence_groups": sorted({r["unique_sentence_group_id"] for r in parts[p]}),
                "tfidf_clusters": sorted({r["tfidf_cluster_id"] for r in parts[p]}),
                "sentence_UIDs": [r["sentence_UID"] for r in parts[p]],
                "records": [record_view(r) for r in parts[p]],
                **composition(parts[p]),
            } for p in PARTITIONS},
            "excluded_test_records": result["excluded_test_records"],
            "validation_carve_out": result["validation_carve_out"],
            "overlap_checks": overlaps,
            "twin_check": twins,
            "sentence_disjoint": clean,
            "training_performed": False,
        })

    clusters_hash = write_json(CLUSTERS_OUT, {
        "task": "Phase 7: official TF-IDF clusters of the S22 sentence texts (captured from the official splitter)",
        "clusters": cluster_rows,
        "text_to_cluster": cluster_of_text,
    })

    audit = {
        "task": "Phase 7 Tasks 1-3: exact official splitter, cluster statistics, leakage audit",
        "official_revision": h.OFFICIAL_REVISION,
        "official_splitter": {
            "class": "brain2qwerty_v1.transforms.Brain2QwertyV1Splitter",
            "configured_as": next(t for t in cfg["data"]["transforms"] if t["name"] == "Brain2QwertyV1Splitter"),
            "realized_parameters": {"splitting_ratios": list(splitter.splitting_ratios), "seed": splitter.seed,
                                    "threshold": splitter.threshold},
            "pipeline_position": "study -> SpanishBCBLPreprocessing -> Brain2QwertyV1Splitter (official Data.build_events)",
            "sentence_normalization": ("none: the raw 'sentence' field of keystroke events (presented text, already "
                                       "lowercase without accents in S22), deduplicated by pandas unique() in event order"),
            "tfidf": "sklearn TfidfVectorizer() with default settings, fit on the unique sentences of the given events",
            "similarity": "sklearn cosine_similarity over the TF-IDF rows",
            "clustering": ("connected components of the graph with an edge where cosine > threshold, grown "
                           "from each unvisited sentence in unique() order"),
            "shuffle": "random.seed(seed); random.shuffle(clusters) with the stdlib random module",
            "allocation": ("first-fit over (train, val, test) with keystroke budgets int(0.8 N), int(0.1 N), "
                           "N - both; a cluster that fits no budget goes to test"),
            "invoked": "the configured official instance's .run(), source unmodified",
            "cluster_capture": ("the official module's `random` reference was swapped for a forwarding recorder "
                                "during the call and restored afterwards"),
            "exact_reproduction": exact,
            "unique_sentence_order_sha256": sha256_bytes("\n".join(unique_sentences).encode("utf-8")),
            "sklearn_version": __import__("sklearn").__version__,
        },
        "events": {"rows": int(len(events)), **stored},
        "scope_note": ("The published official run applies the splitter to all participants' events; here only S22 "
                       "exists, so the TF-IDF vocabulary, clusters and keystroke budgets are those of S22's 128 texts. "
                       "This is the exact official procedure applied to the available universe, not a reproduction "
                       "of the published multi-subject partition."),
        "cluster_statistics": cluster_stats,
        "multi_sentence_clusters": [{k: c[k] for k in ("tfidf_cluster_id", "sentence_texts", "lists",
                                                         "max_within_cluster_cosine", "official_split")}
                                    for c in multi],
        "official_split": official_summary,
        "cross_session_clean": cross_results,
        "leakage_checks": {
            "official_split_clean": official_summary["all_clean"],
            "D_clean_clean": cross_results["D_clean"]["all_clean"],
            "E_clean_clean": cross_results["E_clean"]["all_clean"],
            "literal_D_feasible": cross_results["D_clean"]["literal_definition"]["feasible_without_breaking_clusters"],
            "literal_E_feasible": cross_results["E_clean"]["literal_definition"]["feasible_without_breaking_clusters"],
        },
    }
    write_json(AUDIT_OUT, audit)
    write_json(MANIFESTS_OUT, {
        "task": "Phase 7 Task 5: formal clean split manifests",
        "manifests": {
            **{name: {"path": f"data/manifests/{name}", "sha256": digest} for name, digest in official_hashes.items()},
            **{name: {"path": f"data/manifests/{name}", "sha256": digest} for name, digest in cross_hashes.items()},
        },
        "clusters_file": {"path": "results/phase7_official_clusters.json", "sha256": clusters_hash},
        "summary": {
            "official_v1_clean": {p: composition(official[p]) for p in PARTITIONS},
            **{name: {p: composition_row for p, composition_row in r["composition"].items()}
               for name, r in cross_results.items()},
        },
        "fields_per_record": ["sentence_UID", "unique_sentence_group_id", "tfidf_cluster_id", "session", "block",
                              "list_id", "sentence_presented", "sentence_typed", "keystrokes"],
    })
    print(json.dumps({
        "exact_reproduction": {k: v for k, v in exact.items() if isinstance(v, bool)},
        "cluster_statistics": cluster_stats,
        "official": {p: {k: official_summary["composition"][p][k] for k in
                         ("unique_sentence_groups", "sentence_records", "keystrokes", "tfidf_clusters")}
                     for p in PARTITIONS},
        "official_clean": official_summary["all_clean"],
        "cross": {n: {"literal_feasible": r["literal_definition"]["feasible_without_breaking_clusters"],
                      "excluded_test_records": r["test_records_excluded_for_cluster_integrity"],
                      "parts": {p: (r["composition"][p]["sentence_records"], r["composition"][p]["keystrokes"])
                                for p in PARTITIONS},
                      "clean": r["all_clean"]} for n, r in cross_results.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
