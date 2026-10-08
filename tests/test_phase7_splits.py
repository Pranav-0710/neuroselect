import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data/manifests"


def _common():
    spec = importlib.util.spec_from_file_location("phase7_common", ROOT / "scripts/phase7_common.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c = _common()
OFFICIAL = [MANIFESTS / f"official_v1_clean_{p}.json" for p in c.PARTITIONS]


def _split(name):
    files = OFFICIAL if name == "official_v1_clean" else [MANIFESTS / f"s22_{name}.json"]
    return c.load_split(files)["partitions"]


@pytest.mark.parametrize("name", ["official_v1_clean", "D_clean", "E_clean"])
def test_partitions_are_cluster_and_text_disjoint(name):
    assert c.check_split(_split(name))["disjoint"]


def test_official_split_covers_every_record_and_keeps_twins_together():
    parts = _split("official_v1_clean")
    records = [r for p in c.PARTITIONS for r in parts[p]]
    assert len(records) == 256
    assert len({r["sentence_UID"] for r in records}) == 256
    assert sum(r["keystrokes"] for r in records) == 9650
    where = defaultdict(set)
    count = defaultdict(int)
    for p in c.PARTITIONS:
        for r in parts[p]:
            where[r["unique_sentence_group_id"]].add(p)
            count[r["unique_sentence_group_id"]] += 1
    assert len(where) == 128
    assert all(len(v) == 1 for v in where.values())
    assert all(n == 2 for n in count.values())


@pytest.mark.parametrize("name,train_block,test_block", [
    ("D_clean", ("1", "block1"), ("2", "block1")),
    ("E_clean", ("1", "block2"), ("2", "block2")),
])
def test_cross_session_clean_blocks_and_exclusions(name, train_block, test_block):
    manifest = json.loads((MANIFESTS / f"s22_{name}.json").read_text(encoding="utf-8"))
    parts = _split(name)
    for p in ("train", "val"):
        assert {(r["session"], r["block"]) for r in parts[p]} == {train_block}
    assert {(r["session"], r["block"]) for r in parts["test"]} == {test_block}
    train_clusters = {r["tfidf_cluster_id"] for p in ("train", "val") for r in parts[p]}
    assert all(r["tfidf_cluster_id"] in train_clusters for r in manifest["excluded_test_records"])
    assert len(parts["test"]) + len(manifest["excluded_test_records"]) == 64
    assert manifest["literal_definition_feasible"] is False


@pytest.mark.parametrize("name", ["official_v1_clean", "D_clean", "E_clean"])
def test_control_derangement_is_valid_and_deterministic(name):
    train = _split(name)["train"]
    first = c.stimulus_derangement(train)
    second = c.stimulus_derangement(train)
    assert first["donor_of_uid"] == second["donor_of_uid"]
    assert first["fixed_points"] == 0
    assert first["same_cluster_pairs"] == 0
    assert first["identical_target_text_pairs"] == 0
    by_uid = {r["sentence_UID"]: r for r in train}
    assert set(first["donor_of_uid"]) == set(by_uid)
    for uid, donor in first["donor_of_uid"].items():
        assert by_uid[donor]["unique_sentence_group_id"] == first["group_map"][by_uid[uid]["unique_sentence_group_id"]]


def test_adapt_labels_truncates_and_extends():
    donor = ["a", "b", "c"]
    assert c.adapt_labels(donor, 2) == ["a", "b"]
    assert c.adapt_labels(donor, 3) == ["a", "b", "c"]
    assert c.adapt_labels(donor, 6) == ["a", "b", "c", "<space>", "a", "b"]


def test_metrics_perfect_and_confusion():
    char_index = {0: "a", 1: "b", 2: " "}
    perfect = c.metrics([{"sentence_UID": "s", "target": [0, 1, 2, 0], "predicted": [0, 1, 2, 0]}], char_index, 3)
    assert perfect["keystroke_accuracy"] == 1.0
    assert perfect["character_f1"] == pytest.approx(1.0)
    assert perfect["mean_sentence_cer"] == 0.0
    wrong = c.metrics([{"sentence_UID": "s", "target": [0, 0, 1, 1], "predicted": [0, 1, 1, 1]}], char_index, 3)
    assert wrong["keystroke_accuracy"] == 0.75
    assert wrong["confusion_matrix"][0] == [1, 1, 0]
    assert wrong["per_character_recall"]["a"] == 0.5
    assert wrong["predicted_sequence_length_characters"] == wrong["target_sequence_length_characters"] == 4
