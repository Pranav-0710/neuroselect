import json

import numpy as np
import torch

from neuroselect.data import NeuralTextDataset, collate_batch
from neuroselect.diagnostics import validate_ctc_geometry
from neuroselect.metrics import cer, wer
from neuroselect.models import ConvCTC
from neuroselect.splitting import split_by_unique_text
from neuroselect.vocab import VOCAB, decode_ctc, encode_text
from neuroselect.vocab_spanishbcbl import (
    VOCAB as SPANISH_VOCAB,
    decode_ctc as decode_spanish,
    encode_text as encode_spanish,
)


def write_fixture(tmp_path):
    records = []
    for index, text in enumerate(["ab", "ac", "ad", "ae", "af", "ag"]):
        signal = np.zeros((max(12, len(text) * 5), 7), dtype=np.float32)
        for step, char in enumerate(text):
            signal[step * 5 : (step + 1) * 5, ord(char) - ord("a")] = 3.0
        path = tmp_path / f"{index}.npy"
        np.save(path, signal)
        records.append(
            {
                "id": str(index),
                "signal_path": path.name,
                "text": text,
                "subject": "S1",
                "session": "session-1",
                "split": "train" if index < 4 else ("val" if index == 4 else "test"),
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return manifest


def test_vocab_and_decode():
    encoded = encode_text("a b")
    assert decode_ctc(encoded) == "a b"
    assert len(VOCAB) == 28


def test_dataset_shapes_and_ctc_dimensions(tmp_path):
    manifest = write_fixture(tmp_path)
    dataset = NeuralTextDataset(manifest, "train")
    batch = collate_batch([dataset[0], dataset[1]])
    assert batch["signals"].shape == (2, 12, 7)
    model = ConvCTC(7, len(VOCAB), hidden=16)
    assert model(batch["signals"]).shape == (2, 12, len(VOCAB))


def test_metrics():
    assert cer("abc", "adc") == 1 / 3
    assert wer("one two", "one three") == 1 / 2


def test_split_prevents_duplicate_text_leakage():
    records = [
        {"id": "1", "text": "same"},
        {"id": "2", "text": "same"},
        {"id": "3", "text": "other"},
    ]
    split = split_by_unique_text(records)
    by_text = {}
    for record in split:
        by_text.setdefault(record["text"], set()).add(record["split"])
    assert all(len(splits) == 1 for splits in by_text.values())


def test_spanishbcbl_mapping_round_trip():
    text = "la tasa excede las velocidades"
    encoded = encode_spanish(text)
    assert decode_spanish(encoded) == text
    assert len(SPANISH_VOCAB) == 30
    assert encode_spanish("@ 9") == [14, 9, 29]


def test_real_style_manifest_and_ctc_compatibility(tmp_path):
    signal = np.random.default_rng(4).normal(size=(80, 4)).astype(np.float32)
    np.save(tmp_path / "trial.npy", signal)
    record = {
        "id": "2.0_S22_1_block1",
        "signal_path": "trial.npy",
        "text": "la tasa",
        "subject": "S22",
        "session": "1",
        "split": "train",
        "vocab": "spanishbcbl",
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    dataset = NeuralTextDataset(manifest, "train")
    batch = collate_batch([dataset[0]])
    model = ConvCTC(4, len(SPANISH_VOCAB), hidden=16)
    logits = model(batch["signals"])
    loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)(
        logits.log_softmax(-1).transpose(0, 1),
        batch["targets"],
        batch["input_lengths"],
        batch["target_lengths"],
    )
    assert dataset[0]["signal"].shape == (80, 4)
    assert logits.shape == (1, 80, len(SPANISH_VOCAB))
    assert torch.isfinite(loss)


def test_ctc_geometry_accounts_for_repeated_targets():
    report = validate_ctc_geometry(5, [1, 2, 2, 3])
    assert report["U"] == 4
    assert report["T_over_U"] == 1.25
    assert report["minimum_ctc_timesteps"] == 5
    assert report["structurally_feasible"]
