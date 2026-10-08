"""Phase 7: dataset manifest consumed by the official-v1 GPU harness.

Pins every input the harness reads by sha256: the eight raw S22 recordings and
behavioural logs (checked against the 5X acquisition record), the sentence
metadata, and the stored event extraction used for the events identity check.
The harness re-verifies these hashes before building any loader.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("phase7_common", ROOT / "scripts/phase7_common.py")
c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c)

DATA_ROOT = ROOT / "data/raw/spanishbcbl_s22"
OUT = ROOT / "data/manifests/s22_official_v1_dataset.json"


def main() -> None:
    acquisition = json.loads((ROOT / "results/s22_acquisition.json").read_text(encoding="utf-8"))
    expected = {row["remote_path"]: row["actual_sha256"] for row in acquisition["downloads"]}
    expected.update(acquisition["protected_file_hashes"])
    files = []
    for path in sorted([*DATA_ROOT.glob("MEG/FIF/**/*.fif"), *DATA_ROOT.glob("MEG/logs/*.mat")]):
        relative = path.relative_to(DATA_ROOT).as_posix()
        digest = c.sha256_file(path)
        if expected.get(relative) != digest:
            raise RuntimeError(f"{relative} does not match the acquisition record")
        files.append({"path": path.relative_to(ROOT).as_posix(), "kind": "raw", "sha256": digest,
                      "bytes": path.stat().st_size, "matches_acquisition_record": True})
    for relative, kind in (("data/manifests/s22_sentence_block_metadata.jsonl", "metadata"),
                           ("data/raw/spanishbcbl_s22/events_clean_all_blocks.pkl", "stored_events")):
        path = ROOT / relative
        files.append({"path": relative, "kind": kind, "sha256": c.sha256_file(path), "bytes": path.stat().st_size})
    if sum(1 for f in files if f["kind"] == "raw") != 8:
        raise RuntimeError("Expected 8 raw files")
    OUT.write_text(json.dumps({
        "dataset": "SpanishBCBL MEG, participant S22, four typing blocks",
        "source_repository": acquisition["repository"],
        "source_revision": acquisition["revision"],
        "study": "Pinet2024Meg (official brain2qwerty studies package)",
        "official_revision": "5f9889621d0df391c5aab37c996683d308e6e926",
        "data_root_env": "BRAIN2QWERTY_STUDIES (default: data/raw/spanishbcbl_s22)",
        "counts": {"sentence_records": 256, "unique_sentence_texts": 128, "keystrokes": 9650,
                   "blocks": 4, "sessions": 2},
        "files": files,
        "note": "No file is downloaded by the harness; every path must already exist and match.",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(files), "out": OUT.relative_to(ROOT).as_posix()}))


if __name__ == "__main__":
    main()
