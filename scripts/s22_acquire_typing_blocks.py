"""5X: download the three remaining S22 typing blocks and verify them.

Downloads exactly the files listed in `results/s22_data_scale_inventory.json`
at the pinned dataset revision, verifies every file against the inventory's
SHA-256, and refuses to touch the existing block1 recording.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "results/s22_data_scale_inventory.json"
LOCAL_ROOT = ROOT / "data/raw/spanishbcbl_s22"
OUT_PATH = ROOT / "results/s22_acquisition.json"
REPO = "bcbl190626/SpanishBCBL"
PROTECTED = ("MEG/FIF/22_9788/231214/block1.fif", "MEG/logs/S22-session1_block1_list1.mat")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    plan = inventory["recommended_download_plan"]
    revision = plan["revision"]
    expected = {row["remote_path"]: row for row in inventory["remote_s22_files"]}

    protected_before = {path: sha256_file(LOCAL_ROOT / path) for path in PROTECTED}
    for path in PROTECTED:
        if protected_before[path] != expected[path]["remote_lfs_sha256"]:
            raise RuntimeError(f"Protected file already differs from the inventory hash: {path}")

    results = []
    for remote_path in plan["files"]:
        if remote_path in PROTECTED:
            raise RuntimeError(f"Refusing to re-download protected file {remote_path}")
        target = LOCAL_ROOT / remote_path
        started = time.time()
        downloaded = Path(hf_hub_download(
            REPO,
            remote_path,
            repo_type="dataset",
            revision=revision,
            local_dir=LOCAL_ROOT,
        ))
        elapsed = time.time() - started
        actual = sha256_file(downloaded)
        row = {
            "remote_path": remote_path,
            "local_path": str(downloaded.relative_to(ROOT)),
            "expected_sha256": expected[remote_path]["remote_lfs_sha256"],
            "actual_sha256": actual,
            "sha256_verified": actual == expected[remote_path]["remote_lfs_sha256"],
            "expected_size_bytes": expected[remote_path]["remote_size_bytes"],
            "actual_size_bytes": downloaded.stat().st_size,
            "size_verified": downloaded.stat().st_size == expected[remote_path]["remote_size_bytes"],
            "seconds": round(elapsed, 1),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        if not (row["sha256_verified"] and row["size_verified"]):
            raise RuntimeError(f"Verification failed for {remote_path}")
        if downloaded.resolve() != target.resolve():
            raise RuntimeError(f"Unexpected download location for {remote_path}: {downloaded}")

    protected_after = {path: sha256_file(LOCAL_ROOT / path) for path in PROTECTED}
    artifact = {
        "task": "5X S22 targeted acquisition",
        "repository": REPO,
        "revision": revision,
        "downloads": results,
        "all_verified": all(row["sha256_verified"] and row["size_verified"] for row in results),
        "total_downloaded_bytes": sum(row["actual_size_bytes"] for row in results),
        "protected_files_unchanged": protected_before == protected_after,
        "protected_file_hashes": protected_after,
        "tapping_downloaded": False,
        "decoder_or_classifier_run": False,
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in artifact.items() if k != "downloads"}, indent=2))


if __name__ == "__main__":
    main()
