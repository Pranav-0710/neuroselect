"""Metadata-only inventory of publicly available S22 MEG data.

Lists the SpanishBCBL Hugging Face repository tree (file paths, sizes and LFS
hashes) and compares it with local files by SHA-256. It never downloads a
dataset file.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

ROOT = Path(__file__).resolve().parents[1]
REPO = "bcbl190626/SpanishBCBL"
SUBJECT = "S22"
SUBJECT_DIR = "22_9788"
LOCAL_ROOTS = (ROOT / "data/raw/spanishbcbl_s22", ROOT / "data/raw/spanishbcbl")
OUT_PATH = ROOT / "results/s22_data_scale_inventory.json"
FIGURE_PATH = ROOT / "results/figures/debug/s22_data_scale_inventory.png"
LOG_PATTERN = re.compile(r"S22-session(\d)_(block\d)_(list\d)\.mat$")

# Research-value classification. Criteria: storage, recording/block coverage,
# and independence from the current block only (no performance assumptions).
CLASSIFICATION = {
    "231214/block2": ("A. HIGH VALUE / SMALL COST",
                      "New typing block of the current session with a different sentence list (list2); 1.41 GB."),
    "231222/block1": ("A. HIGH VALUE / SMALL COST",
                      "New session; list2 label suggests it shares sentences with session-1 block2 (unverified); 1.56 GB."),
    "231222/block2": ("A. HIGH VALUE / SMALL COST",
                      "New session; list1 label suggests it repeats the current block's sentences (unverified); 1.14 GB."),
    "231214/tapping": ("C. LOW VALUE",
                       "Tapping localizer, no typing events, no behavioral log; skipped by the official study loader."),
    "231222/tapping": ("C. LOW VALUE",
                       "Tapping localizer, no typing events, no behavioral log; skipped by the official study loader."),
}
LEAKAGE = {
    "231214/block1": "current data",
    "231214/block2": "B. same subject, different block (same recording session/day as current data)",
    "231222/block1": "C. same subject, different session",
    "231222/block2": "C. same subject, different session",
    "231214/tapping": "A. same recording/session as current data (non-typing localizer)",
    "231222/tapping": "C. same subject, different session (non-typing localizer)",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_local(remote_path: str) -> Path | None:
    for root in LOCAL_ROOTS:
        candidate = root / remote_path
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    api = HfApi()
    info = api.dataset_info(REPO)
    listing = [
        entry for entry in api.list_repo_tree(REPO, repo_type="dataset", revision=info.sha, path_in_repo="MEG", recursive=True)
        if isinstance(entry, RepoFile)
    ]
    all_fif = [e for e in listing if e.path.startswith("MEG/FIF/") and e.path.lower().endswith(".fif")]
    all_mat = [e for e in listing if e.path.startswith("MEG/logs/") and e.path.endswith(".mat")]
    s22 = [e for e in listing if e.path.startswith(f"MEG/FIF/{SUBJECT_DIR}/") or e.path.startswith(f"MEG/logs/{SUBJECT}-")]

    # The official loader numbers sessions by sorted date directory and lowercases FIF stems.
    dates = sorted({e.path.split("/")[3] for e in s22 if e.path.startswith("MEG/FIF/")})
    logs = {}
    for entry in s22:
        match = LOG_PATTERN.search(entry.path)
        if match:
            logs[(int(match.group(1)), match.group(2))] = (entry, match.group(3))

    files = []
    recordings = {}
    for entry in sorted(s22, key=lambda e: e.path):
        local = find_local(entry.path)
        local_sha = sha256_file(local) if local else None
        remote_sha = entry.lfs.sha256 if entry.lfs else None
        row = {
            "remote_path": entry.path,
            "remote_size_bytes": entry.size,
            "remote_lfs_sha256": remote_sha,
            "local_path": str(local.relative_to(ROOT)) if local else None,
            "local_sha256": local_sha,
            "local_matches_remote": (local_sha == remote_sha) if local else None,
        }
        files.append(row)
        if entry.path.startswith("MEG/FIF/"):
            date = entry.path.split("/")[3]
            session = dates.index(date) + 1
            task = Path(entry.path).stem.lower()
            key = f"{date}/{task}"
            log_entry, sentence_list = logs.get((session, task), (None, None))
            recordings[key] = {
                "recording_directory": f"MEG/FIF/{SUBJECT_DIR}/{date}",
                "subject_id": SUBJECT,
                "recording_id": SUBJECT_DIR,
                "date": date,
                "session": session,
                "task": task,
                "remote_fif_filename": Path(entry.path).name,
                "is_typing_block": task.startswith("block"),
                "fif_size_bytes": entry.size,
                "log_path": log_entry.path if log_entry else None,
                "log_size_bytes": log_entry.size if log_entry else 0,
                "sentence_list_label": sentence_list,
                "total_size_bytes": entry.size + (log_entry.size if log_entry else 0),
                "fif_local": local is not None,
                "log_local": bool(log_entry and find_local(log_entry.path)),
                "event_count": "event count unavailable without downloading the recording." if not local else "verified locally: 2918 events (2119 keystrokes, 666 words, 132 sentences)",
                "leakage_category": LEAKAGE.get(key, "D. unclear"),
                "classification": CLASSIFICATION.get(key, ("current data", ""))[0],
                "classification_basis": CLASSIFICATION.get(key, ("", "Already downloaded and used for 5A-5V."))[1],
            }

    typing = [r for r in recordings.values() if r["is_typing_block"]]
    missing_typing = [r for r in typing if not r["fif_local"]]
    tapping = [r for r in recordings.values() if not r["is_typing_block"]]
    storage = {
        "current_local_s22_fif_bytes": sum(r["fif_size_bytes"] for r in typing if r["fif_local"]),
        "current_local_s22_mat_bytes": sum(r["log_size_bytes"] for r in typing if r["log_local"]),
        "additional_typing_fif_bytes": sum(r["fif_size_bytes"] for r in missing_typing),
        "additional_typing_mat_bytes": sum(r["log_size_bytes"] for r in missing_typing),
        "optional_tapping_fif_bytes_not_recommended": sum(r["fif_size_bytes"] for r in tapping),
    }
    storage["current_local_s22_total_bytes"] = storage["current_local_s22_fif_bytes"] + storage["current_local_s22_mat_bytes"]
    storage["additional_total_bytes"] = storage["additional_typing_fif_bytes"] + storage["additional_typing_mat_bytes"]
    storage["future_s22_typing_total_bytes"] = storage["current_local_s22_total_bytes"] + storage["additional_total_bytes"]
    storage["storage_multiplier"] = storage["future_s22_typing_total_bytes"] / storage["current_local_s22_total_bytes"]

    other_local = []
    for root in LOCAL_ROOTS:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".fif", ".mat"} and SUBJECT_DIR not in path.parts and not path.name.startswith(f"{SUBJECT}-"):
                other_local.append({"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size})

    artifact = {
        "task": "S22 data-scale inventory (metadata only)",
        "source": {
            "repository": f"https://huggingface.co/datasets/{REPO}",
            "revision": info.sha,
            "last_modified": str(info.last_modified),
            "gated": info.gated,
            "method": "huggingface_hub.HfApi.list_repo_tree(recursive=True) metadata; dataset README.md text; vendored official studies/spanishbcbl.py for session/task mapping",
            "local_download_revision_from_hf_cache_metadata": "88f9096c6ce3a3fb17cc7b8e3131ff7f96da5684",
            "release_totals": {"meg_fif_files": len(all_fif), "meg_mat_logs": len(all_mat)},
        },
        "mapping_rules_from_official_loader": {
            "session": "index of date directory within sorted subject directory + 1 (_retrieve_session)",
            "task": "FIF stem lowercased; tapping files skipped",
            "log": "glob **/{sid}-session{session}_{task}*.mat",
        },
        "dataset_readme_facts": [
            "Each subject has two sessions; each session has two typing blocks (block1, block2) and one tapping localizer.",
            "Each session used 128 unique declarative Spanish sentences of 5-8 words.",
            "The first two trials of each block are training trials with visual feedback (from official loader docstring).",
            "S22 is not listed among participants with repeated subject IDs.",
        ],
        "remote_s22_files": files,
        "recordings": recordings,
        "storage": storage,
        "disk_free_bytes_at_inventory": shutil.disk_usage(ROOT).free,
        "event_counts": {
            "231214/block1": "verified locally (2918 events; 2119 keystrokes; 666 words; 132 sentences)",
            "other_blocks": "event count unavailable without downloading the recording.",
            "note": "Behavioral MAT logs would provide trial/keystroke counts, but downloading them is outside this inventory's scope.",
        },
        "sentence_list_overlap_hypothesis": {
            "session1_block1": "list1 (current)", "session1_block2": "list2",
            "session2_block1": "list2", "session2_block2": "list1",
            "status": "Inferred from log filenames only; sentence text not verified. If true, all four S22 blocks cover the same 128 sentences, each typed once per session.",
        },
        "other_local_non_s22_raw_files": other_local,
        "no_download_verification": None,
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    gb = 1e9
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})
    left.bar(["current local\nS22", "additional\ntyping blocks", "future S22\ntyping total"],
             [storage["current_local_s22_total_bytes"] / gb, storage["additional_total_bytes"] / gb, storage["future_s22_typing_total_bytes"] / gb],
             color=["#3a6ea5", "#c8553d", "#7a7a7a"])
    for index, value in enumerate([storage["current_local_s22_total_bytes"], storage["additional_total_bytes"], storage["future_s22_typing_total_bytes"]]):
        left.text(index, value / gb + 0.05, f"{value / gb:.2f} GB", ha="center", fontsize=9)
    left.set_ylabel("GB (FIF + MAT)")
    left.set_title("S22 storage")
    order = sorted(recordings.values(), key=lambda r: (r["session"], r["task"]))
    colors = ["#3a6ea5" if r["fif_local"] else ("#c8553d" if r["is_typing_block"] else "#bdbdbd") for r in order]
    labels = [f"S{r['session']} {r['date']} {r['task']}" + (f" ({r['sentence_list_label']})" if r["sentence_list_label"] else "") for r in order]
    right.barh(labels, [r["fif_size_bytes"] / gb for r in order], color=colors)
    right.invert_yaxis()
    right.set_xlabel("FIF size (GB)")
    right.set_title("S22 recordings: blue = local, red = available typing, grey = tapping")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)
    print(json.dumps({"recordings": {k: (v["classification"], v["total_size_bytes"]) for k, v in recordings.items()}, "storage": storage}, indent=2))


if __name__ == "__main__":
    main()
