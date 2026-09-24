"""Run the expanded CTC job list with bounded concurrency, resumably.

Each job is one (split, mode, seed) process of `scripts/s22_expanded_ctc.py`.
Jobs whose result JSON already exists are skipped, so the queue can be stopped
and restarted without losing finished work. Concurrency is bounded because 12
simultaneous runs exceed this machine's physical memory and force paging.

    python scripts/s22_expanded_ctc_queue.py --workers 5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results/runs/s22_expanded_ctc"
TRAINER = ROOT / "scripts/s22_expanded_ctc.py"

# Longest jobs first so the tail of the queue is cheap.
JOBS = [
    ("C", "real", 33), ("C", "real", 123), ("C", "real", 777),
    ("C", "control", 33), ("C", "control", 123), ("C", "control", 777),
    ("E", "real", 33), ("E", "real", 123), ("E", "real", 777),
    ("D", "real", 33), ("D", "real", 123), ("D", "real", 777),
]


def result_path(split: str, mode: str, seed: int) -> Path:
    return RUN_DIR / f"{split}_{mode}_seed{seed}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    arguments = parser.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    pending = [job for job in JOBS if not result_path(*job).exists()]
    done = [job for job in JOBS if result_path(*job).exists()]
    print(f"{len(done)} already finished, {len(pending)} to run, {arguments.workers} at a time", flush=True)

    env = {**os.environ, "PYTHONPATH": "src"}
    running: list[tuple[tuple[str, str, int], subprocess.Popen, object]] = []
    queue = list(pending)

    while queue or running:
        while queue and len(running) < arguments.workers:
            split, mode, seed = queue.pop(0)
            log = RUN_DIR / f"log_{split}_{mode}_{seed}.txt"
            handle = log.open("w", encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, str(TRAINER), "--split", split, "--mode", mode, "--seed", str(seed)],
                cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
            )
            running.append(((split, mode, seed), process, handle))
            print(f"started {split}/{mode}/{seed} pid={process.pid}", flush=True)
        time.sleep(arguments.poll_seconds)
        still: list = []
        for job, process, handle in running:
            code = process.poll()
            if code is None:
                still.append((job, process, handle))
                continue
            handle.close()
            split, mode, seed = job
            status = "ok" if result_path(split, mode, seed).exists() else f"FAILED (exit {code})"
            print(f"finished {split}/{mode}/{seed}: {status}", flush=True)
        running = still

    missing = [job for job in JOBS if not result_path(*job).exists()]
    print(f"queue complete; missing={missing}", flush=True)


if __name__ == "__main__":
    main()
