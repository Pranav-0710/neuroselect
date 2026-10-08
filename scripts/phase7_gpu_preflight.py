"""Phase 7 Task 7: GPU memory/runtime preflight of the exact official v1 model.

On a CUDA machine: builds the official-split loaders and the exact official
model, allocates the official optimizer and scheduler (AdamW + OneCycleLR over
the official step budget), takes real training batches through forward,
backward and optimizer step, runs one real validation batch, and records peak
GPU memory and elapsed times. Without CUDA it records "GPU preflight not
executed" and exits without building the model: no estimate from a different
model or device is substituted.

    python scripts/phase7_gpu_preflight.py
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("phase7_harness", ROOT / "scripts/phase7_official_v1_harness.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
c, h, torch, np = H.c, H.h, H.torch, H.np

OUT = ROOT / "results/phase7_gpu_preflight.json"
TRAIN_STEPS = 5
SEED = 33


def main() -> None:
    report = {"task": "Phase 7 Task 7: GPU memory/runtime preflight (exact official v1 model)",
              "provenance": H.provenance()}
    if not torch.cuda.is_available():
        report.update({
            "status": "GPU preflight not executed",
            "reason": "no CUDA device available (torch reports cuda_available = False)",
            "measured": None,
            "what_it_will_measure": [
                "peak GPU memory allocated/reserved for weights + AdamW state + one real 64-keystroke training batch "
                "(forward, backward, optimizer step)",
                "peak GPU memory for one real validation batch (official val batch size 2048)",
                "seconds per training step (median of the steps after the first)",
                "seconds per epoch and for the 300-epoch budget derived from the measured step time",
            ],
            "command": "python scripts/phase7_gpu_preflight.py",
        })
        OUT.write_text(json.dumps(report, indent=2, default=H.json_default), encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("status", "reason")}, indent=2))
        return

    device = torch.device("cuda:0")
    split = c.load_split(H.SPLIT_CHOICES["official_v1_clean"])
    scratch = ROOT / "results/runs/phase7/preflight"
    _, xp, deviations = H.build_experiment(split, SEED, scratch, devices=1)
    started = time.time()
    loaders = H.build_loaders(xp)
    module = H.build_module(xp, loaders)
    parameters = h.parameter_counts(module)
    torch.cuda.reset_peak_memory_stats(device)
    module.to(device)
    after_weights = torch.cuda.memory_allocated(device)
    steps_per_epoch = len(loaders["train"])
    built = module.optimizer.build(module.parameters(), total_steps=steps_per_epoch * xp.n_epochs)
    optimizer, scheduler = built["optimizer"], built["lr_scheduler"]["scheduler"]

    module.train()
    step_seconds = []
    iterator = iter(loaders["train"])
    for _ in range(TRAIN_STEPS):
        batch = next(iterator).to(device)
        torch.cuda.synchronize(device)
        begin = time.time()
        y_true = batch.data[module.y_name].squeeze(1)
        y_pred = module._transformer_forward(batch, module.forward(batch))
        loss = module.loss(y_pred, y_true)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize(device)
        step_seconds.append(time.time() - begin)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss in preflight")
    train_peak = torch.cuda.max_memory_allocated(device)
    train_reserved = torch.cuda.max_memory_reserved(device)

    torch.cuda.reset_peak_memory_stats(device)
    module.eval()
    val_batch = next(iter(loaders["val"])).to(device)
    torch.cuda.synchronize(device)
    begin = time.time()
    with torch.no_grad():
        module._transformer_forward(val_batch, module.forward(val_batch))
    torch.cuda.synchronize(device)
    val_seconds = time.time() - begin
    val_peak = torch.cuda.max_memory_allocated(device)

    gib = 1024 ** 3
    step = float(np.median(step_seconds[1:])) if len(step_seconds) > 1 else step_seconds[0]
    epoch = steps_per_epoch * step + len(loaders["val"]) * val_seconds
    total = torch.cuda.get_device_properties(device).total_memory
    report.update({
        "status": "GPU preflight executed",
        "deviations_from_official": deviations,
        "parameters": parameters,
        "measured": {
            "device": torch.cuda.get_device_name(device),
            "device_total_memory_gib": total / gib,
            "weights_allocated_gib": after_weights / gib,
            "train_peak_allocated_gib": train_peak / gib,
            "train_peak_reserved_gib": train_reserved / gib,
            "val_batch_peak_allocated_gib": val_peak / gib,
            "train_batch_keystrokes": xp.data.batch_size,
            "val_batch_keystrokes": int(val_batch.data["neuro"].shape[0]),
            "train_step_seconds": step_seconds,
            "median_train_step_seconds_excluding_first": step,
            "val_batch_seconds": val_seconds,
            "steps_per_epoch_single_device": steps_per_epoch,
            "derived_seconds_per_epoch": epoch,
            "derived_hours_for_300_epochs_upper_bound": epoch * xp.n_epochs / 3600,
            "fits_with_headroom": train_reserved < 0.9 * total,
            "seconds_total": time.time() - started,
        },
    })
    OUT.write_text(json.dumps(report, indent=2, default=H.json_default), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "measured")}, indent=2, default=H.json_default))


if __name__ == "__main__":
    main()
