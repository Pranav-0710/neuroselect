"""Phase 6 Part 2: minimal smoke test of the exact official Brain2Qwerty v1 model.

On a small deterministic subset of S22 (the first sentence of each of the four
blocks), checks that:

1. official events rebuilt through the official study + SpanishBCBLPreprocessing
   match the stored 5X extraction;
2. batches built from the validated slab equal batches built by the official
   SegmentDataset (neuro, targets, subject ids, channel positions);
3. the exact official model accepts them, yields finite logits and a finite
   loss, and decodes to text through the official character mapping.

It also records the exact parameter count, the transformer's realized settings,
forward cost and memory, and derives the training feasibility estimate.
Forward passes only: training this model is assessed, not attempted.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("phase6_harness", ROOT / "scripts/phase6_official_v1_harness.py")
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)

np, torch, ns = h.np, h.torch, h.ns
OUT_PATH = ROOT / "results/phase6_official_v1_smoke_test.json"
SMOKE_SPLIT = "C"
TRAIN_KEYSTROKES = {"C": 4103, "D": 2040, "E": 2755}
TEST_KEYSTROKES = {"C": 5547, "D": 2792, "E": 2063}


def cpu_description() -> dict:
    command = ["powershell", "-NoProfile", "-Command",
               "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,"
               "NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json"]
    try:
        return json.loads(subprocess.run(command, capture_output=True, text=True, timeout=60).stdout)
    except Exception as error:  # pragma: no cover - descriptive only
        return {"error": str(error)}


def main() -> None:
    report: dict = {"task": "Phase 6 Part 2: official v1 smoke test (forward only)",
                    "official_revision": h.OFFICIAL_REVISION}
    started = time.time()
    report["memory_at_start"] = h.memory()

    cfg, xp = h.official_experiment()
    events = h.build_events(xp)
    report["events"] = {"rows": int(len(events)), **h.check_against_stored_events(events)}

    events, manifest = h.assign_split(events, SMOKE_SPLIT)
    extractors = h.prepare_extractors(xp, events)
    report["memory_after_extractor_prepare"] = h.memory()

    # Deterministic subset: the first sentence (lowest trial) of each block.
    metadata = [json.loads(line) for line in h.METADATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen = []
    for block in sorted({(r["session"], r["block"]) for r in metadata}):
        first = min((r for r in metadata if (r["session"], r["block"]) == block), key=lambda r: r["trial_id"])
        chosen.append(first["sentence_UID"])
    events["split"] = np.where(events["sentence_UID"].astype(str).isin(chosen), "smoke", None)
    segments = h.keystroke_segments(xp, events, "smoke")

    # (2) official SegmentDataset batch vs slab-backed batch, exact.
    official = ns.SegmentDataset(extractors=extractors, segments=segments, remove_incomplete_segments=True)
    official_batch = official.load_all()
    slab = np.load(h.TENSOR_PATH, mmap_mode="r")
    tensors = h.KeystrokeTensors(segments, extractors, slab, h.slab_index())
    parity = {}
    for name, value in official_batch.data.items():
        ours = tensors.data[name]
        parity[name] = {
            "official_shape": list(value.shape),
            "official_dtype": str(value.dtype),
            "slab_shape": list(ours.shape),
            "slab_dtype": str(ours.dtype),
            "exactly_equal": bool(value.shape == ours.shape and value.dtype == ours.dtype
                                  and torch.equal(value, ours)),
            "max_abs_difference": float((value.double() - ours.double()).abs().max()) if value.shape == ours.shape else None,
        }
    positions = tensors.data["channel_positions"]
    report["smoke_subset"] = {
        "sentence_UIDs": chosen,
        "keystrokes": len(tensors),
        "official_segments_kept": len(official.segments),
        "batch_parity": parity,
        "all_inputs_identical_to_official_segment_dataset": all(v["exactly_equal"] for v in parity.values()),
        "subject_ids": sorted({int(v) for v in tensors.data["subject_id"].flatten().tolist()}),
        "channel_positions_identical_across_all_keystrokes": bool((positions == positions[0]).all()),
        "channel_positions_masked_invalid": int((positions[0] == -0.1).all(dim=1).sum()),
        "channel_positions_range": [float(positions[0].min()), float(positions[0].max())],
        "target_classes_official_vs_neuroselect_vocab": "official label = NeuroSelect vocab id - 1 (blank-free)",
    }
    del official, official_batch, extractors
    gc.collect()
    report["memory_before_model"] = h.memory()

    # (3) the exact official model, forward only.
    torch.manual_seed(33)
    module = h.build_module(xp, n_in_channels=int(tensors.data["neuro"].shape[1]))
    loader = tensors.loader(batch_size=xp.data.batch_size)
    h.materialize_lazy_params(module, loader)  # official dummy forward for lazy layers
    report["parameters"] = h.parameter_counts(module)
    report["memory_after_model"] = h.memory()

    transformer = module.transformer
    report["realized_transformer"] = {
        "class": type(transformer).__name__,
        "layers": len(transformer.layers) // 2,
        "heads": transformer.layers[0][1].heads,
        "norm": type(transformer.layers[0][0][0]).__name__,
        "rotary_positional_embedding": type(getattr(transformer, "rotary_pos_emb", None)).__name__,
        "relative_position_bias": type(getattr(transformer, "rel_pos", None)).__name__,
        "attention_dropout": float(transformer.layers[0][1].attend.dropout.p)
        if hasattr(transformer.layers[0][1].attend.dropout, "p") else transformer.layers[0][1].attend.dropout,
    }

    module.eval()
    finite_logits, finite_loss, losses, timings = True, True, [], []
    predicted = np.empty(len(tensors), dtype=np.int64)
    with torch.no_grad():
        for batch_indices in loader.batch_sampler:
            batch = tensors.batch(list(batch_indices))
            begin = time.time()
            logits = module._transformer_forward(batch, module.forward(batch))
            timings.append((time.time() - begin) / len(batch_indices))
            loss = module.loss(logits, batch.data["feature"].squeeze(1))
            finite_logits &= bool(torch.isfinite(logits).all())
            finite_loss &= bool(torch.isfinite(loss))
            losses.append(float(loss))
            predicted[list(batch_indices)] = logits.argmax(dim=1).numpy()
            metric = h.CER()
            metric.update(logits, batch.data["feature"].squeeze(1))
    rows = h.sentence_rows(tensors, predicted)
    report["forward_check"] = {
        "official_batch_size": xp.data.batch_size,
        "batches": len(losses),
        "logits_finite": finite_logits,
        "loss_finite": finite_loss,
        "untrained_cross_entropy": losses,
        "chance_cross_entropy_ln29": float(np.log(h.NUM_CLASSES)),
        "official_cer_metric_ran": True,
        "decoding": "argmax per keystroke -> official CHAR_INDEX; one character per keystroke",
        "decoded_examples_untrained": [{k: row[k] for k in ("block", "target", "decoded", "cer")} for row in rows],
        "forward_seconds_per_keystroke": float(np.median(timings)),
    }

    # Feasibility of training, from measured numbers.
    params = report["parameters"]["total"]
    forward = report["forward_check"]["forward_seconds_per_keystroke"]
    train_seconds_per_keystroke = 3.0 * forward  # forward + backward ~ 3x forward FLOPs
    optimizer_seconds_per_step = 7 * params * 4 / 30e9  # AdamW streams p, g, m, v (~7 tensor passes)
    steps = {split: -(-n // xp.data.batch_size) for split, n in TRAIN_KEYSTROKES.items()}
    epoch_seconds = {split: TRAIN_KEYSTROKES[split] * train_seconds_per_keystroke
                     + steps[split] * optimizer_seconds_per_step for split in steps}
    static_training_gb = params * 4 * 4 / 1024 ** 3  # weights + grads + AdamW exp_avg + exp_avg_sq
    per_subject_tables = report["parameters"]["channel_merger"] + report["parameters"]["subject_layers"]
    used_by_one_subject = params - per_subject_tables * (199 / 200)
    report["feasibility"] = {
        "basis": "measured forward cost and exact parameter count; training cost extrapolated (backward ~2x forward)",
        "exact_parameters": params,
        "per_subject_table_parameters": per_subject_tables,
        "per_subject_table_slots": 200,
        "parameters_reachable_by_one_subject": int(used_by_one_subject),
        "per_subject_note": (
            "The channel-merger heads and the subject layers are tables over 200 subject slots "
            "(neuraltrain defaults; the official config sets merger per_subject=True and "
            "subject_layers_config={}, which parses to an enabled SubjectLayers). S22 reaches one slot."
        ),
        "training_static_memory_gb": static_training_gb,
        "training_static_memory_breakdown": "4 fp32 copies of every parameter: weights, gradients, AdamW exp_avg, exp_avg_sq",
        "process_baseline_commit_gb_after_imports": report["memory_at_start"]["process_private_commit_gb"],
        "estimated_training_process_commit_gb": static_training_gb + report["memory_at_start"]["process_private_commit_gb"] + 1.0,
        "system_available_commit_gb_with_model_loaded": report["memory_after_model"]["system_available_commit_gb"],
        "system_available_physical_gb_with_model_loaded": report["memory_after_model"]["system_available_physical_gb"],
        "system_total_physical_gb": report["memory_after_model"]["system_total_physical_gb"],
        "estimated_seconds_per_epoch": epoch_seconds,
        "estimated_hours_per_run": {
            schedule: {split: epochs * seconds / 3600 for split, seconds in epoch_seconds.items()}
            for schedule, epochs in (("official_300_epochs", 300), ("diagnostic_100_epochs", 100))
        },
        "epochs_fitting_2_hours": {split: int(7200 // seconds) for split, seconds in epoch_seconds.items()},
        "optimizer_steps_per_epoch": steps,
    }
    report["environment"] = {
        "python_executable": os.sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cpu": cpu_description(),
        "overlay": str(h.OVERLAY / "site"),
    }
    report["memory_at_end"] = h.memory()
    report["seconds"] = time.time() - started
    report["status"] = "PASS" if (report["events"]["identical_timeline_start_button"]
                                  and report["smoke_subset"]["all_inputs_identical_to_official_segment_dataset"]
                                  and finite_logits and finite_loss) else "FAIL"
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "events", "parameters", "realized_transformer",
                                             "forward_check", "feasibility")}, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
