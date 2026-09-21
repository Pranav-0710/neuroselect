"""Record exact official v1 architecture and feasibility without training."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/official_v1_style_baseline.json"
FIGURE = ROOT / "results/figures/debug/official_v1_style_baseline.png"
PREDICTIONS = ROOT / "results/figures/debug/official_v1_style_predictions.png"


def main() -> None:
    artifact = {
        "experiment": "official_v1_style_baseline_feasibility_audit",
        "official_revision": "5f9889621d0df391c5aab37c996683d308e6e926",
        "training_performed": False,
        "official_source_files_inspected": [
            "vendor/brain2qwerty/brain2qwerty_v1/config/model_config.py",
            "vendor/brain2qwerty/brain2qwerty_v1/config/xp_config.py",
            "vendor/brain2qwerty/brain2qwerty_v1/pl_module.py",
            "neuraltrain.models.simpleconv.SimpleConvTimeAgg",
            "neuraltrain.models.transformer.TransformerEncoder",
        ],
        "exact_official_architecture": {
            "keystroke_encoder": {
                "component": "SimpleConvTimeAgg",
                "input": "(B, 306, 25) native Conv layout",
                "channel_merger": "per-subject 2D Fourier ChannelMerger, 270 virtual channels, Fourier total_dim 2048, dropout 0.2",
                "initial_linear": 512,
                "hidden": 2048,
                "depth": 8,
                "kernel_size": 3,
                "dilation_period": 3,
                "batch_norm": True,
                "activation": "GELU",
                "skip_connections": True,
                "scale": 0.1,
                "input_dropout": 0.2,
                "convolution_dropout": 0.5,
                "time_aggregation": "Bahdanau attention",
                "output": "one 2048-dimensional vector per keystroke window",
            },
            "sentence_model": {
                "component": "x-transformers TransformerEncoder",
                "layers": 4,
                "heads": 2,
                "dimension": 2048,
                "alibi_positional_bias": True,
                "rotary_positional_embedding": False,
                "normalization": "x-transformers default normalization; official config does not enable ScaleNorm/RMSNorm",
                "input": "(sentence, number_of_keystrokes, 2048)",
                "output": "same number of keystroke positions, 2048 dimensions",
            },
            "head_and_objective": {
                "head": "Linear(2048, NUM_CLASSES)",
                "vocabulary": "official BUTTON_MAPPING / NUM_CLASSES classes",
                "loss": "CrossEntropyLoss per keystroke",
                "decoder": "official character metric uses argmax/greedy-style predictions; optional separate KenLM script is not part of base neural training",
                "language_model_in_base_v1": False,
            },
        },
        "official_training_config": {
            "epochs": 300,
            "batch_size": 64,
            "optimizer": "AdamW",
            "learning_rate": 5e-5,
            "scheduler": "OneCycleLR",
            "reported_experiment_lr": 5e-5,
        },
        "feasibility": {
            "dataset": "8 existing trials; official-v1 event tensors (25,306)",
            "cpu": "not practical for the exact architecture on the available laptop",
            "memory": "high risk: 2048-wide 8-layer encoder plus 4-layer 2048-wide Transformer; Adam-style training requires multiple copies of parameters plus activations",
            "parameter_count": "exact instantiation probe in the pinned environment terminated before producing a count; source configuration establishes a very large model dominated by the 2048-wide Transformer",
            "training_status": "not run",
            "blocker": "Exact official v1 uses a large per-keystroke encoder/channel merger and sentence Transformer, while the current local environment cannot safely instantiate the configured model for an eight-trial CPU experiment.",
        },
        "minimal_faithful_alternative": {
            "status": "proposed only; not implemented or trained",
            "structure_preserved": "event encoder -> one vector per event -> sentence-level Transformer -> per-event character head",
            "required_user_decision": "Choose explicit reduced hidden/channel dimensions and record them as a separate approximation; it must not be called exact official v1.",
        },
        "comparison_target": {
            "current_official_v1_event_sequence_ctc": {
                "train_cer_mean": 0.021630707241080422,
                "evaluation_cer_mean": 0.9193899782135077,
            },
            "hierarchical_result": None,
        },
        "figures": [str(FIGURE), str(PREDICTIONS)],
        "verification": {
            "official_revision_inspected": True,
            "official_event_tensors_used": False,
            "timing_gaps_used": False,
            "historical_preprocessing_used": False,
            "existing_convctc_modified": False,
            "training": False,
            "llm": False,
            "evidence_selector": False,
            "additional_data": False,
        },
        "interpretation": "D. Inconclusive: exact hierarchical baseline was not trained because feasibility was not established; no transfer claim is made.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.text(0.02, 0.85, "Official Brain2Qwerty v1 architecture", fontsize=15, weight="bold")
    axis.text(0.05, 0.62, "(25, 306) event → 8-layer 2048-wide Conv + Fourier merger", fontsize=11)
    axis.text(0.05, 0.45, "one 2048-D vector/event → 4-layer Transformer, 2 heads", fontsize=11)
    axis.text(0.05, 0.28, "per-event Linear head + CrossEntropyLoss", fontsize=11)
    axis.text(0.05, 0.08, "Exact training not run: CPU/memory feasibility blocker", color="#c00000", fontsize=11)
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.text(0.05, 0.75, "Hierarchical baseline predictions", fontsize=15, weight="bold")
    axis.text(0.05, 0.52, "No predictions generated", fontsize=14)
    axis.text(0.05, 0.32, "Training was intentionally not run before", fontsize=11)
    axis.text(0.05, 0.22, "establishing exact-model feasibility.", fontsize=11)
    figure.tight_layout()
    figure.savefig(PREDICTIONS, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
