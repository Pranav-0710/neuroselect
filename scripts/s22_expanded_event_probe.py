"""Step 4: expanded event-level linear probe on official-v1 event tensors.

Repeats the 5H event-information probe on the full four-block dataset. Every
event is the flattened 25 x 306 official-v1 tensor (7,650 features). Scaling,
PCA and the classifier are fitted inside the training side of each split only;
held-out blocks never touch the fit. This is a diagnostic of per-event class
information, not a brain-to-text decoder.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TENSOR_PATH = ROOT / "data/processed/s22_official_v1/events.npy"
INDEX_PATH = ROOT / "data/processed/s22_official_v1/index.jsonl"
MANIFEST_DIR = ROOT / "data/manifests"
OUT_PATH = ROOT / "results/s22_expanded_event_probe.json"
FIGURE_PATH = ROOT / "results/figures/debug/s22_expanded_event_probe.png"

SPLIT_FILES = {
    "A": "split_A_same_session.json",
    "B": "split_B_same_session.json",
    "C": "split_C_sentence_disjoint.json",
    "D": "split_D_cross_session_sentence_disjoint.json",
    "E": "split_E_cross_session_sentence_disjoint.json",
}
PRIMARY = "C"
PCA_VARIANCE = 0.95
PROBE_SEED = 33


def load_index() -> list[dict]:
    return [json.loads(line) for line in INDEX_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def metrics(true: np.ndarray, predicted: np.ndarray, majority: int) -> dict:
    baseline = np.full_like(true, majority)
    return {
        "events": int(len(true)),
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "majority_baseline_accuracy": float(accuracy_score(true, baseline)),
        "accuracy_minus_majority": float(accuracy_score(true, predicted) - accuracy_score(true, baseline)),
        "distinct_true_classes": int(len(np.unique(true))),
        "distinct_predicted_classes": int(len(np.unique(predicted))),
    }


def main() -> None:
    index = load_index()
    slab = np.load(TENSOR_PATH, mmap_mode="r")
    by_sentence: dict[str, list[dict]] = {}
    for row in index:
        by_sentence.setdefault(row["sentence_UID"], []).append(row)

    reports = {}
    for split, filename in SPLIT_FILES.items():
        manifest = json.loads((MANIFEST_DIR / filename).read_text(encoding="utf-8"))
        train_rows = [row for uid in manifest["train_sentence_UIDs"] for row in by_sentence[uid]]
        test_rows = [row for uid in manifest["test_sentence_UIDs"] for row in by_sentence[uid]]
        train_index = np.array([row["row"] for row in train_rows])
        test_index = np.array([row["row"] for row in test_rows])
        train_x = np.asarray(slab[train_index]).reshape(len(train_index), -1)
        test_x = np.asarray(slab[test_index]).reshape(len(test_index), -1)
        train_y = np.array([row["label_id"] for row in train_rows])
        test_y = np.array([row["label_id"] for row in test_rows])

        started = time.time()
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=PROBE_SEED)),
            ("classifier", LogisticRegression(
                class_weight="balanced", max_iter=2000, random_state=PROBE_SEED, n_jobs=-1)),
        ])
        pipeline.fit(train_x, train_y)
        fit_seconds = time.time() - started
        pca = pipeline.named_steps["pca"]
        train_predicted = pipeline.predict(train_x)
        test_predicted = pipeline.predict(test_x)
        values, counts = np.unique(train_y, return_counts=True)
        majority = int(values[counts.argmax()])

        folds = {}
        for block in sorted({f"session{row['session']}/{row['block']}" for row in test_rows}):
            mask = np.array([f"session{row['session']}/{row['block']}" == block for row in test_rows])
            folds[block] = metrics(test_y[mask], test_predicted[mask], majority)

        train_classes = set(int(v) for v in np.unique(train_y))
        test_classes = set(int(v) for v in np.unique(test_y))
        reports[split] = {
            "split_name": manifest["split_name"],
            "classification": manifest["classification"],
            "train_blocks": manifest["train_blocks"],
            "test_blocks": manifest["test_blocks"],
            "primary": split == PRIMARY,
            "features_per_event": int(train_x.shape[1]),
            "pca_components_retained": int(pca.n_components_),
            "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
            "fit_seconds": fit_seconds,
            "train": metrics(train_y, train_predicted, majority),
            "test": metrics(test_y, test_predicted, majority),
            "per_fold_test": folds,
            "majority_class_id": majority,
            "class_availability": {
                "classes_in_train": sorted(train_classes),
                "classes_in_test": sorted(test_classes),
                "classes_in_test_absent_from_train": sorted(test_classes - train_classes),
                "classes_in_train_absent_from_test": sorted(train_classes - test_classes),
                "train_class_counts": {str(int(v)): int(c) for v, c in zip(*np.unique(train_y, return_counts=True))},
                "test_class_counts": {str(int(v)): int(c) for v, c in zip(*np.unique(test_y, return_counts=True))},
            },
        }
        print(json.dumps({"split": split, "test": reports[split]["test"],
                          "pca": reports[split]["pca_components_retained"],
                          "seconds": round(fit_seconds, 1)}), flush=True)
        del train_x, test_x

    artifact = {
        "task": "Step 4 expanded event-level linear probe",
        "representation": "official-v1 event tensor (25, 306) flattened to 7650 features",
        "pipeline": "StandardScaler -> PCA(95% variance) -> LogisticRegression(class_weight='balanced')",
        "fit_scope": "training side of each split only; held-out blocks never enter any fit",
        "primary_split": PRIMARY,
        "primary_split_reason": "C is sentence-disjoint and pools both sessions and all four blocks.",
        "splits_run": sorted(SPLIT_FILES),
        "splits_not_run": ["F", "G"],
        "splits_not_run_reason": "F and G are cross-session with sentence overlap; they are not valid for this diagnostic.",
        "seed": PROBE_SEED,
        "reports": reports,
        "interpretation_guard": (
            "Per-event classification above a majority baseline shows that keystroke-locked event "
            "windows carry some class-linked information. It is not evidence of brain-to-text decoding."
        ),
        "verification": {
            "official_v1_tensors_used": True,
            "preprocessing_fitted_on_training_side_only": True,
            "held_out_blocks_used_for_fitting": False,
            "llm_used": False,
            "evidence_selector_used": False,
        },
    }
    OUT_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")

    order = [PRIMARY] + [split for split in sorted(SPLIT_FILES) if split != PRIMARY]
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 5.5))
    positions = np.arange(len(order))
    width = 0.27
    left.bar(positions - width, [reports[s]["test"]["accuracy"] for s in order], width, label="test accuracy", color="#3a6ea5")
    left.bar(positions, [reports[s]["test"]["balanced_accuracy"] for s in order], width, label="test balanced accuracy", color="#6fa8dc")
    left.bar(positions + width, [reports[s]["test"]["majority_baseline_accuracy"] for s in order], width, label="majority baseline", color="#bbbbbb")
    for position, split in zip(positions, order):
        left.text(position - width, reports[split]["test"]["accuracy"], f"{reports[split]['test']['accuracy']:.3f}",
                  ha="center", va="bottom", fontsize=8)
    left.set_xticks(positions, [f"{s}{' (primary)' if s == PRIMARY else ''}" for s in order])
    left.set_ylabel("score")
    left.set_title("Held-out event classification by split")
    left.legend(fontsize=8)

    right.bar(positions, [reports[s]["test"]["macro_f1"] for s in order], 0.5, color="#c8553d")
    for position, split in zip(positions, order):
        right.text(position, reports[split]["test"]["macro_f1"], f"{reports[split]['test']['macro_f1']:.3f}",
                   ha="center", va="bottom", fontsize=8)
    right.set_xticks(positions, order)
    right.set_ylabel("macro F1")
    right.set_title("Held-out macro F1 (29-class vocabulary)")
    figure.suptitle("Expanded event-level linear probe: 9,650 events, 4 blocks, 2 sessions", fontsize=12)
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)
    print(json.dumps({"artifact": str(OUT_PATH.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
