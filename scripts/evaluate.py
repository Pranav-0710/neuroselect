"""Evaluate a saved compact CTC baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from neuroselect.data import NeuralTextDataset, collate_batch
from neuroselect.metrics import cer, edit_distance, wer
from neuroselect.models import ConvCTC
from neuroselect.vocab import VOCAB, decode_ctc
from neuroselect.vocab_spanishbcbl import VOCAB as SPANISH_VOCAB, decode_ctc as decode_spanish


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default="results/baseline_metrics.json")
    parser.add_argument("--examples", default="results/example_predictions.txt")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = NeuralTextDataset(args.manifest, "test")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    use_spanish = data.records[0].get("vocab") == "spanishbcbl"
    model = ConvCTC(
        data[0]["signal"].shape[1],
        len(SPANISH_VOCAB) if use_spanish else len(VOCAB),
        checkpoint["hidden"],
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    predictions = []
    loader = DataLoader(data, 4, shuffle=False, collate_fn=collate_batch)
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["signals"].to(device))
            ids = logits.argmax(-1).cpu()
            for index, meta in enumerate(batch["meta"]):
                decoder = decode_spanish if use_spanish else decode_ctc
                prediction = decoder(
                    ids[index, : batch["input_lengths"][index]].tolist()
                )
                reference = meta["text"].lower()
                predictions.append(
                    {
                        "id": meta["id"],
                        "reference": reference,
                        "prediction": prediction,
                        "edit_distance": edit_distance(reference, prediction),
                        "cer": cer(reference, prediction),
                        "wer": wer(reference, prediction),
                    }
                )
    metrics = {
        "num_examples": len(predictions),
        "cer": sum(x["cer"] for x in predictions) / max(len(predictions), 1),
        "wer": sum(x["wer"] for x in predictions) / max(len(predictions), 1),
        "predictions": predictions,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ranked = sorted(predictions, key=lambda x: x["cer"])
    selected = ranked[:3] + ranked[len(ranked) // 2 : len(ranked) // 2 + 3] + ranked[-3:]
    Path(args.examples).write_text(
        "\n".join(
            f'{row["id"]}\n  reference: {row["reference"]}\n  prediction: {row["prediction"]}\n  CER: {row["cer"]:.4f}'
            for row in selected
        ),
        encoding="utf-8",
    )
    print(json.dumps({k: metrics[k] for k in ("num_examples", "cer", "wer")}, indent=2))


if __name__ == "__main__":
    main()
