"""Phase 4C: overfit the existing ConvCTC model on an easy signal."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from neuroselect.metrics import cer
from neuroselect.models import ConvCTC
from neuroselect.vocab_spanishbcbl import BLANK_ID, decode_ctc, encode_text, VOCAB


def make_signal() -> tuple[np.ndarray, list[tuple[int, int, str]]]:
    signal = np.zeros((300, 306), dtype=np.float32)
    regions = [(0, 80, "a"), (100, 180, "b"), (200, 280, "c")]
    for channel, (start, stop, label) in enumerate(regions):
        signal[start:stop, channel] = 5.0
    return signal, regions


def main() -> None:
    seed = 33
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    signal, regions = make_signal()
    target_text = "abc"
    target = torch.tensor([encode_text(target_text)], dtype=torch.long)
    input_lengths = torch.tensor([signal.shape[0]], dtype=torch.long)
    target_lengths = torch.tensor([target.shape[1]], dtype=torch.long)
    inputs = torch.from_numpy(signal).unsqueeze(0)

    model = ConvCTC(signal.shape[1], len(VOCAB))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    history = []
    first_gradient_norm = None
    parameter_update_norm = None
    epochs = 1000

    model.train()
    for epoch in range(epochs):
        logits = model(inputs)
        log_probs = logits.log_softmax(-1).transpose(0, 1)
        loss = loss_fn(log_probs, target, input_lengths, target_lengths)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if first_gradient_norm is None:
            first_gradient_norm = float(
                torch.linalg.vector_norm(
                    torch.cat([
                        parameter.grad.detach().reshape(-1)
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ])
                )
            )
        optimizer.step()
        if parameter_update_norm is None:
            parameter_update_norm = float(
                torch.linalg.vector_norm(
                    torch.cat([
                        (parameter.detach() - initial).reshape(-1)
                        for parameter, initial in zip(model.parameters(), initial_parameters)
                    ])
                )
            )
        history.append(float(loss.detach()))

    model.eval()
    with torch.no_grad():
        final_logits = model(inputs)[0]
        decoded_ids = final_logits.argmax(-1).tolist()
    decoded = decode_ctc(decoded_ids)

    output = {
        "seed": seed,
        "T": int(signal.shape[0]),
        "channels": int(signal.shape[1]),
        "classes": len(VOCAB),
        "target": target_text,
        "target_length": len(encode_text(target_text)),
        "learning_rate": 1e-2,
        "optimizer": "Adam",
        "epochs": epochs,
        "input_shape": list(inputs.shape),
        "logits_shape": [1, signal.shape[0], len(VOCAB)],
        "ctc_log_probs_shape": [signal.shape[0], 1, len(VOCAB)],
        "target_tensor_shape": list(target.shape),
        "model": "Conv1D + BiGRU + Linear",
        "initial_loss": history[0],
        "minimum_loss": min(history),
        "final_loss": history[-1],
        "first_gradient_norm": first_gradient_norm,
        "parameter_update_norm_after_first_step": parameter_update_norm,
        "decoded": decoded,
        "cer": cer(target_text, decoded),
        "exact_recovery": decoded == target_text,
        "decoded_indices": decoded_ids,
        "loss_history": history,
    }
    result_path = Path("results/synthetic_easy_overfit.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    figure_path = Path("results/figures/debug/synthetic_easy_signal.png")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(signal[:, 0], label="channel 0 (a)")
    axis.plot(signal[:, 1], label="channel 1 (b)")
    axis.plot(signal[:, 2], label="channel 2 (c)")
    for start, stop, label in regions:
        axis.axvspan(start, stop, alpha=0.15)
        axis.text((start + stop) / 2, 5.1, label, ha="center")
    axis.set_xlabel("timestep")
    axis.set_ylabel("activation")
    axis.set_title("Synthetic easy signal encoding abc")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=120)
    plt.close(figure)

    print(json.dumps({
        "decoded": decoded,
        "cer": output["cer"],
        "initial_loss": output["initial_loss"],
        "minimum_loss": output["minimum_loss"],
        "final_loss": output["final_loss"],
        "first_gradient_norm": first_gradient_norm,
        "parameter_update_norm_after_first_step": parameter_update_norm,
    }, indent=2))


if __name__ == "__main__":
    main()
