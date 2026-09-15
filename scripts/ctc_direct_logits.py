"""Phase 4B: verify CTC optimization using trainable logits only."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from neuroselect.metrics import cer
from neuroselect.vocab_spanishbcbl import (
    BLANK_ID,
    VOCAB,
    decode_ctc,
    encode_text,
)


def train_direct_logits(
    text: str,
    timesteps: int,
    max_iterations: int = 2000,
    seed: int = 33,
) -> dict:
    torch.manual_seed(seed)
    target_ids = encode_text(text)
    classes = len(VOCAB)

    # CTC expects log probabilities as (T, N, C), targets as (N, S).
    logits = torch.nn.Parameter(torch.randn(timesteps, 1, classes))
    targets = torch.tensor([target_ids], dtype=torch.long)
    input_lengths = torch.tensor([timesteps], dtype=torch.long)
    target_lengths = torch.tensor([len(target_ids)], dtype=torch.long)
    loss_fn = torch.nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    optimizer = torch.optim.Adam([logits], lr=0.2)

    initial_loss = None
    minimum_loss = float("inf")
    gradient_norm = None
    final_loss = None
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        log_probs = logits.log_softmax(-1)
        loss = loss_fn(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )
        if initial_loss is None:
            initial_loss = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gradient_norm is None:
            gradient_norm = float(logits.grad.detach().norm())
        optimizer.step()

        final_loss = float(loss.detach())
        minimum_loss = min(minimum_loss, final_loss)
        iterations = iteration
        with torch.no_grad():
            decoded = decode_ctc(logits[:, 0].argmax(-1).tolist())
        if decoded == text:
            break

    with torch.no_grad():
        decoded_ids = logits[:, 0].argmax(-1).tolist()
        decoded = decode_ctc(decoded_ids)

    return {
        "target": text,
        "T": timesteps,
        "batch_size": 1,
        "classes": classes,
        "target_length": len(target_ids),
        "blank_index": BLANK_ID,
        "logits_shape": [timesteps, 1, classes],
        "targets_shape": [1, len(target_ids)],
        "log_probs_shape": [timesteps, 1, classes],
        "logits_requires_grad": True,
        "loss_requires_grad": True,
        "initial_loss": initial_loss,
        "minimum_loss": minimum_loss,
        "final_loss": final_loss,
        "gradient_norm_first_backward": gradient_norm,
        "iterations": iterations,
        "decoded": decoded,
        "cer": cer(text, decoded),
        "exact_recovery": decoded == text,
        "decoded_indices": decoded_ids,
    }


def main() -> None:
    cases = {
        "abc": 30,
        "hello": 50,
        "hi there": 70,
        "la tasa": 80,
    }
    results = {
        text: train_direct_logits(text, timesteps)
        for text, timesteps in cases.items()
    }
    output = Path("results/ctc_direct_logits_test.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({
        text: {
            "decoded": result["decoded"],
            "cer": result["cer"],
            "iterations": result["iterations"],
            "exact_recovery": result["exact_recovery"],
        }
        for text, result in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
