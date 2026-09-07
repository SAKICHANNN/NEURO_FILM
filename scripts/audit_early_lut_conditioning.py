"""Read-only checkpoint-conditioning diagnostic; not a photographic evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_neural_lut import load_model


def main() -> None:
    torch.set_num_threads(4)
    x = torch.linspace(0, 1, 3 * 64 * 64).reshape(1, 3, 64, 64)
    results = []
    for name in (
        "mvp_smoke",
        "challenge_color6_s300",
        "challenge_color6_s800_b12_init",
    ):
        basis, encoder, styles = load_model(
            ROOT / "outputs/neural_lut" / name / "model.pt", torch.device("cpu")
        )
        with torch.no_grad():
            inputs = x.expand(len(styles), -1, -1, -1)
            ids = torch.arange(len(styles))
            weights = encoder(inputs, ids)
            logits = encoder.head(
                torch.cat(
                    [encoder.features(inputs).flatten(1), encoder.style_embedding(ids)],
                    dim=1,
                )
            )
            luts = basis(weights)
        results.append(
            {
                "checkpoint": name,
                "style_weight_span": float((weights - weights[:1]).abs().max()),
                "style_lut_span": float((luts - luts[:1]).abs().max()),
                "logit_min": float(logits.min()),
                "logit_max": float(logits.max()),
                "argmax": weights.argmax(1).tolist(),
                "maximum_weights": weights.max(1).values.tolist(),
            }
        )
    print(
        json.dumps(
            {
                "scope": "fixed synthetic input, stored checkpoints only",
                "rows": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
