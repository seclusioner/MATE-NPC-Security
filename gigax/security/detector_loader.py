from __future__ import annotations

from pathlib import Path
from typing import Any

from gigax.security.detector import RepresentationRiskDetector


def load_raac_scorers(
    *,
    model: Any,
    tokenizer: Any,
    detector_paths: list[str | Path],
    device: str | None = None,
    threshold_scale: float = 1.0,
    margin: float = 0.0,
    chat_template_mode: str = "tokenizer",
) -> list[RepresentationRiskDetector]:
    """
    Load Original Multi-RAAC representation-reading heads.

    Each detector artifact represents one semantic risk family,
    such as harmfulness or injection/memory manipulation.
    """

    scorers: list[
        RepresentationRiskDetector
    ] = []

    for detector_path in detector_paths:
        scorers.append(
            RepresentationRiskDetector(
                model=model,
                tokenizer=tokenizer,
                detector_path=detector_path,
                device=device,
                threshold_scale=threshold_scale,
                margin=margin,
                chat_template_mode=chat_template_mode,
            )
        )

    return scorers
