from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.calibration import (
    binary_metrics,
    choose_youden_threshold,
)
from training.common import (
    EPS,
    load_bundle,
    load_cache,
    normalized_margin,
    risk_family_subset,
    score_mean_difference,
    validate_layers,
)


RISK_FAMILIES = (
    "harm",
    "injection",
)


def parse_layers(
    value: str,
) -> list[int]:
    layers = []

    for item in value.split(","):
        item = item.strip()

        if not item:
            continue

        layers.append(
            int(item)
        )

    layers = sorted(
        set(layers)
    )

    if not layers:
        raise ValueError(
            "No layers supplied."
        )

    return layers


def train_mean_difference_head(
    train_cache: dict[str, Any],
    validation_cache: dict[str, Any],
    *,
    risk_family: str,
    layers: list[int],
    name: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    """
    Train one category-specific Original Multi-RAAC head.

    Formal experiment behavior:
    - category-specific positive/negative examples
    - unit-normalized mean-difference direction
    - midpoint decision bias
    - validation-only Youden-J calibration
    """

    (
        train_features,
        train_labels,
        _,
    ) = risk_family_subset(
        train_cache,
        risk_family,
    )

    (
        validation_features,
        validation_labels,
        _,
    ) = risk_family_subset(
        validation_cache,
        risk_family,
    )

    validate_layers(
        train_cache,
        layers,
    )

    validate_layers(
        validation_cache,
        layers,
    )

    positive_mask = (
        torch.from_numpy(
            train_labels == 1
        )
    )

    negative_mask = (
        torch.from_numpy(
            train_labels == 0
        )
    )

    vec: dict[
        int,
        torch.Tensor,
    ] = {}

    bias: dict[
        int,
        float,
    ] = {}

    for layer in layers:
        positive_mean = (
            train_features[
                positive_mask,
                layer,
                :,
            ]
            .mean(dim=0)
        )

        negative_mean = (
            train_features[
                negative_mask,
                layer,
                :,
            ]
            .mean(dim=0)
        )

        direction = (
            positive_mean
            - negative_mean
        )

        direction = (
            direction
            / (
                torch.linalg
                .vector_norm(
                    direction
                )
                + EPS
            )
        )

        # Preserve the formal experiment implementation.
        midpoint = (
            positive_mean
            + negative_mean
        ) / 2.0

        vec[layer] = (
            direction
            .detach()
            .cpu()
            .float()
        )

        bias[layer] = float(
            torch.dot(
                midpoint,
                direction,
            ).item()
        )

    provisional = {
        "method": (
            "mean_difference"
        ),
        "layers": layers,
        "vec": vec,
        "bias": bias,
    }

    validation_scores = (
        score_mean_difference(
            validation_features,
            provisional,
        )
    )

    (
        threshold,
        calibration,
    ) = choose_youden_threshold(
        validation_labels,
        validation_scores,
    )

    score_scale = max(
        float(
            np.std(
                validation_scores
            )
        ),
        EPS,
    )

    bundle = {
        "format_version": 1,
        "method": (
            "mean_difference"
        ),
        "name": name,
        "risk_type": (
            f"{risk_family}_detector"
        ),
        "layers": layers,
        "vec": vec,
        "bias": bias,
        "threshold": float(
            threshold
        ),
        "score_scale": float(
            score_scale
        ),
        "threshold_rule": (
            "Youden J on category validation"
        ),
        "train_n": int(
            len(train_labels)
        ),
        "validation_n": int(
            len(validation_labels)
        ),
        "train_manifest_sha256": (
            train_cache[
                "manifest_sha256"
            ]
        ),
        "validation_manifest_sha256": (
            validation_cache[
                "manifest_sha256"
            ]
        ),
    }

    diagnostics = {
        "initial_calibration": (
            calibration
        ),
        "initial_validation_metrics": (
            binary_metrics(
                validation_labels,
                validation_scores,
                threshold,
            )
        ),
    }

    return (
        bundle,
        diagnostics,
    )


def recalibrate_serialized_head(
    *,
    bundle_path: Path,
    validation_cache: dict[str, Any],
    risk_family: str,
) -> tuple[
    dict[str, Any],
    np.ndarray,
]:
    """
    Reload the serialized artifact and recalibrate using the exact
    public runtime-equivalent score path.
    """

    bundle = load_bundle(
        bundle_path
    )

    (
        validation_features,
        validation_labels,
        _,
    ) = risk_family_subset(
        validation_cache,
        risk_family,
    )

    scores = score_mean_difference(
        validation_features,
        bundle,
    )

    (
        threshold,
        calibration,
    ) = choose_youden_threshold(
        validation_labels,
        scores,
    )

    score_scale = max(
        float(
            np.std(scores)
        ),
        EPS,
    )

    bundle["threshold"] = float(
        threshold
    )

    bundle["score_scale"] = float(
        score_scale
    )

    bundle["runtime_calibration"] = {
        "criterion": "Youden J",
        "comparison": (
            "score >= threshold"
        ),
        "runtime_scorer": (
            "serialized bundle"
        ),
        **calibration,
        "metrics": binary_metrics(
            validation_labels,
            scores,
            threshold,
        ),
    }

    torch.save(
        bundle,
        bundle_path,
    )

    return (
        bundle,
        scores,
    )


def build_original_multiraac(
    *,
    train_cache: dict[str, Any],
    validation_cache: dict[str, Any],
    output_dir: Path,
    layers: list[int],
) -> dict[str, Any]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_layers(
        train_cache,
        layers,
    )

    validate_layers(
        validation_cache,
        layers,
    )

    if (
        train_cache["features"]
        .shape[2]
        != validation_cache[
            "features"
        ].shape[2]
    ):
        raise ValueError(
            "Train and validation "
            "hidden sizes differ."
        )

    bundles = {}
    initial_diagnostics = {}

    for family in RISK_FAMILIES:
        bundle, diagnostics = (
            train_mean_difference_head(
                train_cache,
                validation_cache,
                risk_family=family,
                layers=layers,
                name=(
                    f"multiraac_{family}"
                ),
            )
        )

        path = (
            output_dir
            / f"{family}_detector.pt"
        )

        torch.save(
            bundle,
            path,
        )

        (
            bundles[family],
            _,
        ) = recalibrate_serialized_head(
            bundle_path=path,
            validation_cache=(
                validation_cache
            ),
            risk_family=family,
        )

        initial_diagnostics[
            family
        ] = diagnostics

    validation_features = (
        validation_cache[
            "features"
        ]
    )

    validation_labels = np.asarray(
        [
            int(row["label"])
            for row
            in validation_cache["rows"]
        ],
        dtype=np.int64,
    )

    head_scores = {
        family: (
            score_mean_difference(
                validation_features,
                bundles[family],
            )
        )
        for family
        in RISK_FAMILIES
    }

    head_margins = {
        family: normalized_margin(
            head_scores[family],
            bundles[family][
                "threshold"
            ],
            bundles[family][
                "score_scale"
            ],
        )
        for family
        in RISK_FAMILIES
    }

    fused_margin = np.maximum.reduce(
        [
            head_margins[family]
            for family
            in RISK_FAMILIES
        ]
    )

    fusion_threshold = 0.0

    runtime = {
        "format_version": 1,
        "name": (
            "Original Multi-RAAC"
        ),
        "method": (
            "multiraac_or_fusion"
        ),
        "layers": layers,
        "risk_families": list(
            RISK_FAMILIES
        ),
        "fusion": (
            "max normalized head margin"
        ),
        "fusion_threshold": (
            fusion_threshold
        ),
        "comparison": (
            "fused_margin >= "
            "fusion_threshold"
        ),
        "head_artifacts": {
            family: (
                f"{family}_detector.pt"
            )
            for family
            in RISK_FAMILIES
        },
        "runtime_calibration": {
            "criterion": (
                "head-wise Youden J"
            ),
            "fusion_rule": (
                "safety-first OR"
            ),
            "fusion_threshold": (
                fusion_threshold
            ),
            "validation_metrics": (
                binary_metrics(
                    validation_labels,
                    fused_margin,
                    fusion_threshold,
                )
            ),
            "head_thresholds": {
                family: float(
                    bundles[family][
                        "threshold"
                    ]
                )
                for family
                in RISK_FAMILIES
            },
            "head_score_scales": {
                family: float(
                    bundles[family][
                        "score_scale"
                    ]
                )
                for family
                in RISK_FAMILIES
            },
        },
        "initial_training_diagnostics": (
            initial_diagnostics
        ),
        "train_manifest_sha256": (
            train_cache[
                "manifest_sha256"
            ]
        ),
        "validation_manifest_sha256": (
            validation_cache[
                "manifest_sha256"
            ]
        ),
    }

    system_path = (
        output_dir
        / "system.json"
    )

    system_path.write_text(
        json.dumps(
            runtime,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return runtime


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build Original Multi-RAAC "
            "mean-difference detectors."
        )
    )

    parser.add_argument(
        "--train-cache",
        required=True,
    )

    parser.add_argument(
        "--validation-cache",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--layers",
        default="24,28,30",
        help=(
            "Comma-separated decoder "
            "layer indices."
        ),
    )

    args = parser.parse_args()

    layers = parse_layers(
        args.layers
    )

    train_cache = load_cache(
        args.train_cache
    )

    validation_cache = (
        load_cache(
            args.validation_cache
        )
    )

    runtime = (
        build_original_multiraac(
            train_cache=train_cache,
            validation_cache=(
                validation_cache
            ),
            output_dir=Path(
                args.output_dir
            ),
            layers=layers,
        )
    )

    print(
        json.dumps(
            runtime,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "[DONE]",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
