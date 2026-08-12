from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gigax.config import (
    MATEConfig,
    load_mate_config,
)
from gigax.runtime import (
    NPCSecurityRuntime,
)
from gigax.security.action_registry import (
    ActionRegistry,
)
from gigax.security.detector import (
    RepresentationRiskDetector,
)
from gigax.security.sparse_multiraac import (
    SharedSparseMultiRAACController,
)
from gigax.security.raac import (
    MultiRAACController,
)


def _load_system_json(
    config: MATEConfig,
) -> dict[str, Any]:
    value = (
        config.layer1.system_path
    )

    if not value:
        raise ValueError(
            "lowfpr_sparse requires "
            "layer1.system in config."
        )

    path = config.resolve_path(
        value
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Sparse system config "
            f"not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _load_detector_paths(
    config: MATEConfig,
) -> list[
    tuple[Any, Path]
]:
    result = []

    for detector_config in (
        config.layer1.detectors
    ):
        if not (
            detector_config.enabled
        ):
            continue

        path = config.resolve_path(
            detector_config.path
        )

        if not path.exists():
            raise FileNotFoundError(
                "Detector artifact "
                f"not found: {path}"
            )

        result.append(
            (
                detector_config,
                path,
            )
        )

    if not result:
        raise ValueError(
            "No enabled Layer-1 "
            "detectors configured."
        )

    return result


def load_mate(
    config_path: str | Path,
) -> NPCSecurityRuntime:
    """
    Build complete MATE runtime from YAML.

    Supported Layer-1 backends:

        original_multiraac
        lowfpr_sparse

    ATP, SCD and TEG remain unchanged when Layer 1 changes.
    """

    config = load_mate_config(
        config_path
    )

    registry = (
        ActionRegistry.from_config(
            catalog=(
                config
                .actions
                .catalog
            ),
            fallback_action=(
                config
                .actions
                .fallback_action
            ),
        )
    )

    runtime = NPCSecurityRuntime(
        model_id=(
            config.model.model_id
        ),
        dtype=(
            config.model.dtype
        ),
        device_map=(
            config.model
            .device_map
        ),
        action_registry=registry,
        default_target=(
            config.actions
            .default_target
        ),
        default_defense=(
            config.runtime.defense
        ),
    )

    detector_specs = (
        _load_detector_paths(
            config
        )
    )

    scorers = []

    # =========================================================
    # Original Multi-RAAC
    # =========================================================
    if (
        config.layer1.type
        == "original_multiraac"
    ):
        for (
            detector_config,
            detector_path,
        ) in detector_specs:
            scorers.append(
                RepresentationRiskDetector(
                    model=runtime.model,
                    tokenizer=(
                        runtime.tokenizer
                    ),
                    detector_path=(
                        detector_path
                    ),
                    threshold_scale=(
                        detector_config
                        .threshold_scale
                    ),
                    margin=(
                        detector_config
                        .margin
                    ),
                    chat_template_mode=(
                        config.model
                        .detector_chat_template
                    ),
                )
            )

        controller = (
            MultiRAACController(
                scorers,
                fusion_threshold=0.0,
                system_name=(
                    "original_multiraac"
                ),
            )
        )

        detector_methods = sorted(
            {
                str(
                    scorer.bundle.get(
                        "method",
                        "",
                    )
                )
                for scorer in scorers
                if scorer.bundle.get(
                    "method"
                )
            }
        )

        detector_layers = sorted(
            {
                int(layer)
                for scorer in scorers
                for layer in (
                    scorer.bundle.get(
                        "layers",
                        []
                    )
                )
            }
        )

        runtime.layer1_system = {
            "type": (
                "original_multiraac"
            ),
            "name": (
                "Original Multi-RAAC"
            ),
            "method": (
                detector_methods[0]
                if len(detector_methods) == 1
                else "+".join(
                    detector_methods
                )
            ),
            "layers": (
                detector_layers
            ),
            "fusion_threshold": 0.0,
        }

    # =========================================================
    # Low-FPR Sparse Multi-RAAC
    # =========================================================
    elif (
        config.layer1.type
        == "lowfpr_sparse"
    ):
        system = (
            _load_system_json(
                config
            )
        )

        method = str(
            system.get(
                "method",
                "",
            )
        )

        if (
            "sparse_multiraac"
            not in method
        ):
            raise ValueError(
                "Sparse system.json has "
                f"unexpected method={method}"
            )

        # The calibrated Sparse system.json and its head
        # artifacts form one deployment operating point.
        #
        # Per-head threshold scaling/margins would invalidate the
        # serialized fusion calibration, so public deployment requires
        # the calibrated head thresholds to remain unchanged.
        modified_heads = [
            detector_config.path
            for (
                detector_config,
                _,
            ) in detector_specs
            if (
                abs(
                    detector_config.threshold_scale
                    - 1.0
                )
                > 1.0e-12
                or abs(
                    detector_config.margin
                )
                > 1.0e-12
            )
        ]

        if modified_heads:
            raise ValueError(
                "lowfpr_sparse requires "
                "threshold_scale=1.0 and margin=0.0 "
                "for every head because system.json "
                "contains a calibrated fusion operating point. "
                f"Modified heads: {modified_heads}"
            )

        if "fusion_threshold" not in system:
            raise KeyError(
                "Sparse system.json is missing "
                "'fusion_threshold'."
            )

        fusion_threshold = float(
            system["fusion_threshold"]
        )

        controller = (
            SharedSparseMultiRAACController(
                model=runtime.model,
                tokenizer=(
                    runtime.tokenizer
                ),
                bundle_paths=[
                    detector_path
                    for (
                        _,
                        detector_path,
                    ) in detector_specs
                ],
                fusion_threshold=(
                    fusion_threshold
                ),
                chat_template_mode=(
                    config.model
                    .detector_chat_template
                ),
            )
        )

        runtime.layer1_system = (
            system
        )

    else:
        raise ValueError(
            "Unsupported Layer-1 "
            f"type={config.layer1.type}"
        )

    runtime.attach_raac_controller(
        controller
    )

    runtime.mate_config = config

    return runtime
