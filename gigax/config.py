from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    dtype: str = "bfloat16"
    device_map: str = "auto"
    detector_chat_template: str = (
        "tokenizer"
    )


@dataclass(frozen=True)
class DetectorConfig:
    path: str
    enabled: bool = True
    threshold_scale: float = 1.0
    margin: float = 0.0


@dataclass(frozen=True)
class Layer1Config:
    type: str
    detectors: list[
        DetectorConfig
    ]

    # Used by Sparse Multi-RAAC.
    system_path: str = ""


@dataclass(frozen=True)
class ActionConfig:
    default_target: str
    fallback_action: str
    catalog: list[
        dict[str, Any]
    ]


@dataclass(frozen=True)
class RuntimeConfig:
    defense: str = (
        "raac_multiaction"
    )


@dataclass(frozen=True)
class MATEConfig:
    source_path: Path
    model: ModelConfig
    layer1: Layer1Config
    actions: ActionConfig
    runtime: RuntimeConfig

    @property
    def base_dir(
        self,
    ) -> Path:
        return (
            self.source_path.parent
        )

    @property
    def detectors(
        self,
    ) -> list[DetectorConfig]:
        """
        Backward-compatible accessor.
        """
        return self.layer1.detectors

    def resolve_path(
        self,
        path: str | Path,
    ) -> Path:
        candidate = Path(path)

        if candidate.is_absolute():
            return candidate

        return (
            self.base_dir
            / candidate
        ).resolve()


def _parse_detectors(
    rows: list[
        dict[str, Any]
    ],
) -> list[DetectorConfig]:
    return [
        DetectorConfig(
            path=str(
                entry["path"]
            ),
            enabled=bool(
                entry.get(
                    "enabled",
                    True,
                )
            ),
            threshold_scale=float(
                entry.get(
                    "threshold_scale",
                    1.0,
                )
            ),
            margin=float(
                entry.get(
                    "margin",
                    0.0,
                )
            ),
        )
        for entry in rows
    ]


def load_mate_config(
    path: str | Path,
) -> MATEConfig:
    path = Path(path).resolve()

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        raw = yaml.safe_load(f)

    if not isinstance(
        raw,
        dict,
    ):
        raise ValueError(
            "MATE config must contain "
            "a YAML mapping."
        )

    model_raw = raw.get(
        "model",
        {},
    )

    action_raw = raw.get(
        "actions",
        {},
    )

    runtime_raw = raw.get(
        "runtime",
        {},
    )

    # ---------------------------------------------------------
    # New schema
    # ---------------------------------------------------------
    if "layer1" in raw:
        layer1_raw = (
            raw.get("layer1")
            or {}
        )

        layer1_type = str(
            layer1_raw.get(
                "type",
                "original_multiraac",
            )
        ).lower()

        detector_rows = list(
            layer1_raw.get(
                "detectors",
                [],
            )
        )

        system_path = str(
            layer1_raw.get(
                "system",
                "",
            )
        )

    # ---------------------------------------------------------
    # Legacy schema compatibility
    # ---------------------------------------------------------
    else:
        layer1_type = (
            "original_multiraac"
        )

        detector_rows = list(
            raw.get(
                "detectors",
                [],
            )
        )

        system_path = ""

    supported_layer1 = {
        "original_multiraac",
        "lowfpr_sparse",
    }

    if (
        layer1_type
        not in supported_layer1
    ):
        raise ValueError(
            "Unsupported layer1.type="
            f"{layer1_type}. "
            f"Supported: "
            f"{sorted(supported_layer1)}"
        )

    model = ModelConfig(
        model_id=str(
            model_raw["id"]
        ),
        dtype=str(
            model_raw.get(
                "dtype",
                "bfloat16",
            )
        ),
        device_map=str(
            model_raw.get(
                "device_map",
                "auto",
            )
        ),
        detector_chat_template=str(
            model_raw.get(
                "detector_chat_template",
                "tokenizer",
            )
        ),
    )

    layer1 = Layer1Config(
        type=layer1_type,
        detectors=_parse_detectors(
            detector_rows
        ),
        system_path=system_path,
    )

    actions = ActionConfig(
        default_target=str(
            action_raw.get(
                "default_target",
                "Aldren",
            )
        ),
        fallback_action=str(
            action_raw.get(
                "fallback_action",
                "Say",
            )
        ),
        catalog=list(
            action_raw.get(
                "catalog",
                [],
            )
        ),
    )

    runtime = RuntimeConfig(
        defense=str(
            runtime_raw.get(
                "defense",
                "raac_multiaction",
            )
        )
    )

    return MATEConfig(
        source_path=path,
        model=model,
        layer1=layer1,
        actions=actions,
        runtime=runtime,
    )
