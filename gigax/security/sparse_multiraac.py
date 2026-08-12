from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch

from gigax.security.detector import (
    format_chat_prompt,
    load_detector_bundle,
)
from gigax.security.raac import (
    MultiRAACController,
    RAACDecision,
)


EPS = 1.0e-8


def resolve_decoder_layers(
    model: Any,
):
    """
    Locate decoder blocks for common Hugging Face causal LMs.
    """

    candidates = (
        ("model", "layers"),
        ("model", "model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )

    for path in candidates:
        current = model

        try:
            for name in path:
                current = getattr(
                    current,
                    name,
                )
        except AttributeError:
            continue

        if isinstance(
            current,
            (
                torch.nn.ModuleList,
                list,
                tuple,
            ),
        ):
            return current

    raise RuntimeError(
        "Cannot locate decoder layers."
    )


def unwrap_hidden(
    output: Any,
) -> torch.Tensor:
    if isinstance(
        output,
        torch.Tensor,
    ):
        return output

    if isinstance(
        output,
        (tuple, list),
    ):
        if (
            output
            and isinstance(
                output[0],
                torch.Tensor,
            )
        ):
            return output[0]

    raise TypeError(
        "Unsupported decoder-layer "
        f"output type: {type(output)!r}"
    )


class SelectedLayerCapture:
    """
    Capture only selected decoder-layer last-token vectors.

    Hooks are registered once and reused for every request.
    """

    def __init__(
        self,
        model: Any,
        layers: Iterable[int],
        *,
        output_device: torch.device,
    ):
        self.layers = sorted(
            {
                int(layer)
                for layer in layers
            }
        )

        if not self.layers:
            raise ValueError(
                "At least one layer "
                "is required."
            )

        decoder_layers = (
            resolve_decoder_layers(
                model
            )
        )

        if (
            self.layers[-1]
            >= len(decoder_layers)
        ):
            raise IndexError(
                f"Requested layer "
                f"{self.layers[-1]}, "
                f"but model has "
                f"{len(decoder_layers)} "
                "decoder layers."
            )

        self.output_device = (
            output_device
        )

        self._enabled = False
        self._last_index = 0
        self._captured = {}
        self._handles = []

        for layer in self.layers:
            handle = (
                decoder_layers[layer]
                .register_forward_hook(
                    self._make_hook(
                        layer
                    )
                )
            )

            self._handles.append(
                handle
            )

    def _make_hook(
        self,
        layer: int,
    ):
        def hook(
            _module,
            _inputs,
            output,
        ):
            if not self._enabled:
                return

            hidden = unwrap_hidden(
                output
            )

            self._captured[
                layer
            ] = (
                hidden[
                    0,
                    self._last_index,
                    :,
                ]
                .detach()
                .to(
                    device=(
                        self.output_device
                    ),
                    dtype=torch.float32,
                )
            )

        return hook

    def begin(
        self,
        last_index: int,
    ) -> None:
        self._last_index = int(
            last_index
        )

        self._captured.clear()
        self._enabled = True

    def end(
        self,
    ) -> torch.Tensor:
        self._enabled = False

        missing = [
            layer
            for layer in self.layers
            if layer
            not in self._captured
        ]

        if missing:
            raise RuntimeError(
                "Hooks failed to capture "
                f"layers: {missing}"
            )

        return torch.stack(
            [
                self._captured[
                    layer
                ]
                for layer
                in self.layers
            ],
            dim=0,
        )

    def disable(
        self,
    ) -> None:
        self._enabled = False
        self._captured.clear()

    def close(
        self,
    ) -> None:
        self.disable()

        for handle in self._handles:
            handle.remove()

        self._handles.clear()


class SharedSparseMultiRAACController(
    MultiRAACController
):
    """
    Shared-forward Low-FPR Sparse Multi-RAAC.

    One request performs:

        one backbone forward
              ↓
        selected layer capture
              ↓
        union feature gather
              ↓
        vectorized sparse heads
              ↓
        normalized margins
              ↓
        calibrated max fusion
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        bundle_paths: Iterable[
            str | Path
        ],
        fusion_threshold: float,
        chat_template_mode: str = (
            "tokenizer"
        ),
    ):
        super().__init__(
            [],
            fusion_threshold=(
                fusion_threshold
            ),
            system_name=(
                "lowfpr_sparse"
            ),
        )

        self.model = model
        self.tokenizer = tokenizer
        self.chat_template_mode = str(
            chat_template_mode
        )

        self.device = next(
            model.parameters()
        ).device

        bundles = [
            load_detector_bundle(
                path
            )
            for path in bundle_paths
        ]

        if len(bundles) < 2:
            raise ValueError(
                "Sparse Multi-RAAC "
                "requires at least "
                "two detector heads."
            )

        if any(
            bundle.get("method")
            != "logistic_hidden"
            for bundle in bundles
        ):
            raise ValueError(
                "Expected logistic_hidden "
                "detector bundles."
            )

        layer_sets = [
            tuple(
                int(layer)
                for layer
                in bundle["layers"]
            )
            for bundle in bundles
        ]

        if len(
            set(layer_sets)
        ) != 1:
            raise ValueError(
                "All sparse heads must "
                "use identical layers."
            )

        self.layers = list(
            layer_sets[0]
        )

        hidden_sizes = {
            int(
                bundle[
                    "hidden_size"
                ]
            )
            for bundle in bundles
        }

        if len(hidden_sizes) != 1:
            raise ValueError(
                "Sparse heads have "
                "different hidden sizes."
            )

        self.hidden_size = (
            hidden_sizes.pop()
        )

        self.names = [
            str(
                bundle.get(
                    "name",
                    f"head_{index}",
                )
            )
            for index, bundle
            in enumerate(bundles)
        ]

        self.risk_types = [
            str(
                bundle.get(
                    "risk_type",
                    self.names[index],
                )
            )
            for index, bundle
            in enumerate(bundles)
        ]

        head_indices = [
            bundle[
                "feature_indices"
            ]
            .long()
            .tolist()
            for bundle in bundles
        ]

        union_indices = sorted(
            {
                index
                for indices
                in head_indices
                for index in indices
            }
        )

        union_position = {
            index: position
            for position, index
            in enumerate(
                union_indices
            )
        }

        max_k = max(
            len(indices)
            for indices
            in head_indices
        )

        position_matrix = []
        means = []
        scales = []
        coefficients = []
        masks = []

        for (
            bundle,
            indices,
        ) in zip(
            bundles,
            head_indices,
        ):
            k = len(indices)

            positions = [
                union_position[
                    index
                ]
                for index in indices
            ]

            padding = (
                max_k - k
            )

            position_matrix.append(
                positions
                + [0] * padding
            )

            means.append(
                bundle[
                    "scaler_mean"
                ]
                .float()
                .tolist()
                + [0.0] * padding
            )

            scales.append(
                bundle[
                    "scaler_scale"
                ]
                .float()
                .clamp_min(EPS)
                .tolist()
                + [1.0] * padding
            )

            coefficients.append(
                bundle["coef"]
                .float()
                .tolist()
                + [0.0] * padding
            )

            masks.append(
                [1.0] * k
                + [0.0] * padding
            )

        self.union_indices = (
            torch.tensor(
                union_indices,
                dtype=torch.long,
                device=self.device,
            )
        )

        self.position_matrix = (
            torch.tensor(
                position_matrix,
                dtype=torch.long,
                device=self.device,
            )
        )

        self.means = torch.tensor(
            means,
            dtype=torch.float32,
            device=self.device,
        )

        self.scales = torch.tensor(
            scales,
            dtype=torch.float32,
            device=self.device,
        )

        self.coefficients = (
            torch.tensor(
                coefficients,
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.masks = torch.tensor(
            masks,
            dtype=torch.float32,
            device=self.device,
        )

        self.intercepts = (
            torch.tensor(
                [
                    float(
                        bundle[
                            "intercept"
                        ]
                    )
                    for bundle
                    in bundles
                ],
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.thresholds = (
            torch.tensor(
                [
                    float(
                        bundle[
                            "threshold"
                        ]
                    )
                    for bundle
                    in bundles
                ],
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.score_scales = (
            torch.tensor(
                [
                    max(
                        float(
                            bundle[
                                "score_scale"
                            ]
                        ),
                        EPS,
                    )
                    for bundle
                    in bundles
                ],
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.capture = (
            SelectedLayerCapture(
                model,
                self.layers,
                output_device=(
                    self.device
                ),
            )
        )

    def score_selected_hidden(
        self,
        selected_hidden: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        expected = (
            len(self.layers),
            self.hidden_size,
        )

        if tuple(
            selected_hidden.shape
        ) != expected:
            raise ValueError(
                f"Expected {expected}, "
                f"got "
                f"{tuple(selected_hidden.shape)}"
            )

        flattened = (
            selected_hidden
            .to(
                device=self.device,
                dtype=torch.float32,
            )
            .reshape(-1)
        )

        union_values = (
            flattened.index_select(
                0,
                self.union_indices,
            )
        )

        per_head = union_values[
            self.position_matrix
        ]

        normalized = (
            per_head
            - self.means
        ) / self.scales

        logits = (
            (
                normalized
                * self.coefficients
                * self.masks
            ).sum(dim=1)
            + self.intercepts
        )

        probabilities = (
            torch.sigmoid(
                logits
            )
        )

        margins = (
            probabilities
            - self.thresholds
        ) / self.score_scales

        fused_margin = (
            torch.max(
                margins
            )
        )

        return (
            probabilities,
            margins,
            fused_margin,
        )

    @torch.inference_mode()
    def decide(
        self,
        text: str,
    ) -> RAACDecision:
        prompt = format_chat_prompt(
            tokenizer=self.tokenizer,
            text=str(text),
            mode=(
                self.chat_template_mode
            ),
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        attention_mask = (
            inputs.get(
                "attention_mask"
            )
        )

        if attention_mask is None:
            last_index = (
                inputs["input_ids"]
                .shape[1]
                - 1
            )
        else:
            positions = torch.arange(
                attention_mask.shape[1],
                device=(
                    attention_mask.device
                ),
            )

            valid = positions[
                attention_mask[0].bool()
            ]

            if valid.numel() == 0:
                raise RuntimeError(
                    "Empty detector input."
                )

            last_index = int(
                valid[-1].item()
            )

        self.capture.begin(
            last_index
        )

        try:
            self.model(
                **inputs,
                use_cache=False,
                return_dict=True,
            )

            selected_hidden = (
                self.capture.end()
            )

        except Exception:
            self.capture.disable()
            raise

        (
            probabilities,
            margins,
            fused_margin,
        ) = self.score_selected_hidden(
            selected_hidden
        )

        fused_value = float(
            fused_margin.item()
        )

        restricted = bool(
            fused_value
            >= self.fusion_threshold
        )

        detector_scores = []

        for index, name in enumerate(
            self.names
        ):
            probability = float(
                probabilities[
                    index
                ].item()
            )

            margin = float(
                margins[
                    index
                ].item()
            )

            threshold = float(
                self.thresholds[
                    index
                ].item()
            )

            scale = float(
                self.score_scales[
                    index
                ].item()
            )

            detector_scores.append(
                {
                    "index": index,
                    "name": name,
                    "score": probability,
                    "threshold": threshold,
                    "score_scale": scale,
                    "relative_margin": (
                        margin
                    ),
                    "restricted": (
                        margin >= 0.0
                    ),
                    "risk_type": (
                        self.risk_types[
                            index
                        ]
                    ),
                }
            )

        if restricted:
            best_index = int(
                torch.argmax(
                    margins
                ).item()
            )

            risk_type = (
                self.risk_types[
                    best_index
                ]
            )
        else:
            risk_type = "none"

        return RAACDecision(
            score=fused_value,
            threshold=float(
                self.fusion_threshold
            ),
            restricted=restricted,
            risk_type=risk_type,
            detector_scores=(
                detector_scores
            ),
        )

    def close(
        self,
    ) -> None:
        self.capture.close()
