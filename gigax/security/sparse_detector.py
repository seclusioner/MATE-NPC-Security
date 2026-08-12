from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from gigax.security.detector import (
    RepresentationRiskScore,
    load_detector_bundle,
    format_chat_prompt,
)


class SparseLogisticRiskDetector:
    """
    Runtime scorer for Low-FPR Sparse Multi-RAAC heads.

    Artifact format:
        method = logistic_hidden

        layers
        hidden_size
        feature_indices
        scaler_mean
        scaler_scale
        coef
        intercept
        threshold
        score_scale

    Runtime:

        selected hidden features
                ↓
        StandardScaler
                ↓
        logistic regression
                ↓
        probability score
                ↓
        normalized margin

        m = (score - threshold) / score_scale
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        detector_path: str | Path,
        device: str | torch.device | None = None,
        threshold_scale: float = 1.0,
        margin: float = 0.0,
        chat_template_mode: str = "tokenizer",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.detector_path = Path(
            detector_path
        )

        self.bundle = load_detector_bundle(
            self.detector_path
        )

        method = str(
            self.bundle.get(
                "method",
                "",
            )
        )

        if method != "logistic_hidden":
            raise ValueError(
                f"{self.detector_path} is not a "
                "logistic_hidden detector."
            )

        self.name = str(
            self.bundle.get(
                "name",
                self.detector_path.stem,
            )
        )

        self.risk_family = str(
            self.bundle.get(
                "risk_type",
                self.name,
            )
        )

        self.layers = [
            int(layer)
            for layer in self.bundle["layers"]
        ]

        self.hidden_size = int(
            self.bundle["hidden_size"]
        )

        self.feature_indices = (
            self.bundle[
                "feature_indices"
            ]
            .detach()
            .cpu()
            .to(torch.long)
        )

        self.scaler_mean = (
            self.bundle[
                "scaler_mean"
            ]
            .detach()
            .cpu()
            .to(torch.float32)
        )

        self.scaler_scale = (
            self.bundle[
                "scaler_scale"
            ]
            .detach()
            .cpu()
            .to(torch.float32)
        )

        self.coef = (
            self.bundle["coef"]
            .detach()
            .cpu()
            .to(torch.float32)
        )

        self.intercept = float(
            self.bundle["intercept"]
        )

        self.base_threshold = float(
            self.bundle["threshold"]
        )

        self.threshold_scale = float(
            threshold_scale
        )

        self.margin = float(margin)

        self.threshold = (
            self.base_threshold
            * self.threshold_scale
        )

        self.score_scale = float(
            self.bundle.get(
                "score_scale",
                1.0,
            )
        )

        if self.score_scale <= 0.0:
            raise ValueError(
                f"Invalid score_scale="
                f"{self.score_scale}"
            )

        self.chat_template_mode = str(
            chat_template_mode
            or "tokenizer"
        )

        if device is None:
            self.device = next(
                model.parameters()
            ).device
        else:
            self.device = torch.device(
                device
            )

        self._validate_artifact()

    def _validate_artifact(
        self,
    ) -> None:
        if not self.layers:
            raise ValueError(
                "Sparse detector contains "
                "no selected layers."
            )

        k = int(
            self.feature_indices.numel()
        )

        for name, tensor in [
            (
                "scaler_mean",
                self.scaler_mean,
            ),
            (
                "scaler_scale",
                self.scaler_scale,
            ),
            (
                "coef",
                self.coef,
            ),
        ]:
            if int(
                tensor.numel()
            ) != k:
                raise ValueError(
                    f"{name} size "
                    f"{tensor.numel()} != "
                    f"feature count {k}"
                )

        full_dimension = (
            len(self.layers)
            * self.hidden_size
        )

        if k <= 0:
            raise ValueError(
                "Sparse detector selects "
                "zero features."
            )

        if int(
            self.feature_indices.min()
        ) < 0:
            raise ValueError(
                "Negative feature index."
            )

        if int(
            self.feature_indices.max()
        ) >= full_dimension:
            raise ValueError(
                "feature_indices exceed "
                "flattened hidden dimension."
            )

        if torch.any(
            self.scaler_scale == 0
        ):
            raise ValueError(
                "scaler_scale contains zero."
            )

    def score_flattened(
        self,
        flattened: torch.Tensor,
    ) -> float:
        """
        Score one already-flattened selected-layer representation.

        This function is separated from model inference so artifact math
        can be unit-tested independently.
        """

        flattened = (
            flattened
            .detach()
            .to(
                device=self.device,
                dtype=torch.float32,
            )
            .reshape(-1)
        )

        expected = (
            len(self.layers)
            * self.hidden_size
        )

        if int(
            flattened.numel()
        ) != expected:
            raise ValueError(
                f"Expected flattened dimension "
                f"{expected}, got "
                f"{flattened.numel()}."
            )

        indices = (
            self.feature_indices
            .to(self.device)
        )

        selected = flattened.index_select(
            0,
            indices,
        )

        mean = self.scaler_mean.to(
            self.device
        )

        scale = self.scaler_scale.to(
            self.device
        )

        coef = self.coef.to(
            self.device
        )

        standardized = (
            selected - mean
        ) / scale

        logit = (
            standardized
            * coef
        ).sum() + self.intercept

        probability = torch.sigmoid(
            logit
        )

        return float(
            probability.item()
        )

    @torch.no_grad()
    def score_text(
        self,
        text: str,
    ) -> float:
        if not isinstance(
            text,
            str,
        ):
            text = str(text)

        prompt = format_chat_prompt(
            tokenizer=self.tokenizer,
            text=text,
            mode=self.chat_template_mode,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.device)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )

        attention_mask = inputs.get(
            "attention_mask"
        )

        if attention_mask is not None:
            last_idx = (
                int(
                    attention_mask[0]
                    .sum()
                    .item()
                )
                - 1
            )
        else:
            last_idx = (
                outputs
                .hidden_states[0]
                .shape[1]
                - 1
            )

        selected_layers = []

        for layer in self.layers:
            # hidden_states[0] is embedding output.
            hidden = (
                outputs
                .hidden_states[
                    layer + 1
                ][0]
                .detach()
                .to(torch.float32)
            )

            representation = hidden[
                last_idx
            ]

            if int(
                representation.numel()
            ) != self.hidden_size:
                raise ValueError(
                    f"Layer {layer} hidden "
                    f"size mismatch: "
                    f"{representation.numel()} "
                    f"!= {self.hidden_size}"
                )

            selected_layers.append(
                representation
            )

        flattened = torch.cat(
            selected_layers,
            dim=0,
        )

        return self.score_flattened(
            flattened
        )

    def score(
        self,
        text: str,
    ) -> float:
        return self.score_text(text)

    def decide(
        self,
        text: str,
    ) -> RepresentationRiskScore:
        score = self.score_text(text)

        operating_threshold = (
            self.threshold
            + self.margin
        )

        relative_margin = (
            score
            - operating_threshold
        ) / self.score_scale

        restricted = bool(
            relative_margin >= 0.0
        )

        return RepresentationRiskScore(
            name=self.name,
            score=float(score),
            threshold=float(
                operating_threshold
            ),
            restricted=restricted,
            risk_type=self.risk_family,
            relative_margin=float(
                relative_margin
            ),
            score_scale=float(
                self.score_scale
            ),
        )
