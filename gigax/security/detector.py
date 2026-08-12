from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def load_detector_bundle(
    path: str | Path,
) -> dict[str, Any]:
    """
    Safely load a serialized MATE detector artifact.

    Detector bundles are expected to contain tensors and simple
    Python metadata only.
    """
    path = Path(path)

    try:
        bundle = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        bundle = torch.load(
            path,
            map_location="cpu",
        )

    if not isinstance(bundle, dict):
        raise TypeError(
            f"Detector artifact must contain a dict, "
            f"got {type(bundle)} from {path}."
        )

    return bundle


@dataclass
class RepresentationRiskScore:
    name: str
    score: float
    threshold: float
    restricted: bool
    risk_type: str
    relative_margin: float = 0.0
    score_scale: float = 1.0


def format_chat_prompt(
    tokenizer: Any,
    text: str,
    mode: str = "tokenizer",
) -> str:
    """
    Convert detector input to a model-specific chat prompt.

    Supported modes:
    - tokenizer:
        use tokenizer.apply_chat_template()
    - phi3_manual:
        reproduce the original thesis Phi-3 formatting
    - plain:
        use raw text directly
    """

    mode_l = str(
        mode or "tokenizer"
    ).lower()

    if mode_l == "phi3_manual":
        return (
            "<|user|>\n"
            + text.strip()
            + "\n<|assistant|>\n"
        )

    if mode_l == "plain":
        return text.strip()

    if mode_l == "tokenizer":
        if not hasattr(
            tokenizer,
            "apply_chat_template",
        ):
            raise ValueError(
                "Tokenizer does not support "
                "apply_chat_template()."
            )

        messages = [
            {
                "role": "user",
                "content": text.strip(),
            }
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    raise ValueError(
        f"Unknown chat template mode: {mode}"
    )


class RepresentationRiskDetector:
    """
    One Original Multi-RAAC representation-reading head.

    Each detector loads a mean-difference direction artifact and scores
    selected hidden layers using:

        s_l(x) = h_l(x)^T v_l - b_l

    The final detector score is the mean across selected layers.

    The detector only estimates semantic risk. It does NOT decide which
    game actions are executable; ATP and TEG own authorization.
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
        self.detector_path = Path(detector_path)

        self.bundle = load_detector_bundle(
            self.detector_path
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

        # Historical artifacts may omit the pool field.
        self.pool = (
            self.bundle.get("pool")
            or "last"
        )

        self.base_threshold = float(
            self.bundle["threshold"]
        )

        self.threshold_scale = float(
            threshold_scale
        )

        self.margin = float(margin)

        self.chat_template_mode = str(
            chat_template_mode
            or "tokenizer"
        )

        self.threshold = (
            self.base_threshold
            * self.threshold_scale
        )

        # Formal Multi-RAAC artifacts provide a calibration scale
        # used to normalize scores from different risk heads before
        # OR / max-margin fusion.
        self.score_scale = float(
            self.bundle.get(
                "score_scale",
                1.0,
            )
        )

        if self.score_scale <= 0.0:
            raise ValueError(
                f"Invalid score_scale={self.score_scale} "
                f"in detector {self.detector_path}"
            )

        # Generic detector format.
        if "vec" in self.bundle:
            raw_vec = self.bundle["vec"]
            raw_bias = self.bundle["bias"]

        # Backward compatibility with early harmfulness artifacts.
        elif "harm_vec" in self.bundle:
            raw_vec = self.bundle["harm_vec"]
            raw_bias = self.bundle["harm_bias"]

        else:
            raise KeyError(
                f"{self.detector_path} does not contain "
                "'vec'/'bias' or legacy 'harm_vec'/'harm_bias'."
            )

        self.vec = {
            int(layer): tensor
            .detach()
            .cpu()
            .to(torch.float32)
            for layer, tensor in raw_vec.items()
        }

        self.bias = {
            int(layer): float(value)
            for layer, value in raw_bias.items()
        }

        if device is None:
            self.device = next(
                model.parameters()
            ).device
        else:
            self.device = torch.device(device)

        self._validate_artifact()

    def _validate_artifact(self) -> None:
        if not self.layers:
            raise ValueError(
                "Detector contains no selected layers."
            )

        for layer in self.layers:
            if layer not in self.vec:
                raise KeyError(
                    f"Missing vector for layer {layer}"
                )

            if layer not in self.bias:
                raise KeyError(
                    f"Missing bias for layer {layer}"
                )

    @torch.no_grad()
    def score_text(self, text: str) -> float:
        if not isinstance(text, str):
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
                outputs.hidden_states[0]
                .shape[1]
                - 1
            )

        layer_scores = []

        for layer in self.layers:
            # Hugging Face hidden_states[0] is the embedding output,
            # therefore transformer layer l is indexed by l + 1.
            hidden = (
                outputs
                .hidden_states[layer + 1][0]
                .detach()
                .to(torch.float32)
            )

            if self.pool == "last":
                representation = hidden[last_idx]

            elif self.pool == "mean":
                representation = hidden[
                    : last_idx + 1
                ].mean(dim=0)

            else:
                raise ValueError(
                    f"Unknown pooling mode: {self.pool}"
                )

            direction = self.vec[layer].to(
                representation.device
            )

            score = (
                torch.dot(
                    representation,
                    direction,
                ).item()
                - self.bias[layer]
            )

            layer_scores.append(score)

        return float(
            sum(layer_scores)
            / len(layer_scores)
        )

    # Compatibility with generic scorer interfaces.
    def score(self, text: str) -> float:
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
            score - operating_threshold
        ) / self.score_scale

        restricted = (
            relative_margin >= 0.0
        )

        return RepresentationRiskScore(
            name=self.name,
            score=float(score),
            threshold=float(
                operating_threshold
            ),
            restricted=bool(restricted),
            risk_type=self.risk_family,
            relative_margin=float(
                relative_margin
            ),
            score_scale=float(
                self.score_scale
            ),
        )
