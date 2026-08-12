from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import outlines
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


@dataclass
class HFModelBundle:
    model: Any
    tokenizer: Any
    outlines_model: Any


class HFModelAdapter:
    """
    Hugging Face model adapter used by MATE.

    Model loading is isolated here so the security architecture does not
    depend on a specific backbone.
    """

    def __init__(
        self,
        model_id: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        self.model_id = model_id
        self.dtype = dtype
        self.device_map = device_map

    def resolve_dtype(self):
        dtype_l = str(
            self.dtype
        ).lower()

        if dtype_l == "float16":
            return torch.float16

        if dtype_l == "bfloat16":
            return torch.bfloat16

        if dtype_l == "float32":
            return torch.float32

        raise ValueError(
            f"Unsupported dtype: {self.dtype}"
        )

    def load(self) -> HFModelBundle:
        tokenizer = (
            AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=False,
            )
        )

        model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=False,
                dtype=self.resolve_dtype(),
                device_map=self.device_map,
            )
        )

        outlines_model = (
            outlines.models.Transformers(
                model,
                tokenizer,
            )
        )

        return HFModelBundle(
            model=model,
            tokenizer=tokenizer,
            outlines_model=outlines_model,
        )
