from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from outlines import regex
from outlines.generator import Generator

from gigax.parse import CharacterAction, get_guided_regex
from gigax.scene import Character, Item, Location, ProtagonistCharacter
from gigax.step_result import StepResult

load_dotenv()

logger = logging.getLogger("uvicorn")
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)


class NPCStepper:
    """
    Wrapper for constrained NPC action generation.

    Responsibilities:
    - Build or receive a guided regex for valid NPC actions.
    - Generate one action string using the local model.
    - Parse the generated action into CharacterAction.
    - Optionally return StepResult for debugging / experiments.
    """

    def __init__(self, model: Any, local: bool = True):
        self.model = model
        self.local = local

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_hf_model(self):
        return self.model.model

    def _get_hf_tokenizer(self):
        return self.model.tokenizer.tokenizer

    def _get_model_device(self) -> torch.device:
        """
        Return a reasonable device for tokenization tensors.

        This keeps the previous behavior but avoids hard-coding CUDA.
        """
        hf_model = self._get_hf_model()

        try:
            return next(hf_model.parameters()).device
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _can_use_cuda_stats(self, input_ids: torch.Tensor) -> bool:
        return bool(torch.cuda.is_available() and input_ids.is_cuda)

    # ------------------------------------------------------------------
    # Optional debug helper
    # ------------------------------------------------------------------

    def get_top_k_proportions(self, prompt: str, k: int = 5) -> None:
        """
        Print top-k next-token probabilities for debugging.

        This is not required by the production pipeline, but is useful when
        diagnosing why the model prefers one action token over another.
        """
        hf_model = self._get_hf_model()
        hf_tokenizer = self._get_hf_tokenizer()
        device = self._get_model_device()

        inputs = hf_tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = hf_model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=k)

        print(f"\n--- Next Token Probability Analysis (Top {k}) ---")
        for i in range(k):
            idx = top_k_indices[0][i]
            token_str = hf_tokenizer.decode(idx)
            probability = top_k_probs[0][i].item()
            print(
                f"Rank {i + 1}: "
                f"Token ID: {idx:5} | "
                f"Token: '{token_str:10}' | "
                f"Prob: {probability:.4f}"
            )
        print("------------------------------------------------\n")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_local(
        self,
        prompt: str,
        llm: Any,
        guided_regex: str,
        return_stats: bool = False,
        max_new_tokens: int = 128,
    ) -> str | dict[str, Any]:
        """
        Generate one action string with regex-constrained decoding.

        Args:
            prompt:
                Full prompt string.
            llm:
                Outlines model wrapper.
            guided_regex:
                Regex pattern describing allowed action outputs.
            return_stats:
                If True, return text and generation statistics.
            max_new_tokens:
                Maximum generation length.

        Returns:
            str if return_stats=False.
            dict if return_stats=True.
        """
        hf_tokenizer = self._get_hf_tokenizer()
        device = self._get_model_device()

        input_ids = hf_tokenizer.encode(
            prompt,
            return_tensors="pt",
        ).to(device)

        input_len = int(input_ids.shape[1])
        use_cuda_stats = self._can_use_cuda_stats(input_ids)

        if use_cuda_stats:
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()

        output_type = regex(guided_regex)
        generator = Generator(
            model=llm,
            output_type=output_type,
        )

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
        }

        # Outlines v1 forwards these generation parameters to the
        # underlying Transformers model. A positive temperature enables
        # stochastic sampling while the regex constraint remains active.
        temperature = float(
            os.environ.get("NPC_GENERATION_TEMPERATURE", "0.0")
        )
        top_p = float(
            os.environ.get("NPC_GENERATION_TOP_P", "1.0")
        )

        if temperature > 0.0:
            # Hugging Face defaults to greedy decoding unless do_sample=True.
            # Without this flag, temperature/top_p are warned as invalid and
            # may be ignored by the underlying Transformers model.
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
            if 0.0 < top_p <= 1.0:
                generation_kwargs["top_p"] = top_p

        result = generator(
            prompt,
            **generation_kwargs,
        )

        if use_cuda_stats:
            torch.cuda.synchronize()

        end_time = time.time()

        if not isinstance(result, str):
            raise ValueError(f"Expected string generation result, got {type(result)}")

        text = result.strip()
        output_ids = hf_tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        output_len = int(len(output_ids))

        if use_cuda_stats:
            vram_peak_mb = float(torch.cuda.max_memory_allocated() / 1024**2)
        else:
            vram_peak_mb = 0.0

        if return_stats:
            return {
                "text": text,
                "input_tokens": input_len,
                "output_tokens": output_len,
                "generation_time": float(end_time - start_time),
                "vram_peak_mb": vram_peak_mb,
            }

        return text

    # ------------------------------------------------------------------
    # Main action step
    # ------------------------------------------------------------------

    async def get_action(
        self,
        context: str,
        locations: list[Location],
        NPCs: list[Character],
        protagonist: ProtagonistCharacter,
        items: list[Item],
        events: list[CharacterAction],
        guided_regex=None,
        guided_actions: list[str] | None = None,
        return_result: bool = False,
    ) -> CharacterAction | StepResult | None:
        """
        Generate and parse one NPC action.

        Note:
            `context` is already a full prompt string in the current runtime.
            The method name is kept for compatibility with existing code.
        """
        prompt = context

        if guided_regex is None:
            if guided_actions is not None:
                guided_action_set = {
                    str(a).strip().lower()
                    for a in guided_actions
                }
                filtered_skills = [
                    skill
                    for skill in protagonist.skills
                    if str(skill.name).strip().lower() in guided_action_set
                ]
            else:
                filtered_skills = protagonist.skills

            guided_regex = get_guided_regex(
                filtered_skills,
                NPCs,
                locations,
                items,
            )

        if not self.local:
            raise NotImplementedError("Remote API generation is not supported in this runtime.")

        try:
            generated = self.generate_local(
                prompt=prompt,
                llm=self.model,
                guided_regex=guided_regex.pattern,
                return_stats=return_result,
            )

            if return_result:
                raw_text = generated["text"]
            else:
                raw_text = generated

        except Exception as exc:
            if return_result:
                return StepResult(
                    raw_output="",
                    parsed_action=None,
                    parse_success=False,
                    guided_regex=getattr(guided_regex, "pattern", str(guided_regex)),
                    guided_actions=guided_actions or [],
                    error=repr(exc),
                )
            raise

        try:
            parsed_action = CharacterAction.from_str(
                raw_text,
                protagonist,
                NPCs,
                locations,
                items,
                guided_regex,
            )
        except Exception as exc:
            logger.debug("Action parsing failed: %s", repr(exc))
            parsed_action = None

        parse_success = bool(
            parsed_action is not None
            and getattr(parsed_action, "parse_success", False)
            and str(getattr(parsed_action, "command", "") or "").lower() != "unknown"
        )

        if return_result:
            return StepResult(
                raw_output=raw_text,
                parsed_action=parsed_action,
                parse_success=parse_success,
                guided_regex=guided_regex.pattern,
                guided_actions=guided_actions or [],
                input_tokens=int(generated.get("input_tokens", 0)),
                output_tokens=int(generated.get("output_tokens", 0)),
                generation_time=float(generated.get("generation_time", 0.0)),
                vram_peak_mb=float(generated.get("vram_peak_mb", 0.0)),
            )

        return parsed_action