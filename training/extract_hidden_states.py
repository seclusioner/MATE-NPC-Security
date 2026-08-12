from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from gigax.security.detector import (
    format_chat_prompt,
)
from training.common import (
    read_jsonl,
    sha256_file,
)


def resolve_dtype(
    name: str,
) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if name not in mapping:
        raise ValueError(
            f"Unsupported dtype: {name}"
        )

    return mapping[name]


def last_non_padding_indices(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Return the final non-padding token position for every sequence.

    Works with both left and right padding.
    """

    if attention_mask.ndim != 2:
        raise ValueError(
            "attention_mask must have "
            "shape [batch, sequence]."
        )

    positions = torch.arange(
        attention_mask.shape[1],
        device=attention_mask.device,
    ).unsqueeze(0)

    masked_positions = (
        positions.masked_fill(
            attention_mask.eq(0),
            -1,
        )
    )

    indices = (
        masked_positions
        .max(dim=1)
        .values
    )

    if torch.any(
        indices.lt(0)
    ):
        raise RuntimeError(
            "Encountered an input with "
            "no non-padding token."
        )

    return indices


def load_model_and_tokenizer(
    model_id: str,
    *,
    dtype: torch.dtype,
):
    """
    Load the backbone used to construct Layer-1 detectors.

    trust_remote_code=False preserves the formal Phi-3
    detector-building setup.
    """

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_id,
            trust_remote_code=False,
        )
    )

    if (
        tokenizer.pad_token_id
        is None
    ):
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "right"

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=False,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        .eval()
    )

    device = next(
        model.parameters()
    ).device

    metadata = {
        "model_id": model_id,
        "model_class": (
            type(model).__name__
        ),
        "config_class": (
            type(model.config).__name__
        ),
        "model_type": getattr(
            model.config,
            "model_type",
            None,
        ),
        "num_hidden_layers": getattr(
            model.config,
            "num_hidden_layers",
            None,
        ),
        "hidden_size": getattr(
            model.config,
            "hidden_size",
            None,
        ),
        "rope_scaling": getattr(
            model.config,
            "rope_scaling",
            None,
        ),
        "trust_remote_code": False,
        "attention_implementation": getattr(
            model.config,
            "_attn_implementation",
            None,
        ),
        "device": str(device),
        "dtype": str(dtype),
    }

    print(
        "[MODEL]",
        json.dumps(
            metadata,
            ensure_ascii=False,
        ),
    )

    return (
        tokenizer,
        model,
        device,
    )


def validate_manifest_rows(
    rows: list[dict],
    *,
    manifest: Path,
) -> None:
    if not rows:
        raise RuntimeError(
            f"No records in {manifest}"
        )

    for index, row in enumerate(
        rows
    ):
        if (
            "text" not in row
            or not str(
                row["text"]
            ).strip()
        ):
            raise KeyError(
                f"{manifest}, row {index}: "
                "missing non-empty 'text'."
            )


def extract_manifest(
    *,
    manifest: Path,
    output: Path,
    tokenizer,
    model,
    device: torch.device,
    model_id: str,
    chat_template_mode: str,
    batch_size: int,
    max_length: int,
    cache_dtype: torch.dtype,
    cache_dtype_name: str,
) -> dict:
    rows = read_jsonl(
        manifest
    )

    validate_manifest_rows(
        rows,
        manifest=manifest,
    )

    num_layers = int(
        model.config.num_hidden_layers
    )

    hidden_size = int(
        model.config.hidden_size
    )

    all_features = []
    per_sample_forward_ms = []

    for start in tqdm(
        range(
            0,
            len(rows),
            batch_size,
        ),
        desc=manifest.stem,
    ):
        batch_rows = rows[
            start:
            start + batch_size
        ]

        prompts = [
            format_chat_prompt(
                tokenizer=tokenizer,
                text=row["text"],
                mode=chat_template_mode,
            )
            for row in batch_rows
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        torch.cuda.synchronize()

        started = (
            time.perf_counter()
        )

        with torch.inference_mode():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        torch.cuda.synchronize()

        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        per_sample_forward_ms.extend(
            [
                elapsed_ms
                / len(batch_rows)
            ]
            * len(batch_rows)
        )

        expected_hidden_states = (
            num_layers + 1
        )

        if (
            len(outputs.hidden_states)
            != expected_hidden_states
        ):
            raise RuntimeError(
                "Unexpected hidden-state "
                f"count={len(outputs.hidden_states)}; "
                f"expected={expected_hidden_states}."
            )

        last_indices = (
            last_non_padding_indices(
                inputs[
                    "attention_mask"
                ]
            )
        )

        batch_indices = torch.arange(
            len(batch_rows),
            device=device,
        )

        layer_features = []

        for layer in range(
            num_layers
        ):
            # hidden_states[0] is the embedding output.
            hidden = (
                outputs
                .hidden_states[
                    layer + 1
                ]
            )

            selected = hidden[
                batch_indices,
                last_indices,
            ]

            layer_features.append(
                selected.detach().to(
                    device="cpu",
                    dtype=cache_dtype,
                )
            )

        all_features.append(
            torch.stack(
                layer_features,
                dim=1,
            )
        )

        del outputs
        del inputs

    features = torch.cat(
        all_features,
        dim=0,
    )

    payload = {
        "format_version": 3,
        "model_id": model_id,
        "model_class": (
            type(model).__name__
        ),
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "rows": rows,
        "features": features,
        "manifest_path": str(
            manifest
        ),
        "manifest_sha256": (
            sha256_file(
                manifest
            )
        ),
        "cache_dtype": (
            cache_dtype_name
        ),
        "chat_template_mode": (
            chat_template_mode
        ),
        "max_length": int(
            max_length
        ),
        "mean_backbone_forward_ms_per_sample": (
            sum(
                per_sample_forward_ms
            )
            / len(
                per_sample_forward_ms
            )
        ),
        "forward_times_ms_per_sample": (
            per_sample_forward_ms
        ),
        "trust_remote_code": False,
        "attention_implementation": getattr(
            model.config,
            "_attn_implementation",
            None,
        ),
    }

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        output.with_suffix(
            output.suffix
            + ".tmp"
        )
    )

    torch.save(
        payload,
        temporary,
    )

    temporary.replace(
        output
    )

    print(
        f"[SAVE] {output} "
        f"shape={tuple(features.shape)}"
    )

    return payload


def run_load_only(
    *,
    tokenizer,
    model,
    device,
    chat_template_mode: str,
) -> None:
    prompt = format_chat_prompt(
        tokenizer=tokenizer,
        text="Hello.",
        mode=chat_template_mode,
    )

    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32,
    ).to(device)

    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    expected = (
        int(
            model.config
            .num_hidden_layers
        )
        + 1
    )

    actual = len(
        outputs.hidden_states
    )

    print(
        "[LOAD-ONLY OK]",
        {
            "expected_hidden_states": (
                expected
            ),
            "actual_hidden_states": (
                actual
            ),
            "last_shape": tuple(
                outputs
                .hidden_states[-1]
                .shape
            ),
            "model_class": (
                type(model).__name__
            ),
            "chat_template_mode": (
                chat_template_mode
            ),
        },
    )

    if actual != expected:
        raise RuntimeError(
            f"Expected {expected} "
            f"hidden states, got {actual}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract last-token decoder "
            "representations for MATE "
            "Layer-1 detector construction."
        )
    )

    parser.add_argument(
        "--model-id",
        default=(
            "Gigax/NPC-LLM-3_8B"
        ),
    )

    parser.add_argument(
        "--manifests",
        nargs="*",
        default=[],
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "artifacts/hidden_cache"
        ),
    )

    parser.add_argument(
        "--dtype",
        choices=[
            "float16",
            "bfloat16",
            "float32",
        ],
        default="bfloat16",
    )

    parser.add_argument(
        "--cache-dtype",
        choices=[
            "float16",
            "float32",
        ],
        default="float16",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--chat-template-mode",
        choices=[
            "phi3_manual",
            "tokenizer",
            "plain",
        ],
        default="phi3_manual",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--load-only",
        action="store_true",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for "
            "hidden-state extraction."
        )

    if (
        not args.load_only
        and not args.manifests
    ):
        parser.error(
            "--manifests is required "
            "unless --load-only is used."
        )

    dtype = resolve_dtype(
        args.dtype
    )

    cache_dtype = resolve_dtype(
        args.cache_dtype
    )

    (
        tokenizer,
        model,
        device,
    ) = load_model_and_tokenizer(
        args.model_id,
        dtype=dtype,
    )

    if args.load_only:
        run_load_only(
            tokenizer=tokenizer,
            model=model,
            device=device,
            chat_template_mode=(
                args.chat_template_mode
            ),
        )
        return

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for manifest_value in (
        args.manifests
    ):
        manifest = Path(
            manifest_value
        ).resolve()

        if not manifest.exists():
            raise FileNotFoundError(
                f"Manifest not found: "
                f"{manifest}"
            )

        output = (
            output_dir
            / f"{manifest.stem}.pt"
        )

        if (
            output.exists()
            and not args.overwrite
        ):
            print(
                "[SKIP]",
                output,
            )
            continue

        extract_manifest(
            manifest=manifest,
            output=output,
            tokenizer=tokenizer,
            model=model,
            device=device,
            model_id=args.model_id,
            chat_template_mode=(
                args.chat_template_mode
            ),
            batch_size=(
                args.batch_size
            ),
            max_length=(
                args.max_length
            ),
            cache_dtype=(
                cache_dtype
            ),
            cache_dtype_name=(
                args.cache_dtype
            ),
        )

    print(
        "[DONE]",
        output_dir,
    )


if __name__ == "__main__":
    main()
