from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable

from training.common import text_hash


SUPPORTED_SOURCES = (
    "advbench",
    "alpaca",
    "jailbreakbench",
    "harmbench",
    "prompt_injection",
)


def _load_dataset_function():
    """
    Import datasets lazily so importing this module does not require
    Hugging Face datasets unless an actual download is requested.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face 'datasets' package is required "
            "to download public datasets. Install it with:\n"
            "  pip install datasets"
        ) from exc

    return load_dataset



def normalize_text(
    text: Any,
) -> str:
    if not isinstance(
        text,
        str,
    ):
        return ""

    # Preserve source prompt content.
    # Canonical normalization is only used
    # for hashing and deduplication.
    return text.strip()


def first_text(
    row: dict[str, Any],
    keys: tuple[str, ...],
) -> str:
    for key in keys:
        text = normalize_text(
            row.get(key)
        )

        if text:
            return text

    return ""


def extract_advbench(
    row: dict[str, Any],
) -> str:
    return first_text(
        row,
        (
            "prompt",
            "goal",
            "instruction",
        ),
    )



def extract_alpaca(
    row: dict[str, Any],
) -> str:
    instruction = normalize_text(
        row.get(
            "instruction"
        )
    )

    context = normalize_text(
        row.get(
            "input"
        )
    )

    if not instruction:
        return ""

    if context:
        return (
            f"{instruction}\n\n"
            f"Input:\n"
            f"{context}"
        )

    return instruction


def extract_jailbreakbench(
    row: dict[str, Any],
) -> str:
    return first_text(
        row,
        (
            "Goal",
            "goal",
            "Behavior",
            "behavior",
            "prompt",
            "text",
            "instruction",
        ),
    )


def extract_harmbench(
    row: dict[str, Any],
) -> str:
    return first_text(
        row,
        (
            "Behavior",
            "behavior",
            "prompt",
            "text",
            "Goal",
            "goal",
            "instruction",
        ),
    )


def extract_prompt_injection(
    row: dict[str, Any],
) -> str:
    return first_text(
        row,
        (
            "prompt",
            "text",
            "input",
            "instruction",
            "query",
            "content",
        ),
    )


EXTRACTORS: dict[
    str,
    Callable[
        [dict[str, Any]],
        str,
    ],
] = {
    "advbench": (
        extract_advbench
    ),
    "alpaca": (
        extract_alpaca
    ),
    "jailbreakbench": (
        extract_jailbreakbench
    ),
    "harmbench": (
        extract_harmbench
    ),
    "prompt_injection": (
        extract_prompt_injection
    ),
}


SOURCE_METADATA = {
    "advbench": {
        "dataset_id": (
            "walledai/AdvBench"
        ),
        "role": (
            "harm_positive"
        ),
    },
    "alpaca": {
        "dataset_id": (
            "tatsu-lab/alpaca"
        ),
        "role": (
            "harm_negative"
        ),
    },
    "jailbreakbench": {
        "dataset_id": (
            "JailbreakBench/"
            "JBB-Behaviors"
        ),
        "role": (
            "optional_harm_pool"
        ),
    },
    "harmbench": {
        "dataset_id": (
            "walledai/HarmBench"
        ),
        "role": (
            "optional_harm_pool"
        ),
    },
    "prompt_injection": {
        "dataset_id": (
            "neuralchemy/"
            "Prompt-injection-dataset"
        ),
        "role": (
            "optional_injection_pool"
        ),
    },
}


def adapt_records(
    *,
    source: str,
    records: list[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    """
    Convert raw HF rows into simple text-pool rows.

    This adapter intentionally does NOT add detector labels or
    train/validation splits. That responsibility belongs to
    training.prepare_data.
    """
    source = str(
        source
    ).strip().lower()

    if source not in EXTRACTORS:
        raise ValueError(
            f"Unsupported source "
            f"{source!r}. "
            f"Choose from "
            f"{SUPPORTED_SOURCES}."
        )

    extractor = (
        EXTRACTORS[source]
    )

    metadata = (
        SOURCE_METADATA[
            source
        ]
    )

    output = []
    seen = set()

    for index, row in enumerate(
        records
    ):
        if not isinstance(
            row,
            dict,
        ):
            continue

        text = extractor(
            row
        )

        if not text:
            continue

        # Text-level deduplication is deliberate.
        # prepare_data.py independently hashes again when
        # constructing train/validation manifests.
        dedupe_key = text_hash(
            text
        )

        if dedupe_key in seen:
            continue

        seen.add(
            dedupe_key
        )

        item = {
            "text": text,
            "source": (
                metadata[
                    "dataset_id"
                ]
            ),
            "source_kind": (
                source
            ),
            "source_role": (
                metadata[
                    "role"
                ]
            ),
            "source_index": (
                index
            ),
        }

        # Preserve a source-side class/label only as provenance.
        # It is NOT automatically interpreted as a detector label.
        for key in (
            "label",
            "class",
            "category",
            "type",
        ):
            if key in row:
                value = row[
                    key
                ]

                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                    ),
                ):
                    item[
                        "source_label"
                    ] = value
                    item[
                        "source_label_field"
                    ] = key
                    break

        output.append(
            item
        )

    if not output:
        raise RuntimeError(
            f"No usable text rows "
            f"were extracted from "
            f"{source!r}."
        )

    return output


def download_records(
    source: str,
) -> list[dict[str, Any]]:
    source = str(
        source
    ).strip().lower()

    if source not in (
        SUPPORTED_SOURCES
    ):
        raise ValueError(
            f"Unsupported source "
            f"{source!r}."
        )

    load_dataset = (
        _load_dataset_function()
    )

    if source == "advbench":
        dataset = load_dataset(
            "walledai/AdvBench",
            split="train",
        )

    elif source == "alpaca":
        dataset = load_dataset(
            "tatsu-lab/alpaca",
            split="train",
        )

    elif source == (
        "jailbreakbench"
    ):
        dataset = load_dataset(
            "JailbreakBench/"
            "JBB-Behaviors",
            "behaviors",
            split="harmful",
        )

    elif source == "harmbench":
        try:
            dataset = load_dataset(
                "walledai/HarmBench",
                "standard",
                split="train",
            )
        except Exception:
            dataset = load_dataset(
                "walledai/HarmBench",
                split="train",
            )

    elif source == (
        "prompt_injection"
    ):
        dataset = load_dataset(
            "neuralchemy/"
            "Prompt-injection-dataset",
            split="train",
        )

    else:
        raise AssertionError(
            source
        )

    return [
        dict(row)
        for row in dataset
    ]



def select_records(
    rows: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = list(
        rows
    )

    # limit=0 means keep the complete source
    # in its original order. This is important
    # for exact downstream deterministic splits.
    if limit <= 0:
        return rows

    random.Random(
        seed
    ).shuffle(
        rows
    )

    return rows[:limit]


def write_jsonl(
    path: str | Path,
    rows: list[dict[str, Any]],
) -> None:
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = (
        argparse.ArgumentParser(
            description=(
                "Download and normalize "
                "public Hugging Face "
                "datasets for MATE "
                "detector construction."
            )
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        choices=(
            SUPPORTED_SOURCES
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Maximum number of unique "
            "records. 0 keeps all."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--summary",
        default="",
        help=(
            "Optional JSON summary path."
        ),
    )

    args = parser.parse_args()

    raw_records = (
        download_records(
            args.source
        )
    )

    adapted = adapt_records(
        source=args.source,
        records=raw_records,
    )

    selected = select_records(
        adapted,
        limit=args.limit,
        seed=args.seed,
    )

    write_jsonl(
        args.output,
        selected,
    )

    metadata = (
        SOURCE_METADATA[
            args.source
        ]
    )

    summary = {
        "format_version": 1,
        "source": (
            args.source
        ),
        "dataset_id": (
            metadata[
                "dataset_id"
            ]
        ),
        "role": (
            metadata[
                "role"
            ]
        ),
        "raw_rows": (
            len(raw_records)
        ),
        "unique_text_rows": (
            len(adapted)
        ),
        "written_rows": (
            len(selected)
        ),
        "seed": (
            args.seed
        ),
        "limit": (
            args.limit
        ),
        "output": str(
            Path(
                args.output
            ).name
        ),
        "note": (
            "This adapter creates raw "
            "text pools only. Detector "
            "labels and train/validation "
            "splits are assigned by "
            "training.prepare_data."
        ),
    }

    if args.summary:
        summary_path = Path(
            args.summary
        )

        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
