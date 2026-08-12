from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from training.common import (
    deterministic_partition,
    extract_text,
    load_source,
    load_text_pool,
    text_hash,
    write_jsonl,
)




TEXT_KEYS = (
    "text",
    "player_input",
    "prompt",
    "instruction",
    "goal",
    "Goal",
    "behavior",
    "Behavior",
    "query",
    "content",
)

POSITIVE_LABELS = {
    "1",
    "true",
    "positive",
    "pos",
    "attack",
    "harmful",
    "unsafe",
    "malicious",
    "injection",
    "risk",
}

NEGATIVE_LABELS = {
    "0",
    "false",
    "negative",
    "neg",
    "benign",
    "safe",
    "normal",
    "clean",
    "non-injection",
    "non_injection",
}


# ---------------------------------------------------------------------
# Generic source loading
# ---------------------------------------------------------------------





# ---------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------







# ---------------------------------------------------------------------
# Labels and splitting
# ---------------------------------------------------------------------

def parse_binary_label(
    value: Any,
) -> int:
    if isinstance(value, bool):
        return int(value)

    if isinstance(
        value,
        (int, float),
    ):
        if float(value) in {
            0.0,
            1.0,
        }:
            return int(value)

    normalized = (
        str(value)
        .strip()
        .casefold()
    )

    if normalized in POSITIVE_LABELS:
        return 1

    if normalized in NEGATIVE_LABELS:
        return 0

    raise ValueError(
        f"Unsupported binary label "
        f"{value!r}"
    )




# ---------------------------------------------------------------------
# Standard detector manifest
# ---------------------------------------------------------------------

def enrich(
    rows: list[dict[str, Any]],
    *,
    label: int,
    risk_family: str,
    domain: str,
    split: str,
    prefix: str,
) -> list[dict[str, Any]]:
    output = []

    for index, row in enumerate(
        rows
    ):
        output.append(
            {
                "id": (
                    f"{prefix}_"
                    f"{split}_"
                    f"{index:04d}"
                ),
                "text": row["text"],
                "text_hash": (
                    row["text_hash"]
                ),
                "label": int(label),
                "risk_family": (
                    risk_family
                ),
                "risk_subtype": "",
                "domain": domain,
                "split": split,
                "source": row["source"],
                "source_index": (
                    row["source_index"]
                ),
                "evaluation_group": (
                    "attack"
                    if label == 1
                    else "benign"
                ),
            }
        )

    return output


def load_labeled_manifest(
    path: str | Path,
    *,
    split: str,
    text_field: str,
    label_field: str,
    type_field: str | None,
) -> list[dict[str, Any]]:
    source_path = Path(path)

    output = []
    seen = set()

    for index, row in enumerate(
        load_source(source_path)
    ):
        if label_field not in row:
            raise KeyError(
                f"{source_path}, row "
                f"{index}: missing "
                f"{label_field!r}"
            )

        text = extract_text(
            row,
            text_field,
        )

        key = text_hash(text)

        if key in seen:
            continue

        seen.add(key)

        label = parse_binary_label(
            row[label_field]
        )

        subtype = (
            str(
                row.get(
                    type_field,
                    "",
                )
            ).strip()
            if type_field
            else ""
        )

        output.append(
            {
                "id": str(
                    row.get("id")
                    or (
                        f"injection_"
                        f"{split}_"
                        f"{index:04d}"
                    )
                ),
                "text": text,
                "text_hash": key,
                "label": label,
                "risk_family": (
                    "injection"
                ),
                "risk_subtype": subtype,
                "domain": (
                    "npc_injection"
                    if label == 1
                    else "npc_benign"
                ),
                "split": split,
                "source": (
                    source_path.name
                ),
                "source_index": index,
                "evaluation_group": (
                    "attack"
                    if label == 1
                    else "benign"
                ),
            }
        )

    if not output:
        raise RuntimeError(
            f"No usable records in "
            f"{source_path}"
        )

    labels = Counter(
        row["label"]
        for row in output
    )

    if (
        labels[0] == 0
        or labels[1] == 0
    ):
        raise RuntimeError(
            f"{source_path} must "
            "contain both labels. "
            f"Counts={dict(labels)}"
        )

    return output


def assert_no_overlap(
    named: dict[
        str,
        list[dict[str, Any]],
    ],
) -> None:
    owners: dict[str, str] = {}

    conflicts = []

    for split_name, rows in (
        named.items()
    ):
        for row in rows:
            key = row["text_hash"]

            previous = owners.get(
                key
            )

            if (
                previous is not None
                and previous != split_name
            ):
                conflicts.append(
                    (
                        previous,
                        split_name,
                        key,
                    )
                )

            owners[key] = split_name

    if conflicts:
        raise RuntimeError(
            "Data leakage across "
            "train/validation splits. "
            f"First conflicts="
            f"{conflicts[:10]}"
        )




def summarize(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "n": len(rows),
        "labels": dict(
            Counter(
                str(row["label"])
                for row in rows
            )
        ),
        "risk_families": dict(
            Counter(
                str(
                    row[
                        "risk_family"
                    ]
                )
                for row in rows
            )
        ),
        "domains": dict(
            Counter(
                str(
                    row.get(
                        "domain",
                        "",
                    )
                )
                for row in rows
            )
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare deterministic "
            "Multi-RAAC detector "
            "training manifests."
        )
    )

    # Harm detector
    parser.add_argument(
        "--harm-positive",
        required=True,
        help=(
            "Harmful positive pool "
            "(e.g. AdvBench)."
        ),
    )

    parser.add_argument(
        "--harm-negative",
        required=True,
        help=(
            "Benign negative pool "
            "(e.g. Alpaca)."
        ),
    )

    parser.add_argument(
        "--harm-positive-field",
    )

    parser.add_argument(
        "--harm-negative-field",
    )

    # Injection detector:
    # either fixed labeled train/val manifests,
    # or positive/negative pools.
    injection_group = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    injection_group.add_argument(
        "--injection-train-manifest"
    )

    injection_group.add_argument(
        "--injection-positive"
    )

    parser.add_argument(
        "--injection-validation-manifest"
    )

    parser.add_argument(
        "--injection-negative"
    )

    parser.add_argument(
        "--injection-positive-field"
    )

    parser.add_argument(
        "--injection-negative-field"
    )

    parser.add_argument(
        "--injection-manifest-text-field",
        default="text",
    )

    parser.add_argument(
        "--injection-manifest-label-field",
        default="label",
    )

    parser.add_argument(
        "--injection-manifest-type-field",
        default="type",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # Thesis-compatible defaults.
    parser.add_argument(
        "--harm-train-pos",
        type=int,
        default=201,
    )

    parser.add_argument(
        "--harm-val-pos",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--harm-train-neg",
        type=int,
        default=199,
    )

    parser.add_argument(
        "--harm-val-neg",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--injection-train-pos",
        type=int,
        default=31,
    )

    parser.add_argument(
        "--injection-val-pos",
        type=int,
        default=9,
    )

    parser.add_argument(
        "--injection-train-neg",
        type=int,
        default=29,
    )

    parser.add_argument(
        "--injection-val-neg",
        type=int,
        default=11,
    )

    args = parser.parse_args()

    if args.injection_train_manifest:
        if (
            not args
            .injection_validation_manifest
        ):
            parser.error(
                "--injection-validation-manifest "
                "is required with "
                "--injection-train-manifest."
            )
    else:
        if not args.injection_negative:
            parser.error(
                "--injection-negative is "
                "required with "
                "--injection-positive."
            )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Harm
    # -------------------------------------------------------------

    harm_positive = load_text_pool(
        args.harm_positive,
        args.harm_positive_field,
    )

    harm_negative = load_text_pool(
        args.harm_negative,
        args.harm_negative_field,
    )

    (
        hp_train,
        hp_validation,
    ) = deterministic_partition(
        harm_positive,
        [
            args.harm_train_pos,
            args.harm_val_pos,
        ],
        args.seed + 11,
    )

    (
        hn_train,
        hn_validation,
    ) = deterministic_partition(
        harm_negative,
        [
            args.harm_train_neg,
            args.harm_val_neg,
        ],
        args.seed + 12,
    )

    harm_train = (
        enrich(
            hp_train,
            label=1,
            risk_family="harm",
            domain="public_harm",
            split="train",
            prefix="harm_pos",
        )
        + enrich(
            hn_train,
            label=0,
            risk_family="harm",
            domain="public_benign",
            split="train",
            prefix="harm_neg",
        )
    )

    harm_validation = (
        enrich(
            hp_validation,
            label=1,
            risk_family="harm",
            domain="public_harm",
            split="validation",
            prefix="harm_pos",
        )
        + enrich(
            hn_validation,
            label=0,
            risk_family="harm",
            domain="public_benign",
            split="validation",
            prefix="harm_neg",
        )
    )

    # -------------------------------------------------------------
    # Injection / memory
    # -------------------------------------------------------------

    if args.injection_train_manifest:
        injection_train = (
            load_labeled_manifest(
                args.injection_train_manifest,
                split="train",
                text_field=(
                    args
                    .injection_manifest_text_field
                ),
                label_field=(
                    args
                    .injection_manifest_label_field
                ),
                type_field=(
                    args
                    .injection_manifest_type_field
                ),
            )
        )

        injection_validation = (
            load_labeled_manifest(
                args.injection_validation_manifest,
                split="validation",
                text_field=(
                    args
                    .injection_manifest_text_field
                ),
                label_field=(
                    args
                    .injection_manifest_label_field
                ),
                type_field=(
                    args
                    .injection_manifest_type_field
                ),
            )
        )

        injection_source_mode = (
            "existing_fixed_manifests"
        )

    else:
        injection_positive = (
            load_text_pool(
                args.injection_positive,
                args.injection_positive_field,
            )
        )

        injection_negative = (
            load_text_pool(
                args.injection_negative,
                args.injection_negative_field,
            )
        )

        (
            ip_train,
            ip_validation,
        ) = deterministic_partition(
            injection_positive,
            [
                args.injection_train_pos,
                args.injection_val_pos,
            ],
            args.seed + 21,
        )

        (
            in_train,
            in_validation,
        ) = deterministic_partition(
            injection_negative,
            [
                args.injection_train_neg,
                args.injection_val_neg,
            ],
            args.seed + 22,
        )

        injection_train = (
            enrich(
                ip_train,
                label=1,
                risk_family="injection",
                domain="npc_injection",
                split="train",
                prefix="inj_pos",
            )
            + enrich(
                in_train,
                label=0,
                risk_family="injection",
                domain="npc_benign",
                split="train",
                prefix="inj_neg",
            )
        )

        injection_validation = (
            enrich(
                ip_validation,
                label=1,
                risk_family="injection",
                domain="npc_injection",
                split="validation",
                prefix="inj_pos",
            )
            + enrich(
                in_validation,
                label=0,
                risk_family="injection",
                domain="npc_benign",
                split="validation",
                prefix="inj_neg",
            )
        )

        injection_source_mode = (
            "resampled_positive_negative_pools"
        )

    # -------------------------------------------------------------
    # Combined manifests
    # -------------------------------------------------------------

    train = (
        harm_train
        + injection_train
    )

    validation = (
        harm_validation
        + injection_validation
    )

    assert_no_overlap(
        {
            "train": train,
            "validation": validation,
        }
    )

    outputs = {
        "harm_train": harm_train,
        "harm_validation": (
            harm_validation
        ),
        "injection_train": (
            injection_train
        ),
        "injection_validation": (
            injection_validation
        ),
        "train": train,
        "validation": validation,
    }

    for name, rows in outputs.items():
        write_jsonl(
            output_dir
            / f"{name}.jsonl",
            rows,
        )

    summary = {
        "format_version": 1,
        "seed": args.seed,
        "purpose": (
            "Multi-RAAC detector "
            "construction only"
        ),
        "injection_source_mode": (
            injection_source_mode
        ),
        "harm_source_unique_counts": {
            "positive": (
                len(harm_positive)
            ),
            "negative": (
                len(harm_negative)
            ),
        },
        "splits": {
            name: summarize(rows)
            for name, rows
            in outputs.items()
        },
        "design": {
            "risk_families": [
                "harm",
                "injection",
            ],
            "threshold_calibration": (
                "validation only"
            ),
            "held_out_policy": (
                "No held-out evaluation "
                "data is used for "
                "training or calibration."
            ),
        },
    }

    (
        output_dir
        / "dataset_summary.json"
    ).write_text(
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

    print(
        "[DONE]",
        output_dir,
    )


if __name__ == "__main__":
    main()
