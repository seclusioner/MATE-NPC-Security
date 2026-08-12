import json

import pytest

from training.prepare_data import (
    assert_no_overlap,
    deterministic_partition,
    load_text_pool,
    parse_binary_label,
    text_hash,
)


def test_parse_binary_label():
    assert parse_binary_label(1) == 1
    assert parse_binary_label(0) == 0
    assert parse_binary_label(
        "harmful"
    ) == 1
    assert parse_binary_label(
        "benign"
    ) == 0
    assert parse_binary_label(
        "injection"
    ) == 1
    assert parse_binary_label(
        "safe"
    ) == 0

    with pytest.raises(
        ValueError
    ):
        parse_binary_label(
            "unknown-label"
        )


def test_load_text_pool_deduplicates(
    tmp_path,
):
    path = (
        tmp_path
        / "data.jsonl"
    )

    rows = [
        {
            "player_input": (
                "Hello world"
            )
        },
        {
            "player_input": (
                "Hello world"
            )
        },
        {
            "player_input": (
                "Different text"
            )
        },
    ]

    path.write_text(
        "".join(
            json.dumps(row)
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    output = load_text_pool(
        path
    )

    assert len(output) == 2

    assert (
        output[0]["source"]
        == "data.jsonl"
    )


def test_partition_is_deterministic():
    rows = [
        {
            "text": str(index),
            "text_hash": (
                text_hash(
                    str(index)
                )
            ),
        }
        for index in range(20)
    ]

    first = deterministic_partition(
        rows,
        [5, 5],
        seed=42,
    )

    second = deterministic_partition(
        rows,
        [5, 5],
        seed=42,
    )

    assert first == second

    first_hashes = {
        row["text_hash"]
        for part in first
        for row in part
    }

    assert len(
        first_hashes
    ) == 10


def test_overlap_detection():
    shared = {
        "text_hash": text_hash(
            "same text"
        )
    }

    with pytest.raises(
        RuntimeError
    ):
        assert_no_overlap(
            {
                "train": [shared],
                "validation": [
                    shared
                ],
            }
        )
