import pytest
import torch

from training.extract_hidden_states import (
    last_non_padding_indices,
    resolve_dtype,
)


def test_last_non_padding_indices_right_padding():
    mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ]
    )

    indices = (
        last_non_padding_indices(
            mask
        )
    )

    assert indices.tolist() == [
        2,
        1,
    ]


def test_last_non_padding_indices_left_padding():
    mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
        ]
    )

    indices = (
        last_non_padding_indices(
            mask
        )
    )

    assert indices.tolist() == [
        4,
        4,
    ]


def test_last_non_padding_indices_rejects_empty_sequence():
    mask = torch.tensor(
        [
            [1, 1, 0],
            [0, 0, 0],
        ]
    )

    with pytest.raises(
        RuntimeError
    ):
        last_non_padding_indices(
            mask
        )


def test_resolve_dtype():
    assert (
        resolve_dtype("float16")
        is torch.float16
    )

    assert (
        resolve_dtype("bfloat16")
        is torch.bfloat16
    )

    assert (
        resolve_dtype("float32")
        is torch.float32
    )
