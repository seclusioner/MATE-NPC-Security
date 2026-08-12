import numpy as np
import torch

from training.evaluate_layer1 import (
    aggregate_heldout,
    evaluate_dataset,
    normalized_margin,
    score_mean_difference,
)


def test_normalized_margin():
    scores = np.asarray(
        [
            10.0,
            20.0,
            30.0,
        ]
    )

    result = normalized_margin(
        scores,
        threshold=20.0,
        score_scale=10.0,
    )

    assert np.allclose(
        result,
        [
            -1.0,
            0.0,
            1.0,
        ],
    )


def test_mean_difference_scoring():
    features = torch.zeros(
        2,
        3,
        2,
    )

    features[
        0,
        1,
    ] = torch.tensor(
        [
            2.0,
            0.0,
        ]
    )

    features[
        1,
        1,
    ] = torch.tensor(
        [
            -2.0,
            0.0,
        ]
    )

    bundle = {
        "method": (
            "mean_difference"
        ),
        "layers": [1],
        "vec": {
            1: torch.tensor(
                [
                    1.0,
                    0.0,
                ]
            )
        },
        "bias": {
            1: 0.0
        },
    }

    scores = (
        score_mean_difference(
            features,
            bundle,
        )
    )

    assert np.allclose(
        scores,
        [
            2.0,
            -2.0,
        ],
    )


def test_authorized_is_excluded_from_binary_metrics():
    rows = [
        {
            "id": "attack",
            "label": 1,
            "evaluation_group": (
                "attack"
            ),
        },
        {
            "id": "benign",
            "label": 0,
            "evaluation_group": (
                "benign"
            ),
        },
        {
            "id": "authorized",
            "label": 1,
            "evaluation_group": (
                "authorized"
            ),
        },
    ]

    scores = np.asarray(
        [
            1.0,
            -1.0,
            1.0,
        ]
    )

    summary, _ = (
        evaluate_dataset(
            dataset="test",
            method="method",
            rows=rows,
            scores=scores,
            threshold=0.0,
        )
    )

    assert summary["n"] == 2
    assert summary["tp"] == 1
    assert summary["tn"] == 1

    assert (
        summary[
            "authorized_n"
        ]
        == 1
    )

    assert (
        summary[
            "authorized_trigger"
        ]
        == 1.0
    )


def test_micro_aggregation():
    summaries = [
        {
            "method": "m",
            "tp": 8,
            "fp": 2,
            "tn": 8,
            "fn": 2,
            "authorized_n": 0,
            "authorized_trigger": (
                float("nan")
            ),
        },
        {
            "method": "m",
            "tp": 9,
            "fp": 1,
            "tn": 9,
            "fn": 1,
            "authorized_n": 2,
            "authorized_trigger": 0.5,
        },
    ]

    result = aggregate_heldout(
        summaries,
        method="m",
    )

    assert result["tp"] == 17
    assert result["fp"] == 3
    assert result["tn"] == 17
    assert result["fn"] == 3

    assert np.isclose(
        result["tpr"],
        0.85,
    )

    assert np.isclose(
        result["fpr"],
        0.15,
    )

    assert (
        result[
            "authorized_trigger"
        ]
        == 0.5
    )
