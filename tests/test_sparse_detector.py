import math

import torch

from gigax.security.sparse_detector import (
    SparseLogisticRiskDetector,
)


def test_sparse_logistic_artifact_math(
    tmp_path,
):
    """
    Synthetic logistic_hidden artifact.

    flattened:
        [3, 0, 0, 6]

    selected:
        indices [0, 3] -> [3, 6]

    standardized:
        [(3-1)/2, (6-2)/4]
        = [1, 1]

    logit:
        1*1 + 1*(-2) + 0.5
        = -0.5

    probability:
        sigmoid(-0.5)
    """

    path = tmp_path / "sparse.pt"

    bundle = {
        "format_version": 2,
        "method": "logistic_hidden",
        "name": "test_sparse",
        "risk_type": "harm_detector",
        "layers": [0, 1],
        "hidden_size": 2,
        "feature_indices": torch.tensor(
            [0, 3],
            dtype=torch.long,
        ),
        "scaler_mean": torch.tensor(
            [1.0, 2.0]
        ),
        "scaler_scale": torch.tensor(
            [2.0, 4.0]
        ),
        "coef": torch.tensor(
            [1.0, -2.0]
        ),
        "intercept": 0.5,
        "threshold": 0.5,
        "score_scale": 0.25,
        "top_k": 2,
    }

    torch.save(bundle, path)

    model = torch.nn.Linear(
        1,
        1,
    )

    detector = (
        SparseLogisticRiskDetector(
            model=model,
            tokenizer=None,
            detector_path=path,
            device="cpu",
            chat_template_mode="plain",
        )
    )

    score = detector.score_flattened(
        torch.tensor(
            [
                3.0,
                0.0,
                0.0,
                6.0,
            ]
        )
    )

    expected = (
        1.0
        / (
            1.0
            + math.exp(0.5)
        )
    )

    assert abs(
        score - expected
    ) < 1e-6
