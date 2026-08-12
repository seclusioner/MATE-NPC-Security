from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def choose_youden_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[
    float,
    dict[str, float],
]:
    """
    Maximize Youden's statistic:

        J = TPR - FPR

    Tie breaking follows the formal experiment:
    lower FPR first, then higher threshold.
    """
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    fpr, tpr, thresholds = (
        roc_curve(
            labels,
            scores,
        )
    )

    candidates = []

    for index, threshold in enumerate(
        thresholds
    ):
        if not np.isfinite(
            threshold
        ):
            continue

        candidates.append(
            (
                float(
                    tpr[index]
                    - fpr[index]
                ),
                -float(
                    fpr[index]
                ),
                float(threshold),
                index,
            )
        )

    if not candidates:
        raise RuntimeError(
            "No finite ROC threshold."
        )

    (
        _,
        _,
        threshold,
        best,
    ) = max(candidates)

    return (
        float(threshold),
        {
            "validation_tpr": float(
                tpr[best]
            ),
            "validation_fpr": float(
                fpr[best]
            ),
            "validation_youden_j": float(
                tpr[best]
                - fpr[best]
            ),
        },
    )


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    predictions = (
        scores
        >= float(threshold)
    ).astype(
        np.int64
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    tpr = (
        float(
            tp / (tp + fn)
        )
        if tp + fn
        else float("nan")
    )

    fpr = (
        float(
            fp / (fp + tn)
        )
        if fp + tn
        else float("nan")
    )

    specificity = (
        float(
            tn / (tn + fp)
        )
        if tn + fp
        else float("nan")
    )

    try:
        auroc = float(
            roc_auc_score(
                labels,
                scores,
            )
        )
    except ValueError:
        auroc = float("nan")

    try:
        auprc = float(
            average_precision_score(
                labels,
                scores,
            )
        )
    except ValueError:
        auprc = float("nan")

    return {
        "n": int(
            len(labels)
        ),
        "positive_n": int(
            labels.sum()
        ),
        "negative_n": int(
            (labels == 0).sum()
        ),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": float(
            threshold
        ),
        "tpr": tpr,
        "fpr": fpr,
        "specificity": (
            specificity
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "youden_j": float(
            tpr - fpr
        ),
        "auroc": auroc,
        "auprc": auprc,
    }



def select_threshold_under_fpr(
    labels: np.ndarray,
    scores: np.ndarray,
    target_fpr: float,
) -> tuple[
    float,
    dict[str, float],
]:
    """
    Choose the highest-TPR threshold satisfying:

        FPR <= target_fpr

    Tie breaking follows the formal Low-FPR Sparse experiment:
    choose the higher threshold for the more conservative
    operating point.
    """

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if not (
        0.0 <= target_fpr < 1.0
    ):
        raise ValueError(
            "target_fpr must be in [0, 1)."
        )

    if set(
        labels.tolist()
    ) != {0, 1}:
        raise ValueError(
            "Both binary classes are required "
            "for FPR calibration."
        )

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
    )

    valid = np.flatnonzero(
        fpr <= target_fpr + 1.0e-12
    )

    if valid.size == 0:
        threshold = float(
            np.nextafter(
                np.max(scores),
                np.inf,
            )
        )
    else:
        best_tpr = np.max(
            tpr[valid]
        )

        candidates = valid[
            np.isclose(
                tpr[valid],
                best_tpr,
                atol=1.0e-12,
            )
        ]

        best = candidates[
            np.argmax(
                thresholds[candidates]
            )
        ]

        threshold = float(
            thresholds[best]
        )

    predictions = (
        scores >= threshold
    )

    positives = labels == 1
    negatives = labels == 0

    actual_tpr = float(
        predictions[
            positives
        ].mean()
    )

    actual_fpr = float(
        predictions[
            negatives
        ].mean()
    )

    return (
        threshold,
        {
            "target_fpr": float(
                target_fpr
            ),
            "actual_tpr": actual_tpr,
            "actual_fpr": actual_fpr,
            "youden_j": (
                actual_tpr
                - actual_fpr
            ),
        },
    )
