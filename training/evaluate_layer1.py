from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


EPS = 1.0e-8


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

def load_cache(
    path: str | Path,
) -> dict[str, Any]:
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def load_bundle(
    path: str | Path,
) -> dict[str, Any]:
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def write_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Detector scoring
# ---------------------------------------------------------------------

def _bundle_value(
    mapping: dict[Any, Any],
    layer: int,
):
    if layer in mapping:
        return mapping[layer]

    key = str(layer)

    if key in mapping:
        return mapping[key]

    raise KeyError(
        f"Layer {layer} missing "
        "from detector bundle."
    )


def score_mean_difference(
    features: torch.Tensor,
    bundle: dict[str, Any],
) -> np.ndarray:
    layers = [
        int(layer)
        for layer in bundle["layers"]
    ]

    selected = (
        features[:, layers, :]
        .float()
    )

    directions = torch.stack(
        [
            _bundle_value(
                bundle["vec"],
                layer,
            ).float()
            for layer in layers
        ]
    )

    bias = torch.tensor(
        [
            float(
                _bundle_value(
                    bundle["bias"],
                    layer,
                )
            )
            for layer in layers
        ],
        dtype=torch.float32,
    )

    scores = (
        torch.einsum(
            "nlh,lh->nl",
            selected,
            directions,
        )
        - bias
    ).mean(
        dim=1
    )

    return (
        scores
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )


def score_sparse_logistic(
    features: torch.Tensor,
    bundle: dict[str, Any],
) -> np.ndarray:
    layers = [
        int(layer)
        for layer in bundle["layers"]
    ]

    vector = (
        features[:, layers, :]
        .float()
        .reshape(
            features.shape[0],
            -1,
        )
    )

    indices = bundle.get(
        "feature_indices"
    )

    if indices is not None:
        if not isinstance(
            indices,
            torch.Tensor,
        ):
            indices = torch.as_tensor(
                indices,
                dtype=torch.long,
            )

        vector = vector.index_select(
            1,
            indices.long(),
        )

    mean = torch.as_tensor(
        bundle["scaler_mean"],
        dtype=torch.float32,
    )

    scale = torch.as_tensor(
        bundle["scaler_scale"],
        dtype=torch.float32,
    ).clamp_min(
        EPS
    )

    coef = torch.as_tensor(
        bundle["coef"],
        dtype=torch.float32,
    ).reshape(-1)

    intercept = float(
        bundle["intercept"]
    )

    logits = (
        (
            vector - mean
        )
        / scale
    ) @ coef + intercept

    return (
        torch.sigmoid(
            logits
        )
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
    )


def score_hidden_bundle(
    features: torch.Tensor,
    bundle: dict[str, Any],
) -> np.ndarray:
    method = str(
        bundle.get(
            "method",
            "",
        )
    )

    if method == "mean_difference":
        return score_mean_difference(
            features,
            bundle,
        )

    # Sparse public artifacts can be recognized
    # structurally rather than by one hard-coded
    # historical method name.
    sparse_keys = {
        "feature_indices",
        "scaler_mean",
        "scaler_scale",
        "coef",
        "intercept",
    }

    if sparse_keys.issubset(
        bundle.keys()
    ):
        return score_sparse_logistic(
            features,
            bundle,
        )

    raise ValueError(
        "Unsupported detector bundle "
        f"method={method!r}"
    )


def normalized_margin(
    scores: np.ndarray,
    threshold: float,
    score_scale: float,
) -> np.ndarray:
    scale = max(
        abs(
            float(
                score_scale
            )
        ),
        EPS,
    )

    return (
        np.asarray(
            scores,
            dtype=np.float64,
        )
        - float(
            threshold
        )
    ) / scale


def score_multiraac(
    features: torch.Tensor,
    harm_bundle: dict[str, Any],
    injection_bundle: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    harm_scores = score_hidden_bundle(
        features,
        harm_bundle,
    )

    injection_scores = score_hidden_bundle(
        features,
        injection_bundle,
    )

    harm_margin = normalized_margin(
        harm_scores,
        float(
            harm_bundle["threshold"]
        ),
        float(
            harm_bundle["score_scale"]
        ),
    )

    injection_margin = normalized_margin(
        injection_scores,
        float(
            injection_bundle[
                "threshold"
            ]
        ),
        float(
            injection_bundle[
                "score_scale"
            ]
        ),
    )

    fused = np.maximum(
        harm_margin,
        injection_margin,
    )

    return (
        fused,
        harm_scores,
        injection_scores,
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

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
        )
        .ravel()
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
            (
                labels == 0
            ).sum()
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
        "balanced_accuracy": (
            float(
                balanced_accuracy_score(
                    labels,
                    predictions,
                )
            )
        ),
        "youden_j": (
            tpr - fpr
        ),
        "auroc": auroc,
        "auprc": auprc,
    }


def evaluate_dataset(
    *,
    dataset: str,
    method: str,
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    threshold: float = 0.0,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    groups = np.asarray(
        [
            str(
                row.get(
                    "evaluation_group",
                    "attack",
                )
            )
            for row in rows
        ]
    )

    labels = np.asarray(
        [
            int(
                row["label"]
            )
            for row in rows
        ],
        dtype=np.int64,
    )

    # Formal evaluation:
    # authorized rows are not part of the
    # binary confusion matrix.
    binary_mask = np.isin(
        groups,
        [
            "attack",
            "benign",
        ],
    )

    metrics = binary_metrics(
        labels[
            binary_mask
        ],
        scores[
            binary_mask
        ],
        threshold,
    )

    predictions = (
        scores
        >= threshold
    )

    authorized_mask = (
        groups
        == "authorized"
    )

    authorized_trigger = (
        float(
            predictions[
                authorized_mask
            ].mean()
        )
        if authorized_mask.any()
        else float("nan")
    )

    summary = {
        "dataset": dataset,
        "method": method,
        **metrics,
        "authorized_n": int(
            authorized_mask.sum()
        ),
        "authorized_trigger": (
            authorized_trigger
        ),
    }

    prediction_rows = []

    for index, (
        row,
        score,
        prediction,
    ) in enumerate(
        zip(
            rows,
            scores,
            predictions,
        )
    ):
        prediction_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "sample_id": row.get(
                    "id",
                    f"{dataset}_{index:04d}",
                ),
                "label": int(
                    row["label"]
                ),
                "evaluation_group": (
                    row.get(
                        "evaluation_group",
                        "",
                    )
                ),
                "risk_family": (
                    row.get(
                        "risk_family",
                        "",
                    )
                ),
                "score": float(
                    score
                ),
                "threshold": float(
                    threshold
                ),
                "triggered": bool(
                    prediction
                ),
            }
        )

    return (
        summary,
        prediction_rows,
    )


def safe_div(
    numerator: float,
    denominator: float,
) -> float:
    if not denominator:
        return float("nan")

    return (
        numerator
        / denominator
    )


def aggregate_heldout(
    summaries: list[
        dict[str, Any]
    ],
    *,
    method: str,
) -> dict[str, Any]:
    selected = [
        row
        for row in summaries
        if row["method"] == method
    ]

    tp = sum(
        int(row["tp"])
        for row in selected
    )

    fp = sum(
        int(row["fp"])
        for row in selected
    )

    tn = sum(
        int(row["tn"])
        for row in selected
    )

    fn = sum(
        int(row["fn"])
        for row in selected
    )

    tpr = safe_div(
        tp,
        tp + fn,
    )

    fpr = safe_div(
        fp,
        fp + tn,
    )

    specificity = safe_div(
        tn,
        tn + fp,
    )

    precision = safe_div(
        tp,
        tp + fp,
    )

    if (
        np.isnan(
            precision
        )
        or np.isnan(
            tpr
        )
        or (
            precision
            + tpr
        ) == 0
    ):
        f1 = 0.0
    else:
        f1 = (
            2.0
            * precision
            * tpr
            / (
                precision
                + tpr
            )
        )

    balanced_accuracy = (
        (
            tpr
            + specificity
        )
        / 2.0
    )

    authorized_n = sum(
        int(
            row.get(
                "authorized_n",
                0,
            )
        )
        for row in selected
    )

    authorized_hits = sum(
        float(
            row.get(
                "authorized_trigger",
                0.0,
            )
        )
        * int(
            row.get(
                "authorized_n",
                0,
            )
        )
        for row in selected
        if not np.isnan(
            float(
                row.get(
                    "authorized_trigger",
                    float("nan"),
                )
            )
        )
    )

    authorized_trigger = (
        authorized_hits
        / authorized_n
        if authorized_n
        else float("nan")
    )

    return {
        "method": method,
        "positive_n": (
            tp + fn
        ),
        "negative_n": (
            tn + fp
        ),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "youden_j": (
            tpr - fpr
        ),
        "authorized_n": (
            authorized_n
        ),
        "authorized_trigger": (
            authorized_trigger
        ),
        "aggregation": (
            "Micro aggregate over "
            "attack/benign samples. "
            "Authorized samples are "
            "excluded from the binary "
            "confusion matrix and "
            "reported separately."
        ),
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_named_paths(
    values: list[str],
) -> dict[str, Path]:
    output = {}

    for value in values:
        if "=" not in value:
            raise ValueError(
                "Expected NAME=PATH, "
                f"got {value!r}"
            )

        name, path = (
            value.split(
                "=",
                1,
            )
        )

        output[
            name.strip()
        ] = Path(
            path.strip()
        ).expanduser()

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Original or Sparse "
            "Multi-RAAC Layer-1 detectors "
            "on cached hidden states."
        )
    )

    parser.add_argument(
        "--detectors",
        nargs="+",
        required=True,
        help=(
            "NAME=DETECTOR_DIR. "
            "Each directory must contain "
            "harm_detector.pt and "
            "injection_detector.pt."
        ),
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help=(
            "NAME=CACHE_PT."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    detector_dirs = (
        parse_named_paths(
            args.detectors
        )
    )

    dataset_paths = (
        parse_named_paths(
            args.datasets
        )
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_summaries = []
    all_predictions = []

    for method_name, detector_dir in (
        detector_dirs.items()
    ):
        harm_bundle = load_bundle(
            detector_dir
            / "harm_detector.pt"
        )

        injection_bundle = load_bundle(
            detector_dir
            / "injection_detector.pt"
        )

        for dataset_name, cache_path in (
            dataset_paths.items()
        ):
            cache = load_cache(
                cache_path
            )

            rows = cache[
                "rows"
            ]

            features = cache[
                "features"
            ]

            (
                fused_scores,
                _,
                _,
            ) = score_multiraac(
                features,
                harm_bundle,
                injection_bundle,
            )

            (
                summary,
                prediction_rows,
            ) = evaluate_dataset(
                dataset=dataset_name,
                method=method_name,
                rows=rows,
                scores=fused_scores,
                threshold=0.0,
            )

            all_summaries.append(
                summary
            )

            all_predictions.extend(
                prediction_rows
            )

    combined = [
        aggregate_heldout(
            all_summaries,
            method=method_name,
        )
        for method_name
        in detector_dirs
    ]

    write_csv(
        output_dir
        / "dataset_summary.csv",
        all_summaries,
    )

    write_csv(
        output_dir
        / "predictions.csv",
        all_predictions,
    )

    write_csv(
        output_dir
        / "combined_summary.csv",
        combined,
    )

    (
        output_dir
        / "dataset_summary.json"
    ).write_text(
        json.dumps(
            all_summaries,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (
        output_dir
        / "combined_summary.json"
    ).write_text(
        json.dumps(
            combined,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "===== PER-DATASET ====="
    )

    fields = [
        "dataset",
        "method",
        "positive_n",
        "negative_n",
        "tpr",
        "fpr",
        "precision",
        "f1",
        "balanced_accuracy",
        "authorized_n",
        "authorized_trigger",
    ]

    print(
        ",".join(fields)
    )

    for row in all_summaries:
        print(
            ",".join(
                str(
                    row.get(
                        field,
                        "",
                    )
                )
                for field in fields
            )
        )

    print()
    print(
        "===== COMBINED HELD-OUT ====="
    )

    combined_fields = [
        "method",
        "positive_n",
        "negative_n",
        "tp",
        "fp",
        "tn",
        "fn",
        "tpr",
        "fpr",
        "precision",
        "f1",
        "balanced_accuracy",
        "authorized_n",
        "authorized_trigger",
    ]

    print(
        ",".join(
            combined_fields
        )
    )

    for row in combined:
        print(
            ",".join(
                str(
                    row.get(
                        field,
                        "",
                    )
                )
                for field
                in combined_fields
            )
        )

    print()
    print(
        "[SAVE]",
        output_dir,
    )


if __name__ == "__main__":
    main()
