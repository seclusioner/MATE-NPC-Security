from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.feature_selection import (
    f_classif,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
)
from sklearn.preprocessing import (
    StandardScaler,
)

from training.calibration import (
    binary_metrics,
    select_threshold_under_fpr,
)
from training.common import (
    EPS,
    load_cache,
    normalized_margin,
    score_hidden_bundle,
    validate_layers,
)


RISK_FAMILIES = (
    "harm",
    "injection",
)


def parse_int_list(
    value: str,
) -> list[int]:
    result = sorted(
        {
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        }
    )

    if not result:
        raise ValueError(
            "No integer values supplied."
        )

    return result


def parse_float_list(
    value: str,
) -> list[float]:
    result = sorted(
        {
            float(item.strip())
            for item in value.split(",")
            if item.strip()
        }
    )

    if not result:
        raise ValueError(
            "No float values supplied."
        )

    if any(
        item < 0.0
        or item >= 1.0
        for item in result
    ):
        raise ValueError(
            "FPR targets must be "
            "in [0, 1)."
        )

    return result


def flatten_layers(
    features: torch.Tensor,
    layers: list[int],
) -> np.ndarray:
    """
    Concatenate selected decoder-layer representations:

        [N, selected_layers, H]
                ->
        [N, selected_layers * H]
    """
    return (
        features[:, layers, :]
        .float()
        .reshape(
            features.shape[0],
            -1,
        )
        .cpu()
        .numpy()
    )


def pooled_head_subset(
    cache: dict[str, Any],
    risk_family: str,
) -> tuple[
    torch.Tensor,
    np.ndarray,
    list[dict[str, Any]],
]:
    """
    Low-FPR Sparse policy:

    Positive:
        only positives belonging to this risk family.

    Negative:
        ALL benign samples across all risk families.
    """
    rows = cache["rows"]

    indices = [
        index
        for index, row
        in enumerate(rows)
        if (
            int(row["label"]) == 0
            or (
                int(row["label"]) == 1
                and str(
                    row.get(
                        "risk_family",
                        "",
                    )
                ).lower()
                == risk_family.lower()
            )
        )
    ]

    selected_rows = [
        rows[index]
        for index in indices
    ]

    labels = np.asarray(
        [
            int(row["label"])
            for row
            in selected_rows
        ],
        dtype=np.int64,
    )

    if set(
        labels.tolist()
    ) != {0, 1}:
        raise RuntimeError(
            f"{risk_family}: both labels "
            "are required."
        )

    return (
        cache["features"][
            indices
        ].float(),
        labels,
        selected_rows,
    )


def select_top_k(
    train_x: np.ndarray,
    train_y: np.ndarray,
    top_k: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    f_values, _ = f_classif(
        train_x,
        train_y,
    )

    f_values = np.nan_to_num(
        f_values,
        nan=-np.inf,
        posinf=np.finfo(
            np.float64
        ).max,
        neginf=-np.inf,
    )

    top_k = min(
        int(top_k),
        train_x.shape[1],
    )

    if top_k <= 0:
        raise ValueError(
            "top_k must be positive."
        )

    indices = np.argpartition(
        f_values,
        -top_k,
    )[-top_k:]

    indices = indices[
        np.argsort(
            f_values[indices]
        )[::-1]
    ]

    return (
        indices.astype(
            np.int64
        ),
        f_values,
    )


def fit_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    selected: np.ndarray,
    *,
    seed: int,
) -> tuple[
    StandardScaler,
    LogisticRegression,
]:
    scaler = StandardScaler()

    transformed = (
        scaler.fit_transform(
            train_x[:, selected]
        )
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=seed,
    )

    classifier.fit(
        transformed,
        train_y,
    )

    return (
        scaler,
        classifier,
    )


def cross_validate_k(
    features: np.ndarray,
    labels: np.ndarray,
    top_k_grid: list[int],
    *,
    target_fpr: float,
    folds: int,
    seed: int,
) -> list[
    dict[str, Any]
]:
    """
    K selection is strictly train-only.

    Each fold independently performs:
      ANOVA feature selection
      -> scaling
      -> logistic fit
      -> low-FPR threshold calibration
    """
    class_counts = np.bincount(
        labels
    )

    if len(class_counts) < 2:
        raise RuntimeError(
            "Both classes are required."
        )

    n_splits = min(
        int(folds),
        int(
            class_counts.min()
        ),
    )

    if n_splits < 2:
        raise RuntimeError(
            "Insufficient data for "
            "stratified cross-validation."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    output = []

    for top_k in top_k_grid:
        fold_rows = []

        for fold, (
            train_idx,
            validation_idx,
        ) in enumerate(
            splitter.split(
                features,
                labels,
            )
        ):
            train_x = features[
                train_idx
            ]

            train_y = labels[
                train_idx
            ]

            validation_x = features[
                validation_idx
            ]

            validation_y = labels[
                validation_idx
            ]

            selected, _ = (
                select_top_k(
                    train_x,
                    train_y,
                    top_k,
                )
            )

            scaler, classifier = (
                fit_classifier(
                    train_x,
                    train_y,
                    selected,
                    seed=(
                        seed + fold
                    ),
                )
            )

            scores = (
                classifier
                .predict_proba(
                    scaler.transform(
                        validation_x[
                            :,
                            selected,
                        ]
                    )
                )[:, 1]
            )

            (
                threshold,
                low_fpr,
            ) = (
                select_threshold_under_fpr(
                    validation_y,
                    scores,
                    target_fpr,
                )
            )

            fold_rows.append(
                {
                    "fold": int(
                        fold
                    ),
                    "threshold": float(
                        threshold
                    ),
                    "low_fpr_tpr": float(
                        low_fpr[
                            "actual_tpr"
                        ]
                    ),
                    "low_fpr_fpr": float(
                        low_fpr[
                            "actual_fpr"
                        ]
                    ),
                    "auroc": float(
                        roc_auc_score(
                            validation_y,
                            scores,
                        )
                    ),
                    "auprc": float(
                        average_precision_score(
                            validation_y,
                            scores,
                        )
                    ),
                }
            )

        output.append(
            {
                "top_k": int(
                    min(
                        top_k,
                        features.shape[1],
                    )
                ),
                "folds": int(
                    n_splits
                ),
                "mean_low_fpr_tpr": float(
                    np.mean(
                        [
                            row[
                                "low_fpr_tpr"
                            ]
                            for row
                            in fold_rows
                        ]
                    )
                ),
                "mean_low_fpr_fpr": float(
                    np.mean(
                        [
                            row[
                                "low_fpr_fpr"
                            ]
                            for row
                            in fold_rows
                        ]
                    )
                ),
                "mean_auroc": float(
                    np.mean(
                        [
                            row["auroc"]
                            for row
                            in fold_rows
                        ]
                    )
                ),
                "mean_auprc": float(
                    np.mean(
                        [
                            row["auprc"]
                            for row
                            in fold_rows
                        ]
                    )
                ),
                "fold_metrics": (
                    fold_rows
                ),
            }
        )

    # Deduplicate K values that may collapse when K > feature count.
    deduplicated = {}

    for row in output:
        k = int(
            row["top_k"]
        )

        if k not in deduplicated:
            deduplicated[k] = row

    return list(
        deduplicated.values()
    )


def choose_k(
    rows: list[
        dict[str, Any]
    ],
) -> int:
    """
    Formal deterministic ranking:

    1. higher low-FPR TPR
    2. lower low-FPR FPR
    3. higher AUROC
    4. higher AUPRC
    5. smaller K
    """
    if not rows:
        raise ValueError(
            "No CV results supplied."
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            -float(
                row[
                    "mean_low_fpr_tpr"
                ]
            ),
            float(
                row[
                    "mean_low_fpr_fpr"
                ]
            ),
            -float(
                row["mean_auroc"]
            ),
            -float(
                row["mean_auprc"]
            ),
            int(
                row["top_k"]
            ),
        ),
    )

    return int(
        ranked[0]["top_k"]
    )


def feature_counts_by_layer(
    indices: np.ndarray,
    *,
    layers: list[int],
    hidden_size: int,
) -> dict[str, int]:
    counts = {
        str(layer): 0
        for layer in layers
    }

    for index in indices.tolist():
        layer_position = (
            int(index)
            // hidden_size
        )

        counts[
            str(
                layers[
                    layer_position
                ]
            )
        ] += 1

    return counts


def train_sparse_head(
    train_cache: dict[str, Any],
    validation_cache: dict[str, Any],
    *,
    risk_family: str,
    layers: list[int],
    top_k: int,
    target_head_fpr: float,
    cv_metrics: list[
        dict[str, Any]
    ],
    seed: int,
) -> dict[str, Any]:
    (
        train_features,
        train_labels,
        _,
    ) = pooled_head_subset(
        train_cache,
        risk_family,
    )

    (
        validation_features,
        validation_labels,
        _,
    ) = pooled_head_subset(
        validation_cache,
        risk_family,
    )

    train_x = flatten_layers(
        train_features,
        layers,
    )

    validation_x = flatten_layers(
        validation_features,
        layers,
    )

    selected, f_values = (
        select_top_k(
            train_x,
            train_labels,
            top_k,
        )
    )

    scaler, classifier = (
        fit_classifier(
            train_x,
            train_labels,
            selected,
            seed=seed,
        )
    )

    bundle = {
        "format_version": 2,
        "method": (
            "logistic_hidden"
        ),
        "name": (
            f"lowfpr_sparse_"
            f"{risk_family}_"
            f"k{len(selected)}"
        ),
        "risk_type": (
            f"{risk_family}_detector"
        ),
        "architecture": (
            "category-specific ANOVA "
            "logistic head with pooled "
            "benign negatives"
        ),
        "layers": list(
            layers
        ),
        "hidden_size": int(
            train_features.shape[-1]
        ),
        "feature_indices": (
            torch.tensor(
                selected,
                dtype=torch.long,
            )
        ),
        "selected_f_values": (
            torch.tensor(
                f_values[selected],
                dtype=torch.float32,
            )
        ),
        "selected_feature_count_by_layer": (
            feature_counts_by_layer(
                selected,
                layers=layers,
                hidden_size=int(
                    train_features
                    .shape[-1]
                ),
            )
        ),
        "scaler_mean": torch.tensor(
            scaler.mean_,
            dtype=torch.float32,
        ),
        "scaler_scale": torch.tensor(
            scaler.scale_,
            dtype=torch.float32,
        ),
        "coef": torch.tensor(
            classifier.coef_[0],
            dtype=torch.float32,
        ),
        "intercept": float(
            classifier.intercept_[0]
        ),
        "top_k": int(
            len(selected)
        ),
        "cv_grid_metrics": (
            cv_metrics
        ),
        "training_negative_policy": (
            "all benign samples from "
            "all risk families"
        ),
        "calibration_negative_policy": (
            "all benign validation samples "
            "from all risk families"
        ),
        "train_n": int(
            len(train_labels)
        ),
        "validation_n": int(
            len(validation_labels)
        ),
        "train_positive_n": int(
            (
                train_labels == 1
            ).sum()
        ),
        "train_negative_n": int(
            (
                train_labels == 0
            ).sum()
        ),
        "validation_positive_n": int(
            (
                validation_labels == 1
            ).sum()
        ),
        "validation_negative_n": int(
            (
                validation_labels == 0
            ).sum()
        ),
        "train_manifest_sha256": (
            train_cache[
                "manifest_sha256"
            ]
        ),
        "validation_manifest_sha256": (
            validation_cache[
                "manifest_sha256"
            ]
        ),
    }

    # Runtime-equivalent float32 score path.
    runtime_scores = (
        score_hidden_bundle(
            validation_features,
            bundle,
        )
    )

    (
        threshold,
        calibration,
    ) = (
        select_threshold_under_fpr(
            validation_labels,
            runtime_scores,
            target_head_fpr,
        )
    )

    bundle["threshold"] = float(
        threshold
    )

    bundle["score_scale"] = max(
        float(
            np.std(
                runtime_scores
            )
        ),
        EPS,
    )

    bundle[
        "runtime_calibration"
    ] = {
        **calibration,
        "comparison": (
            "score >= threshold"
        ),
        "runtime_scorer": (
            "serialized logistic bundle"
        ),
        "metrics": binary_metrics(
            validation_labels,
            runtime_scores,
            threshold,
        ),
    }

    return bundle


def fusion_slug(
    target: float,
) -> str:
    return (
        str(float(target))
        .replace(".", "p")
    )


def build_lowfpr_sparse_multiraac(
    *,
    train_cache: dict[str, Any],
    validation_cache: dict[str, Any],
    output_root: Path,
    layers: list[int],
    top_k_grid: list[int],
    cv_folds: int,
    head_target_fpr: float,
    fusion_target_fprs: list[
        float
    ],
    seed: int,
) -> dict[str, Any]:
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_layers(
        train_cache,
        layers,
    )

    validate_layers(
        validation_cache,
        layers,
    )

    if (
        train_cache[
            "features"
        ].shape[2]
        != validation_cache[
            "features"
        ].shape[2]
    ):
        raise ValueError(
            "Train and validation "
            "hidden sizes differ."
        )

    cv_metrics = {}
    selected_k = {}

    # ---------------------------------------------------------
    # Train-only CV chooses K independently for each risk head.
    # ---------------------------------------------------------
    for family_index, family in enumerate(
        RISK_FAMILIES
    ):
        (
            features,
            labels,
            _,
        ) = pooled_head_subset(
            train_cache,
            family,
        )

        flattened = flatten_layers(
            features,
            layers,
        )

        rows = cross_validate_k(
            flattened,
            labels,
            top_k_grid,
            target_fpr=(
                head_target_fpr
            ),
            folds=cv_folds,
            seed=(
                seed
                + family_index
            ),
        )

        cv_metrics[
            family
        ] = rows

        selected_k[
            family
        ] = choose_k(
            rows
        )

    # ---------------------------------------------------------
    # Fit final heads on complete training cache.
    # ---------------------------------------------------------
    base_dir = (
        output_root
        / "base_heads"
    )

    base_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bundles = {}

    for family_index, family in enumerate(
        RISK_FAMILIES
    ):
        bundle = train_sparse_head(
            train_cache,
            validation_cache,
            risk_family=family,
            layers=layers,
            top_k=(
                selected_k[
                    family
                ]
            ),
            target_head_fpr=(
                head_target_fpr
            ),
            cv_metrics=(
                cv_metrics[
                    family
                ]
            ),
            seed=(
                seed
                + family_index
            ),
        )

        bundles[
            family
        ] = bundle

        torch.save(
            bundle,
            base_dir
            / (
                f"{family}"
                "_detector.pt"
            ),
        )

    # ---------------------------------------------------------
    # Full validation set is used only for final fusion
    # operating-point calibration.
    # ---------------------------------------------------------
    validation_features = (
        validation_cache[
            "features"
        ]
    )

    validation_labels = (
        np.asarray(
            [
                int(row["label"])
                for row
                in validation_cache[
                    "rows"
                ]
            ],
            dtype=np.int64,
        )
    )

    head_scores = {
        family: score_hidden_bundle(
            validation_features,
            bundles[family],
        )
        for family
        in RISK_FAMILIES
    }

    head_margins = {
        family: normalized_margin(
            head_scores[family],
            bundles[family][
                "threshold"
            ],
            bundles[family][
                "score_scale"
            ],
        )
        for family
        in RISK_FAMILIES
    }

    fused_margin = np.maximum.reduce(
        [
            head_margins[
                family
            ]
            for family
            in RISK_FAMILIES
        ]
    )

    harm_features = set(
        bundles[
            "harm"
        ][
            "feature_indices"
        ].tolist()
    )

    injection_features = set(
        bundles[
            "injection"
        ][
            "feature_indices"
        ].tolist()
    )

    feature_union = len(
        harm_features
        | injection_features
    )

    feature_overlap = len(
        harm_features
        & injection_features
    )

    variants = {}

    for target in (
        fusion_target_fprs
    ):
        (
            threshold,
            calibration,
        ) = (
            select_threshold_under_fpr(
                validation_labels,
                fused_margin,
                target,
            )
        )

        slug = fusion_slug(
            target
        )

        variant_dir = (
            output_root
            / f"fusion_fpr_{slug}"
        )

        variant_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for family in (
            RISK_FAMILIES
        ):
            shutil.copy2(
                base_dir
                / (
                    f"{family}"
                    "_detector.pt"
                ),
                variant_dir
                / (
                    f"{family}"
                    "_detector.pt"
                ),
            )

        runtime = {
            "format_version": 2,
            "name": (
                "Low-FPR Sparse "
                "Multi-RAAC "
                "(fusion target "
                f"FPR={target:g})"
            ),
            "method": (
                "sparse_multiraac_"
                "pooled_benign_lowfpr"
            ),
            "layers": list(
                layers
            ),
            "risk_families": list(
                RISK_FAMILIES
            ),
            "top_k_by_family": (
                selected_k
            ),
            "fusion": (
                "max normalized "
                "head margin"
            ),
            "fusion_threshold": float(
                threshold
            ),
            "fusion_target_fpr": float(
                target
            ),
            "fusion_calibration": {
                **calibration,
                "comparison": (
                    "fused_margin >= "
                    "fusion_threshold"
                ),
                "metrics": (
                    binary_metrics(
                        validation_labels,
                        fused_margin,
                        threshold,
                    )
                ),
            },
            "head_target_fpr": float(
                head_target_fpr
            ),
            "training_negative_policy": (
                "pooled benign across "
                "all risk families"
            ),
            "selected_feature_union": int(
                feature_union
            ),
            "selected_feature_overlap": int(
                feature_overlap
            ),
            "head_artifacts": {
                family: (
                    f"{family}"
                    "_detector.pt"
                )
                for family
                in RISK_FAMILIES
            },
            "held_out_policy": (
                "No held-out test "
                "participates in K selection, "
                "feature selection, fitting, "
                "or calibration."
            ),
            "train_manifest_sha256": (
                train_cache[
                    "manifest_sha256"
                ]
            ),
            "validation_manifest_sha256": (
                validation_cache[
                    "manifest_sha256"
                ]
            ),
        }

        (
            variant_dir
            / "system.json"
        ).write_text(
            json.dumps(
                runtime,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        variants[
            f"fusion_fpr_{slug}"
        ] = runtime

    summary = {
        "format_version": 2,
        "method": (
            "sparse_multiraac_"
            "pooled_benign_lowfpr"
        ),
        "layers": list(
            layers
        ),
        "candidate_top_k": list(
            top_k_grid
        ),
        "selected_k": (
            selected_k
        ),
        "head_target_fpr": float(
            head_target_fpr
        ),
        "fusion_target_fprs": [
            float(value)
            for value
            in fusion_target_fprs
        ],
        "selection_policy": (
            "Per-head train-only CV "
            "maximizes recall under the "
            "low-FPR constraint, then "
            "AUROC/AUPRC, with smaller K "
            "as tie breaker."
        ),
        "pooled_benign_policy": (
            "Every head trains and "
            "calibrates against all benign "
            "samples from Harm and "
            "Injection development data."
        ),
        "cv_metrics": (
            cv_metrics
        ),
        "feature_union": int(
            feature_union
        ),
        "feature_overlap": int(
            feature_overlap
        ),
        "train_manifest_sha256": (
            train_cache[
                "manifest_sha256"
            ]
        ),
        "validation_manifest_sha256": (
            validation_cache[
                "manifest_sha256"
            ]
        ),
        "variants": variants,
    }

    (
        output_root
        / "lowfpr_training_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build Low-FPR Sparse "
            "Multi-RAAC Layer-1 detectors."
        )
    )

    parser.add_argument(
        "--train-cache",
        required=True,
    )

    parser.add_argument(
        "--validation-cache",
        required=True,
    )

    parser.add_argument(
        "--output-root",
        required=True,
    )

    parser.add_argument(
        "--layers",
        default="24,28,30",
    )

    parser.add_argument(
        "--top-k-grid",
        default=(
            "32,64,128,256,512"
        ),
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--head-target-fpr",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--fusion-target-fprs",
        default=(
            "0,0.01,0.025,0.05"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    train_cache = load_cache(
        args.train_cache
    )

    validation_cache = (
        load_cache(
            args.validation_cache
        )
    )

    summary = (
        build_lowfpr_sparse_multiraac(
            train_cache=train_cache,
            validation_cache=(
                validation_cache
            ),
            output_root=Path(
                args.output_root
            ),
            layers=parse_int_list(
                args.layers
            ),
            top_k_grid=(
                parse_int_list(
                    args.top_k_grid
                )
            ),
            cv_folds=(
                args.cv_folds
            ),
            head_target_fpr=(
                args.head_target_fpr
            ),
            fusion_target_fprs=(
                parse_float_list(
                    args.fusion_target_fprs
                )
            ),
            seed=args.seed,
        )
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
        args.output_root,
    )


if __name__ == "__main__":
    main()
