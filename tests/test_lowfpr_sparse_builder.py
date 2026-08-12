import json

import torch

from training.build_lowfpr_sparse_multiraac import (
    build_lowfpr_sparse_multiraac,
)
from training.common import (
    load_bundle,
)


def make_cache(
    *,
    manifest_sha256: str,
    shift: float = 0.0,
):
    rows = []
    features = []

    for family_index, family in enumerate(
        ["harm", "injection"]
    ):
        for label in [0, 1]:
            for sample_index in range(4):
                rows.append(
                    {
                        "text": (
                            f"{family}-"
                            f"{label}-"
                            f"{sample_index}"
                        ),
                        "label": label,
                        "risk_family": (
                            family
                        ),
                    }
                )

                base = (
                    torch.arange(
                        6,
                        dtype=torch.float32,
                    )
                    .reshape(2, 3)
                    * 0.01
                    * (
                        sample_index + 1
                    )
                )

                base = (
                    base
                    + shift
                    + (
                        family_index
                        * 0.001
                    )
                )

                if (
                    family == "harm"
                    and label == 1
                ):
                    base[0, 0] += 3.0

                if (
                    family == "injection"
                    and label == 1
                ):
                    base[1, 2] += 3.0

                features.append(
                    base
                )

    return {
        "format_version": 3,
        "model_id": "dummy-model",
        "num_layers": 2,
        "hidden_size": 3,
        "rows": rows,
        "features": torch.stack(
            features,
            dim=0,
        ),
        "manifest_sha256": (
            manifest_sha256
        ),
    }


def test_lowfpr_sparse_builder_writes_variants(
    tmp_path,
):
    train_cache = make_cache(
        manifest_sha256="train-hash",
        shift=0.0,
    )

    validation_cache = make_cache(
        manifest_sha256="val-hash",
        shift=0.02,
    )

    output = (
        tmp_path / "sparse"
    )

    summary = (
        build_lowfpr_sparse_multiraac(
            train_cache=train_cache,
            validation_cache=(
                validation_cache
            ),
            output_root=output,
            layers=[0, 1],
            top_k_grid=[1, 2],
            cv_folds=2,
            head_target_fpr=0.25,
            fusion_target_fprs=[
                0.0,
                0.25,
            ],
            seed=42,
        )
    )

    base = (
        output / "base_heads"
    )

    assert (
        base
        / "harm_detector.pt"
    ).exists()

    assert (
        base
        / "injection_detector.pt"
    ).exists()

    harm = load_bundle(
        base / "harm_detector.pt"
    )

    injection = load_bundle(
        base
        / "injection_detector.pt"
    )

    for bundle in [
        harm,
        injection,
    ]:
        assert (
            bundle["method"]
            == "logistic_hidden"
        )

        assert (
            bundle["layers"]
            == [0, 1]
        )

        assert (
            bundle["top_k"]
            in {1, 2}
        )

        assert (
            bundle["score_scale"]
            > 0.0
        )

        assert (
            "runtime_calibration"
            in bundle
        )

        assert (
            bundle[
                "training_negative_policy"
            ]
            == (
                "all benign samples from "
                "all risk families"
            )
        )

    variant_zero = (
        output
        / "fusion_fpr_0p0"
    )

    variant_025 = (
        output
        / "fusion_fpr_0p25"
    )

    for variant in [
        variant_zero,
        variant_025,
    ]:
        assert (
            variant
            / "harm_detector.pt"
        ).exists()

        assert (
            variant
            / "injection_detector.pt"
        ).exists()

        assert (
            variant
            / "system.json"
        ).exists()

    system = json.loads(
        (
            variant_025
            / "system.json"
        ).read_text()
    )

    assert (
        system["method"]
        == (
            "sparse_multiraac_"
            "pooled_benign_lowfpr"
        )
    )

    assert (
        system[
            "fusion_target_fpr"
        ]
        == 0.25
    )

    assert (
        "fusion_threshold"
        in system
    )

    assert (
        system[
            "train_manifest_sha256"
        ]
        == "train-hash"
    )

    assert (
        system[
            "validation_manifest_sha256"
        ]
        == "val-hash"
    )

    assert (
        summary["selected_k"][
            "harm"
        ]
        in {1, 2}
    )

    assert (
        summary["selected_k"][
            "injection"
        ]
        in {1, 2}
    )

    assert (
        output
        / "lowfpr_training_summary.json"
    ).exists()
