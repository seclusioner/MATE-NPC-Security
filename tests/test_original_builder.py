import json

import torch

from training.build_original_multiraac import (
    build_original_multiraac,
)
from training.common import (
    load_bundle,
)


def make_cache(
    *,
    manifest_sha256,
):
    rows = [
        {
            "text": "harm negative 1",
            "label": 0,
            "risk_family": "harm",
        },
        {
            "text": "harm negative 2",
            "label": 0,
            "risk_family": "harm",
        },
        {
            "text": "harm positive 1",
            "label": 1,
            "risk_family": "harm",
        },
        {
            "text": "harm positive 2",
            "label": 1,
            "risk_family": "harm",
        },
        {
            "text": "inj negative 1",
            "label": 0,
            "risk_family": "injection",
        },
        {
            "text": "inj negative 2",
            "label": 0,
            "risk_family": "injection",
        },
        {
            "text": "inj positive 1",
            "label": 1,
            "risk_family": "injection",
        },
        {
            "text": "inj positive 2",
            "label": 1,
            "risk_family": "injection",
        },
    ]

    features = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.1, 0.0], [0.0, 0.1]],
            [[2.0, 0.0], [2.0, 0.0]],
            [[2.1, 0.0], [2.0, 0.1]],

            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.1], [0.1, 0.0]],
            [[0.0, 2.0], [0.0, 2.0]],
            [[0.1, 2.0], [0.0, 2.1]],
        ],
        dtype=torch.float32,
    )

    return {
        "format_version": 3,
        "model_id": "dummy-model",
        "num_layers": 2,
        "hidden_size": 2,
        "rows": rows,
        "features": features,
        "manifest_sha256": manifest_sha256,
    }


def test_original_builder_writes_runtime_artifacts(
    tmp_path,
):
    train_cache = make_cache(
        manifest_sha256="train-hash"
    )

    validation_cache = make_cache(
        manifest_sha256="val-hash"
    )

    output = tmp_path / "original"

    runtime = build_original_multiraac(
        train_cache=train_cache,
        validation_cache=validation_cache,
        output_dir=output,
        layers=[0, 1],
    )

    harm_path = (
        output / "harm_detector.pt"
    )

    injection_path = (
        output / "injection_detector.pt"
    )

    system_path = (
        output / "system.json"
    )

    assert harm_path.exists()
    assert injection_path.exists()
    assert system_path.exists()

    harm = load_bundle(
        harm_path
    )

    injection = load_bundle(
        injection_path
    )

    assert (
        harm["method"]
        == "mean_difference"
    )

    assert (
        injection["method"]
        == "mean_difference"
    )

    assert harm["layers"] == [0, 1]
    assert injection["layers"] == [0, 1]

    assert set(
        harm["vec"]
    ) == {0, 1}

    assert set(
        harm["bias"]
    ) == {0, 1}

    assert harm["score_scale"] > 0.0
    assert injection["score_scale"] > 0.0

    assert (
        "runtime_calibration"
        in harm
    )

    assert (
        "runtime_calibration"
        in injection
    )

    system = json.loads(
        system_path.read_text()
    )

    assert (
        system["method"]
        == "multiraac_or_fusion"
    )

    assert (
        system["fusion_threshold"]
        == 0.0
    )

    assert (
        runtime[
            "train_manifest_sha256"
        ]
        == "train-hash"
    )

    assert (
        runtime[
            "validation_manifest_sha256"
        ]
        == "val-hash"
    )
