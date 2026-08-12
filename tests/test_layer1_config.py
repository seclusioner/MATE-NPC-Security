from gigax.config import (
    load_mate_config,
)


def test_original_layer1_config():
    config = load_mate_config(
        "configs/phi3_original.yaml"
    )

    assert (
        config.layer1.type
        == "original_multiraac"
    )

    assert len(
        config.layer1.detectors
    ) == 2

    assert (
        config.layer1.system_path
        == ""
    )


def test_sparse_layer1_config():
    config = load_mate_config(
        "configs/phi3_lowfpr_sparse.yaml"
    )

    assert (
        config.layer1.type
        == "lowfpr_sparse"
    )

    assert len(
        config.layer1.detectors
    ) == 2

    assert (
        "system.json"
        in config.layer1.system_path
    )


