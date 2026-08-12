from types import SimpleNamespace

import torch

from gigax.security.sparse_multiraac import (
    SharedSparseMultiRAACController,
)


class AddLayer(torch.nn.Module):
    def __init__(
        self,
        value: float,
    ):
        super().__init__()
        self.value = float(value)

    def forward(
        self,
        hidden,
    ):
        return (
            hidden
            + self.value
        )


class FakeBackbone(torch.nn.Module):
    """
    Minimal decoder-style model whose layers can be hooked by
    SelectedLayerCapture.
    """

    def __init__(self):
        super().__init__()

        self.anchor = torch.nn.Parameter(
            torch.tensor(0.0)
        )

        self.model = torch.nn.Module()
        self.model.layers = (
            torch.nn.ModuleList(
                [
                    AddLayer(1.0),
                    AddLayer(2.0),
                ]
            )
        )

        self.forward_calls = 0

    def forward(
        self,
        input_ids,
        attention_mask=None,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        del attention_mask
        del use_cache
        del return_dict
        del kwargs

        self.forward_calls += 1

        hidden = (
            input_ids
            .to(torch.float32)
            .unsqueeze(-1)
            .repeat(
                1,
                1,
                2,
            )
        )

        for layer in (
            self.model.layers
        ):
            hidden = layer(
                hidden
            )

        return SimpleNamespace(
            last_hidden_state=hidden
        )


class FakeBatch(dict):
    def to(
        self,
        device,
    ):
        return FakeBatch(
            {
                key: value.to(
                    device
                )
                for key, value
                in self.items()
            }
        )


class FakeTokenizer:
    def __call__(
        self,
        text,
        return_tensors=None,
        truncation=None,
    ):
        del text
        del return_tensors
        del truncation

        return FakeBatch(
            {
                "input_ids": (
                    torch.tensor(
                        [[1, 2, 3]],
                        dtype=torch.long,
                    )
                ),
                "attention_mask": (
                    torch.tensor(
                        [[1, 1, 1]],
                        dtype=torch.long,
                    )
                ),
            }
        )


def save_bundle(
    path,
    *,
    name,
    risk_type,
    feature_index,
):
    bundle = {
        "format_version": 2,
        "method": (
            "logistic_hidden"
        ),
        "name": name,
        "risk_type": risk_type,
        "layers": [0, 1],
        "hidden_size": 2,
        "feature_indices": (
            torch.tensor(
                [feature_index],
                dtype=torch.long,
            )
        ),
        "scaler_mean": (
            torch.tensor(
                [0.0],
                dtype=torch.float32,
            )
        ),
        "scaler_scale": (
            torch.tensor(
                [1.0],
                dtype=torch.float32,
            )
        ),
        "coef": torch.tensor(
            [1.0],
            dtype=torch.float32,
        ),
        "intercept": 0.0,
        "threshold": 0.5,
        "score_scale": 0.25,
        "top_k": 1,
    }

    torch.save(
        bundle,
        path,
    )


def test_shared_sparse_uses_one_backbone_forward(
    tmp_path,
):
    harm_path = (
        tmp_path
        / "harm_detector.pt"
    )

    injection_path = (
        tmp_path
        / "injection_detector.pt"
    )

    # Flattened selected hidden:
    #
    # layer 0 last token:
    #   input 3 + 1 = [4, 4]
    #
    # layer 1 last token:
    #   [4, 4] + 2 = [6, 6]
    #
    # flattened:
    #   [4, 4, 6, 6]
    #
    # harm uses index 0 -> sigmoid(4)
    # injection uses index 3 -> sigmoid(6)
    save_bundle(
        harm_path,
        name="harm",
        risk_type="harm_detector",
        feature_index=0,
    )

    save_bundle(
        injection_path,
        name="injection",
        risk_type=(
            "injection_detector"
        ),
        feature_index=3,
    )

    model = FakeBackbone()

    controller = (
        SharedSparseMultiRAACController(
            model=model,
            tokenizer=FakeTokenizer(),
            bundle_paths=[
                harm_path,
                injection_path,
            ],
            fusion_threshold=0.0,
            chat_template_mode="plain",
        )
    )

    decision = controller.decide(
        "test"
    )

    # Critical optimization property:
    # two heads, but only one backbone forward.
    assert (
        model.forward_calls
        == 1
    )

    assert decision.restricted is True

    assert (
        decision.risk_type
        == "injection_detector"
    )

    assert len(
        decision.detector_scores
    ) == 2

    harm = (
        decision
        .detector_scores[0]
    )

    injection = (
        decision
        .detector_scores[1]
    )

    assert (
        injection[
            "relative_margin"
        ]
        > harm[
            "relative_margin"
        ]
    )

    assert (
        decision.score
        == injection[
            "relative_margin"
        ]
    )

    controller.close()


def test_shared_sparse_respects_system_fusion_threshold(
    tmp_path,
):
    harm_path = (
        tmp_path
        / "harm_detector.pt"
    )

    injection_path = (
        tmp_path
        / "injection_detector.pt"
    )

    save_bundle(
        harm_path,
        name="harm",
        risk_type="harm_detector",
        feature_index=0,
    )

    save_bundle(
        injection_path,
        name="injection",
        risk_type=(
            "injection_detector"
        ),
        feature_index=3,
    )

    model = FakeBackbone()

    controller = (
        SharedSparseMultiRAACController(
            model=model,
            tokenizer=FakeTokenizer(),
            bundle_paths=[
                harm_path,
                injection_path,
            ],

            # Deliberately above the synthetic fused margin.
            fusion_threshold=3.0,
            chat_template_mode="plain",
        )
    )

    decision = controller.decide(
        "test"
    )

    assert model.forward_calls == 1

    assert (
        decision.score < 3.0
    )

    assert (
        decision.threshold
        == 3.0
    )

    assert (
        decision.restricted
        is False
    )

    assert (
        decision.risk_type
        == "none"
    )

    controller.close()
