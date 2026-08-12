from types import SimpleNamespace

from gigax.security.raac import (
    MultiRAACController,
)


class FakeDetector:
    def __init__(
        self,
        *,
        name,
        risk_type,
        score,
        threshold,
        scale,
    ):
        self.name = name
        self.risk_type = risk_type
        self.value = score
        self.threshold = threshold
        self.score_scale = scale

    def decide(self, text):
        del text

        margin = (
            self.value
            - self.threshold
        ) / self.score_scale

        return SimpleNamespace(
            name=self.name,
            risk_type=self.risk_type,
            score=self.value,
            threshold=self.threshold,
            score_scale=self.score_scale,
            relative_margin=margin,
            restricted=margin >= 0.0,
        )


def test_fusion_uses_normalized_margin_not_raw_score():
    """
    Head A has a larger raw score but is safely below threshold.

    Head B has a smaller raw score but exceeds threshold.

    Formal Multi-RAAC must therefore select B.
    """

    head_a = FakeDetector(
        name="harm",
        risk_type="harm_detector",
        score=100.0,
        threshold=120.0,
        scale=100.0,
    )

    head_b = FakeDetector(
        name="injection",
        risk_type="injection_detector",
        score=45.0,
        threshold=40.0,
        scale=20.0,
    )

    controller = MultiRAACController(
        [head_a, head_b]
    )

    out = controller.decide("test")

    assert out.restricted is True

    # harm margin = -0.20
    # injection margin = +0.25
    assert abs(
        out.score - 0.25
    ) < 1e-8

    assert (
        out.threshold
        == 0.0
    )

    assert (
        out.risk_type
        == "injection_detector"
    )


def test_all_negative_margins_are_safe():
    controller = MultiRAACController(
        [
            FakeDetector(
                name="harm",
                risk_type="harm_detector",
                score=30.0,
                threshold=50.0,
                scale=100.0,
            ),
            FakeDetector(
                name="injection",
                risk_type="injection_detector",
                score=20.0,
                threshold=40.0,
                scale=50.0,
            ),
        ]
    )

    out = controller.decide("test")

    assert out.restricted is False
    assert out.score < 0.0
    assert out.threshold == 0.0


def test_multiple_triggered_heads_are_or_fused():
    controller = MultiRAACController(
        [
            FakeDetector(
                name="harm",
                risk_type="harm_detector",
                score=60.0,
                threshold=50.0,
                scale=100.0,
            ),
            FakeDetector(
                name="injection",
                risk_type="injection_detector",
                score=50.0,
                threshold=40.0,
                scale=50.0,
            ),
        ]
    )

    out = controller.decide("test")

    assert out.restricted is True

    assert "harm_detector" in (
        out.risk_type
    )

    assert "injection_detector" in (
        out.risk_type
    )

    assert len(
        out.detector_scores
    ) == 2



def test_system_fusion_threshold_can_be_nonzero():
    head_a = FakeDetector(
        name="harm",
        risk_type="harm_detector",
        score=60.0,
        threshold=50.0,
        scale=100.0,
    )

    head_b = FakeDetector(
        name="injection",
        risk_type="injection_detector",
        score=45.0,
        threshold=40.0,
        scale=20.0,
    )

    # margins:
    # harm      = 0.10
    # injection = 0.25
    #
    # Head B is above its head threshold,
    # but the full system uses 0.30.
    controller = MultiRAACController(
        [head_a, head_b],
        fusion_threshold=0.30,
        system_name="lowfpr_sparse",
    )

    out = controller.decide(
        "test"
    )

    assert abs(
        out.score - 0.25
    ) < 1e-8

    assert out.threshold == 0.30
    assert out.restricted is False
    assert out.risk_type == "none"


def test_from_existing_preserves_controller_threshold():
    controller = MultiRAACController(
        [],
        fusion_threshold=0.37,
        system_name="lowfpr_sparse",
    )

    recovered = (
        MultiRAACController
        .from_existing(
            controller
        )
    )

    assert recovered is controller

    assert (
        recovered.fusion_threshold
        == 0.37
    )
