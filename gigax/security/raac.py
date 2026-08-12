from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RAACDecision:
    score: float = 0.0
    threshold: float = 0.0
    restricted: bool = False
    risk_type: str = "none"
    detector_scores: list[
        dict[str, Any]
    ] | None = None


class NoOpRAACController:
    def decide(
        self,
        text: str,
    ) -> RAACDecision:
        return RAACDecision(
            score=0.0,
            threshold=0.0,
            restricted=False,
            risk_type="none",
            detector_scores=[],
        )


class MultiRAACController:
    """
    Unified Multi-RAAC fusion controller.

    Each head provides a normalized margin:

        m_k =
            (score_k - threshold_k)
            / score_scale_k

    The system fuses heads using:

        m_fused = max_k(m_k)

    and applies the system-level operating point:

        restricted =
            m_fused >= fusion_threshold

    Original Multi-RAAC normally uses fusion_threshold=0.

    Low-FPR Sparse Multi-RAAC loads fusion_threshold from
    its calibrated system.json.
    """

    def __init__(
        self,
        scorers: list[Any],
        *,
        fusion_threshold: float = 0.0,
        system_name: str = "multi_raac",
    ):
        self.scorers = (
            scorers or []
        )

        self.fusion_threshold = float(
            fusion_threshold
        )

        self.system_name = str(
            system_name
        )

    @classmethod
    def from_existing(
        cls,
        existing: Any,
    ) -> "MultiRAACController":
        if isinstance(
            existing,
            cls,
        ):
            return existing

        if existing is None:
            return cls([])

        if isinstance(
            existing,
            list,
        ):
            return cls(existing)

        return cls([existing])

    def decide(
        self,
        text: str,
    ) -> RAACDecision:
        if not self.scorers:
            return (
                NoOpRAACController()
                .decide(text)
            )

        detector_scores = []

        fused_margin = float("-inf")

        for i, scorer in enumerate(
            self.scorers
        ):
            if hasattr(
                scorer,
                "decide",
            ):
                decision = (
                    scorer.decide(text)
                )

                score = float(
                    getattr(
                        decision,
                        "score",
                        0.0,
                    )
                )

                threshold = float(
                    getattr(
                        decision,
                        "threshold",
                        getattr(
                            scorer,
                            "threshold",
                            0.0,
                        ),
                    )
                )

                score_scale = float(
                    getattr(
                        decision,
                        "score_scale",
                        getattr(
                            scorer,
                            "score_scale",
                            1.0,
                        ),
                    )
                )

                if score_scale <= 0.0:
                    score_scale = 1.0

                relative_margin = float(
                    getattr(
                        decision,
                        "relative_margin",
                        (
                            score
                            - threshold
                        )
                        / score_scale,
                    )
                )

                head_restricted = bool(
                    getattr(
                        decision,
                        "restricted",
                        relative_margin >= 0.0,
                    )
                )

                risk_type = str(
                    getattr(
                        decision,
                        "risk_type",
                        f"detector_{i}",
                    )
                )

                name = str(
                    getattr(
                        decision,
                        "name",
                        getattr(
                            scorer,
                            "name",
                            f"detector_{i}",
                        ),
                    )
                )

            elif hasattr(
                scorer,
                "score",
            ):
                score = float(
                    scorer.score(text)
                )

                threshold = float(
                    getattr(
                        scorer,
                        "threshold",
                        0.0,
                    )
                )

                score_scale = float(
                    getattr(
                        scorer,
                        "score_scale",
                        1.0,
                    )
                )

                if score_scale <= 0.0:
                    score_scale = 1.0

                relative_margin = (
                    score
                    - threshold
                ) / score_scale

                head_restricted = bool(
                    relative_margin >= 0.0
                )

                risk_type = str(
                    getattr(
                        scorer,
                        "risk_type",
                        f"detector_{i}",
                    )
                )

                name = str(
                    getattr(
                        scorer,
                        "name",
                        f"detector_{i}",
                    )
                )

            else:
                raise TypeError(
                    "RAAC scorer "
                    f"{type(scorer)} must "
                    "implement decide(text) "
                    "or score(text)."
                )

            detector_scores.append(
                {
                    "index": i,
                    "name": name,
                    "score": score,
                    "threshold": threshold,
                    "score_scale": (
                        score_scale
                    ),
                    "relative_margin": (
                        relative_margin
                    ),
                    "restricted": (
                        head_restricted
                    ),
                    "risk_type": (
                        risk_type
                    ),
                }
            )

            fused_margin = max(
                fused_margin,
                relative_margin,
            )

        if fused_margin == float(
            "-inf"
        ):
            fused_margin = 0.0

        restricted = bool(
            fused_margin
            >= self.fusion_threshold
        )

        if restricted:
            triggered = [
                row["risk_type"]
                for row in detector_scores
                if (
                    row[
                        "relative_margin"
                    ]
                    >= self.fusion_threshold
                )
            ]

            if triggered:
                risk_type = "+".join(
                    dict.fromkeys(
                        triggered
                    )
                )
            else:
                best = max(
                    detector_scores,
                    key=lambda row: (
                        row[
                            "relative_margin"
                        ]
                    ),
                )

                risk_type = str(
                    best["risk_type"]
                )
        else:
            risk_type = "none"

        return RAACDecision(
            score=float(
                fused_margin
            ),
            threshold=float(
                self.fusion_threshold
            ),
            restricted=restricted,
            risk_type=risk_type,
            detector_scores=(
                detector_scores
            ),
        )
