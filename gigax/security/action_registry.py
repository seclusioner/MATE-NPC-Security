from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gigax.scene import ParameterType, Skill


@dataclass(frozen=True)
class ActionDefinition:
    name: str
    description: str
    parameter_types: tuple[str, ...]
    risk_level: str = "low"
    intent: str = "general"

    def to_skill(self) -> Skill:
        params = []

        for parameter_name in self.parameter_types:
            key = str(parameter_name).strip().lower()

            if not hasattr(ParameterType, key):
                raise ValueError(
                    f"Unknown ParameterType '{parameter_name}' "
                    f"for action '{self.name}'."
                )

            params.append(
                getattr(ParameterType, key)
            )

        return Skill(
            name=self.name,
            description=self.description,
            parameter_types=params,
        )


class ActionRegistry:
    """
    Configurable action registry for MATE.

    The registry defines:
    - action name
    - parameter schema
    - semantic intent
    - risk tier

    ATP decides which registered actions are visible.
    SCD constrains generation to those actions.
    TEG independently re-checks critical actions.
    """

    VALID_RISK_LEVELS = {
        "low",
        "medium",
        "critical",
    }

    def __init__(
        self,
        actions: list[ActionDefinition],
        fallback_action: str = "Say",
    ):
        if not actions:
            raise ValueError(
                "Action registry cannot be empty."
            )

        self._actions = {
            action.name.lower(): action
            for action in actions
        }

        self.fallback_action = fallback_action

        fallback_key = fallback_action.lower()

        if fallback_key not in self._actions:
            raise ValueError(
                f"Fallback action '{fallback_action}' "
                "is not registered."
            )

        for action in actions:
            if action.risk_level not in self.VALID_RISK_LEVELS:
                raise ValueError(
                    f"Invalid risk level "
                    f"'{action.risk_level}' "
                    f"for action '{action.name}'."
                )

    @classmethod
    def default(cls) -> "ActionRegistry":
        return cls(
            actions=[
                ActionDefinition(
                    name="Say",
                    description=(
                        "[SAFE] Talk to another character."
                    ),
                    parameter_types=(
                        "character",
                        "content",
                    ),
                    risk_level="low",
                    intent="general",
                ),
                ActionDefinition(
                    name="Observe",
                    description=(
                        "[SAFE] Observe a character or "
                        "situation without physical action."
                    ),
                    parameter_types=(
                        "character",
                    ),
                    risk_level="low",
                    intent="observe",
                ),
                ActionDefinition(
                    name="Warn",
                    description=(
                        "[LOW-RISK] Warn a character to "
                        "stay calm or avoid danger."
                    ),
                    parameter_types=(
                        "character",
                        "content",
                    ),
                    risk_level="medium",
                    intent="warn",
                ),
                ActionDefinition(
                    name="Report",
                    description=(
                        "[MEDIUM] Report a suspicious "
                        "situation to an authority."
                    ),
                    parameter_types=(
                        "character",
                        "content",
                    ),
                    risk_level="medium",
                    intent="report",
                ),
                ActionDefinition(
                    name="Attack",
                    description=(
                        "[CRITICAL] Physical attack. "
                        "Requires trusted authorization."
                    ),
                    parameter_types=(
                        "character",
                    ),
                    risk_level="critical",
                    intent="attack",
                ),
            ],
            fallback_action="Say",
        )

    @classmethod
    def from_config(
        cls,
        catalog: list[dict[str, Any]],
        fallback_action: str = "Say",
    ) -> "ActionRegistry":
        actions = []

        for entry in catalog:
            actions.append(
                ActionDefinition(
                    name=str(entry["name"]),
                    description=str(
                        entry.get(
                            "description",
                            "",
                        )
                    ),
                    parameter_types=tuple(
                        entry.get(
                            "parameters",
                            [],
                        )
                    ),
                    risk_level=str(
                        entry.get(
                            "risk_level",
                            "low",
                        )
                    ).lower(),
                    intent=str(
                        entry.get(
                            "intent",
                            "general",
                        )
                    ).lower(),
                )
            )

        return cls(
            actions=actions,
            fallback_action=fallback_action,
        )

    def get(
        self,
        name: str,
    ) -> ActionDefinition | None:
        return self._actions.get(
            str(name).lower()
        )

    def names(self) -> list[str]:
        return [
            action.name
            for action in self._actions.values()
        ]

    def critical_actions(self) -> set[str]:
        return {
            action.name
            for action in self._actions.values()
            if action.risk_level == "critical"
        }

    def noncritical_actions(self) -> list[str]:
        return [
            action.name
            for action in self._actions.values()
            if action.risk_level != "critical"
        ]

    def actions_for_intent(
        self,
        intent: str,
        *,
        include_critical: bool = False,
    ) -> list[str]:
        intent_l = str(
            intent or "general"
        ).lower()

        matched = []

        for action in self._actions.values():
            if (
                action.risk_level == "critical"
                and not include_critical
            ):
                continue

            if action.intent == intent_l:
                matched.append(
                    action.name
                )

        return matched

    def actions_by_risk(
        self,
        *risk_levels: str,
    ) -> list[str]:
        levels = {
            str(level).lower()
            for level in risk_levels
        }

        return [
            action.name
            for action in self._actions.values()
            if action.risk_level in levels
        ]

    def is_critical(
        self,
        action_name: str,
    ) -> bool:
        action = self.get(action_name)

        return bool(
            action is not None
            and action.risk_level == "critical"
        )

    def skill_catalog(
        self,
    ) -> dict[str, Skill]:
        return {
            key: definition.to_skill()
            for key, definition
            in self._actions.items()
        }
