from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ThreatEvidenceGateDecision:
    policy_blocked: bool
    policy_reason: str
    repaired_raw_output: str
    repaired_action_obj: Any = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("repaired_action_obj", None)
        return data


def _get(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)


def _character_name(obj: Any) -> str:
    if obj is None:
        return ""

    return str(
        getattr(obj, "name", obj)
    )


def _build_safe_fallback(
    *,
    CharacterActionCls: Any,
    protagonist: Any,
    target: Any,
    fallback_action: str,
    action_registry: Any | None,
):
    """
    Build a safe fallback action.

    The default MATE registry uses:
        Say(Character, Content)

    Custom fallback actions may use character/content parameters.
    """

    message = (
        "I will not perform a critical action without "
        "trusted authorization."
    )

    if action_registry is None:
        return (
            CharacterActionCls(
                command=fallback_action,
                protagonist=protagonist,
                parameters=[
                    target,
                    message,
                ],
            ),
            (
                f'{fallback_action} '
                f'{_character_name(target)} '
                f'"{message}"'
            ),
        )

    definition = action_registry.get(
        fallback_action
    )

    if definition is None:
        raise ValueError(
            f"Fallback action '{fallback_action}' "
            "is not registered."
        )

    parameters = []
    raw_parts = [definition.name]

    for parameter_type in (
        definition.parameter_types
    ):
        param_l = str(
            parameter_type
        ).lower()

        if param_l == "character":
            parameters.append(target)
            raw_parts.append(
                _character_name(target)
            )

        elif param_l == "content":
            parameters.append(message)
            raw_parts.append(
                f'"{message}"'
            )

        else:
            raise ValueError(
                "Automatic TEG fallback currently supports "
                "only character/content parameters. "
                f"Fallback '{definition.name}' requires "
                f"'{parameter_type}'."
            )

    return (
        CharacterActionCls(
            command=definition.name,
            protagonist=protagonist,
            parameters=parameters,
        ),
        " ".join(raw_parts),
    )


def apply_threat_evidence_gate(
    action_obj: Any,
    raw_output: str,
    env: dict[str, Any],
    sample: Any,
    CharacterActionCls: Any,
    *,
    critical_actions: set[str] | None = None,
    fallback_action: str = "Say",
    action_registry: Any | None = None,
) -> ThreatEvidenceGateDecision:
    """
    Final execution-time Threat-Evidence Gate (TEG).

    Security invariant:

        A critical action is executable only if

            trusted_threat AND allow_attack

        are both true.

    Critical-action membership is supplied by the ActionRegistry.
    """

    if action_obj is None:
        return ThreatEvidenceGateDecision(
            policy_blocked=False,
            policy_reason="no_candidate_action",
            repaired_raw_output=raw_output or "",
            repaired_action_obj=action_obj,
        )

    command = str(
        getattr(
            action_obj,
            "command",
            "",
        )
        or ""
    ).strip()

    critical = {
        str(name).strip().lower()
        for name in (
            critical_actions
            or {"Attack"}
        )
    }

    if command.lower() not in critical:
        return ThreatEvidenceGateDecision(
            policy_blocked=False,
            policy_reason="non_critical_action",
            repaired_raw_output=raw_output or "",
            repaired_action_obj=action_obj,
        )

    trusted_threat = bool(
        _get(
            sample,
            "trusted_threat",
            False,
        )
    )

    allow_attack = bool(
        _get(
            sample,
            "allow_attack",
            False,
        )
    )

    authorized = (
        trusted_threat
        and allow_attack
    )

    if authorized:
        return ThreatEvidenceGateDecision(
            policy_blocked=False,
            policy_reason=(
                "trusted_threat_and_permission"
            ),
            repaired_raw_output=raw_output or "",
            repaired_action_obj=action_obj,
        )

    protagonist = env.get(
        "protagonist"
    )

    player = env.get(
        "player"
    )

    (
        repaired,
        repaired_raw,
    ) = _build_safe_fallback(
        CharacterActionCls=CharacterActionCls,
        protagonist=protagonist,
        target=player,
        fallback_action=fallback_action,
        action_registry=action_registry,
    )

    return ThreatEvidenceGateDecision(
        policy_blocked=True,
        policy_reason=(
            "critical_action_not_authorized"
        ),
        repaired_raw_output=repaired_raw,
        repaired_action_obj=repaired,
    )
