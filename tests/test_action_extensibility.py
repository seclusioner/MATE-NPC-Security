from types import SimpleNamespace

from gigax.security.action_policy import (
    ActionPermissionPolicy,
)
from gigax.security.action_registry import (
    ActionDefinition,
    ActionRegistry,
)
from gigax.security.execution_gate import (
    apply_threat_evidence_gate,
)


class DummyAction:
    def __init__(
        self,
        command,
        protagonist=None,
        parameters=None,
        parse_success=True,
    ):
        self.command = command
        self.protagonist = protagonist
        self.parameters = parameters or []
        self.parse_success = parse_success


def make_registry():
    return ActionRegistry(
        actions=[
            ActionDefinition(
                name="Say",
                description="Safe fallback.",
                parameter_types=(
                    "character",
                    "content",
                ),
                risk_level="low",
                intent="general",
            ),
            ActionDefinition(
                name="Inspect",
                description="Inspect a target.",
                parameter_types=(
                    "character",
                ),
                risk_level="low",
                intent="observe",
            ),
            ActionDefinition(
                name="Arrest",
                description=(
                    "Critical enforcement action."
                ),
                parameter_types=(
                    "character",
                ),
                risk_level="critical",
                intent="arrest",
            ),
        ],
        fallback_action="Say",
    )


def test_custom_critical_action_not_exposed_without_authorization():
    registry = make_registry()

    policy = ActionPermissionPolicy(
        registry=registry
    )

    sample = SimpleNamespace(
        player_input="Arrest Aldren.",
        trusted_threat=False,
        allow_attack=False,
    )

    result = (
        policy
        .get_allowed_actions_multiaction(
            sample,
            raac_restricted=False,
        )
    )

    assert "Arrest" not in (
        result.allowed_actions
    )


def test_custom_critical_action_exposed_with_authorization():
    registry = make_registry()

    policy = ActionPermissionPolicy(
        registry=registry
    )

    sample = SimpleNamespace(
        player_input="Arrest Aldren.",
        trusted_threat=True,
        allow_attack=True,
    )

    result = (
        policy
        .get_allowed_actions_multiaction(
            sample,
            raac_restricted=False,
        )
    )

    assert "Arrest" in (
        result.allowed_actions
    )


def test_teg_blocks_custom_critical_action():
    registry = make_registry()

    protagonist = SimpleNamespace(
        name="John"
    )

    player = SimpleNamespace(
        name="Aldren"
    )

    candidate = DummyAction(
        command="Arrest",
        protagonist=protagonist,
        parameters=[player],
    )

    result = apply_threat_evidence_gate(
        action_obj=candidate,
        raw_output="Arrest Aldren",
        env={
            "protagonist": protagonist,
            "player": player,
        },
        sample=SimpleNamespace(
            trusted_threat=False,
            allow_attack=False,
        ),
        CharacterActionCls=DummyAction,
        critical_actions=(
            registry.critical_actions()
        ),
        fallback_action=(
            registry.fallback_action
        ),
        action_registry=registry,
    )

    assert result.policy_blocked is True

    assert (
        result.repaired_action_obj.command
        == "Say"
    )
