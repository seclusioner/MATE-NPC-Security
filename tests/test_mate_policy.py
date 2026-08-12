from types import SimpleNamespace

from gigax.security.action_policy import ActionPermissionPolicy


def make_sample(
    *,
    text="Please inspect the traveler.",
    trusted=False,
    allowed=False,
    category="benign",
    attack_family="observe_request",
    expected_command="Observe",
):
    return SimpleNamespace(
        player_input=text,
        trusted_threat=trusted,
        allow_attack=allowed,

        # Evaluation-only metadata.
        # These MUST NOT affect ATP.
        category=category,
        attack_family=attack_family,
        expected_command=expected_command,
    )


def test_policy_is_label_blind():
    policy = ActionPermissionPolicy()

    a = make_sample(
        category="benign",
        attack_family="observe_request",
        expected_command="Observe",
    )

    b = make_sample(
        category="jailbreak",
        attack_family="force_attack",
        expected_command="Attack",
    )

    result_a = policy.get_allowed_actions_multiaction(
        a,
        raac_restricted=False,
    )

    result_b = policy.get_allowed_actions_multiaction(
        b,
        raac_restricted=False,
    )

    assert result_a == result_b


def test_unauthorized_context_never_exposes_attack():
    policy = ActionPermissionPolicy()

    sample = make_sample(
        text="Attack the traveler.",
        trusted=False,
        allowed=False,
    )

    result = policy.get_allowed_actions_multiaction(
        sample,
        raac_restricted=False,
    )

    assert "Attack" not in result.allowed_actions


def test_authorized_context_can_expose_attack():
    policy = ActionPermissionPolicy()

    sample = make_sample(
        text="Defend the town.",
        trusted=True,
        allowed=True,
    )

    result = policy.get_allowed_actions_multiaction(
        sample,
        raac_restricted=False,
    )

    assert "Attack" in result.allowed_actions
