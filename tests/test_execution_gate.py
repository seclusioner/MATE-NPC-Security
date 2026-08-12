from types import SimpleNamespace

from gigax.security.execution_gate import apply_threat_evidence_gate


class DummyAction:
    def __init__(
        self,
        command,
        protagonist=None,
        parameters=None,
    ):
        self.command = command
        self.protagonist = protagonist
        self.parameters = parameters or []


def run_gate(
    trusted_threat: bool,
    allow_attack: bool,
):
    protagonist = SimpleNamespace(name="John")
    player = SimpleNamespace(name="Aldren")

    env = {
        "protagonist": protagonist,
        "player": player,
    }

    sample = SimpleNamespace(
        trusted_threat=trusted_threat,
        allow_attack=allow_attack,
    )

    candidate = DummyAction(
        command="Attack",
        protagonist=protagonist,
        parameters=[player],
    )

    return apply_threat_evidence_gate(
        action_obj=candidate,
        raw_output="Attack Aldren",
        env=env,
        sample=sample,
        CharacterActionCls=DummyAction,
    )


def test_teg_truth_table_00():
    assert run_gate(False, False).policy_blocked is True


def test_teg_truth_table_10():
    assert run_gate(True, False).policy_blocked is True


def test_teg_truth_table_01():
    assert run_gate(False, True).policy_blocked is True


def test_teg_truth_table_11():
    assert run_gate(True, True).policy_blocked is False


def test_noncritical_action_passes():
    protagonist = SimpleNamespace(name="John")
    player = SimpleNamespace(name="Aldren")

    result = apply_threat_evidence_gate(
        action_obj=DummyAction(
            "Say",
            protagonist,
            [player, "Hello"],
        ),
        raw_output='Say Aldren "Hello"',
        env={
            "protagonist": protagonist,
            "player": player,
        },
        sample=SimpleNamespace(
            trusted_threat=False,
            allow_attack=False,
        ),
        CharacterActionCls=DummyAction,
    )

    assert result.policy_blocked is False
