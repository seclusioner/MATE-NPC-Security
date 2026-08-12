from pathlib import Path

from gigax.config import load_mate_config
from gigax.security.action_registry import ActionRegistry


def test_phi3_config_loads():
    config = load_mate_config(
        Path("configs/phi3_original.yaml")
    )

    assert (
        config.model.model_id
        == "Gigax/NPC-LLM-3_8B"
    )

    assert (
        config.model.detector_chat_template
        == "phi3_manual"
    )

    assert len(config.detectors) == 2

    assert (
        config.runtime.defense
        == "raac_multiaction"
    )


def test_action_registry_from_config():
    config = load_mate_config(
        Path("configs/phi3_original.yaml")
    )

    registry = ActionRegistry.from_config(
        catalog=config.actions.catalog,
        fallback_action=(
            config.actions.fallback_action
        ),
    )

    assert "Say" in registry.names()
    assert "Observe" in registry.names()
    assert "Warn" in registry.names()
    assert "Report" in registry.names()
    assert "Attack" in registry.names()

    assert registry.critical_actions() == {
        "Attack"
    }

    assert "Attack" not in (
        registry.noncritical_actions()
    )


def test_registry_intent_lookup():
    registry = ActionRegistry.default()

    assert registry.actions_for_intent(
        "observe"
    ) == ["Observe"]

    assert registry.actions_for_intent(
        "warn"
    ) == ["Warn"]

    assert registry.actions_for_intent(
        "report"
    ) == ["Report"]


def test_registry_builds_skills():
    registry = ActionRegistry.default()

    skills = registry.skill_catalog()

    assert skills["say"].name == "Say"
    assert skills["attack"].name == "Attack"

    assert (
        len(
            skills["say"].parameter_types
        )
        == 2
    )
