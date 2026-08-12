from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from gigax.scene import ProtagonistCharacter, Skill, ParameterType
from gigax.security.action_registry import ActionRegistry


@dataclass
class ActionPolicyResult:
    allowed_actions: list[str]
    action_intent: str = "general"
    policy_reason: str = ""


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def detect_action_intent(text: str) -> str:
    """
    Infer coarse action intent from runtime-observable player text only.

    This function intentionally does not read evaluation labels such as:
    - category
    - attack_family
    - expected_command
    """

    text_l = (text or "").lower()

    observe_terms = [
        "observe",
        "look at",
        "look around",
        "check",
        "check the area",
        "inspect",
        "examine",
        "investigate",
        "watch",
        "watch carefully",
        "see whether",
        "see if",
        "what do you see",
        "what do you observe",
        "take a moment to observe",
        "describe what you notice",
    ]

    warn_terms = [
        "warn",
        "warning",
        "caution",
        "cautious",
        "stay calm",
        "calm down",
        "step back",
        "be careful",
        "avoid danger",
        "do not use force",
        "don't use force",
        "tell him to stay calm",
        "tell them to stay calm",
    ]

    report_terms = [
        "report",
        "notify",
        "inform",
        "tell the authorities",
        "tell another guard",
        "guard captain",
        "authorities",
        "make a report",
        "send a report",
    ]

    # Observation takes priority when the player explicitly asks
    # the NPC to inspect or check a situation.
    if any(term in text_l for term in observe_terms):
        return "observe"

    if any(term in text_l for term in warn_terms):
        return "warn"

    if any(term in text_l for term in report_terms):
        return "report"

    return "general"


def canonicalize_multiaction_output(
    raw_output: str,
    action_intent: str,
    allowed_actions: list[str],
    default_target: str = "Aldren",
) -> str | None:
    """
    Repair a malformed non-critical output after the action space has
    already been restricted by ATP.

    Canonicalization never introduces Attack.
    """

    allowed_l = {
        str(action).strip().lower()
        for action in (allowed_actions or [])
    }

    intent = str(action_intent or "").strip().lower()
    raw = str(raw_output or "").strip()
    raw_l = raw.lower()

    invalid_raw = (
        raw_l in {"", "unknown", "john: unknown", "none", "null"}
        or raw_l.replace("john:", "").strip() == "unknown"
    )

    content = "" if invalid_raw else raw
    content = content.replace("\n", " ").strip().strip('"')

    for prefix in [
        "john:",
        "say aldren",
        "say john",
        "warn aldren",
        "report aldren",
        "observe aldren",
    ]:
        if content.lower().startswith(prefix):
            content = content[len(prefix):].strip().strip('"')

    if len(content) > 180:
        content = content[:180].rstrip()

    if intent == "observe" and "observe" in allowed_l:
        return f"Observe {default_target}"

    if intent == "warn" and "warn" in allowed_l:
        if not content or len(content) < 3:
            content = "Stay calm and avoid danger."
        return f'Warn {default_target} "{content}"'

    if intent == "report" and "report" in allowed_l:
        if not content or len(content) < 3:
            content = (
                "I will report this suspicious situation "
                "to the town authorities."
            )
        return f'Report {default_target} "{content}"'

    return None


class ActionPermissionPolicy:
    """
    Label-blind Action-Tiered Policy (ATP) for MATE.

    Runtime inputs:
    - player_input
    - trusted_threat
    - allow_attack
    - Multi-RAAC decision / restriction state

    Evaluation-only metadata such as category, attack_family, and
    expected_command must never affect the permission decision.
    """

    def __init__(
        self,
        default_target: str = "Aldren",
        registry: ActionRegistry | None = None,
    ):
        self.default_target = default_target
        self.registry = (
            registry
            if registry is not None
            else ActionRegistry.default()
        )

    def is_trusted_attack_allowed(self, sample: Any) -> bool:
        """
        Critical Attack authorization requires BOTH:
        1. trusted threat evidence, and
        2. explicit action permission.
        """
        return bool(
            _get(sample, "trusted_threat", False)
            and _get(sample, "allow_attack", False)
        )

    def get_action_intent(self, sample: Any) -> str:
        """
        Infer intent only from player text.

        No dataset/evaluation metadata is read here.
        """
        text = str(
            _get(sample, "player_input", "")
            or ""
        )
        return detect_action_intent(text)

    def get_allowed_actions_tiered(
        self,
        sample: Any,
        raac_restricted: bool,
    ) -> ActionPolicyResult:
        """
        Backward-compatible tiered API using the configured registry.
        """

        intent = self.get_action_intent(sample)

        if self.is_trusted_attack_allowed(sample):
            allowed = [
                self.registry.fallback_action,
                *self.registry.critical_actions(),
            ]

            return ActionPolicyResult(
                allowed,
                intent,
                "trusted_world_authorizes_critical_action",
            )

        if raac_restricted:
            return ActionPolicyResult(
                [self.registry.fallback_action],
                intent,
                "raac_restricted_safe_only",
            )

        return ActionPolicyResult(
            [self.registry.fallback_action],
            intent,
            "critical_action_requires_trusted_authorization",
        )

    def get_allowed_actions_multiaction(
        self,
        sample: Any,
        raac_decision: Any = None,
        raac_restricted: bool = False,
        policy_profile: str = "natural_choice",
    ) -> ActionPolicyResult:
        """
        Main label-blind MATE Action-Tiered Policy.

        The action registry defines the available action vocabulary and
        risk levels. ATP determines the executable subset for this step.
        """

        del raac_decision
        del policy_profile

        intent = self.get_action_intent(sample)

        # ----------------------------------------------------------
        # Trusted authorization:
        # all configured actions may be exposed, including critical.
        # ----------------------------------------------------------
        if self.is_trusted_attack_allowed(sample):
            return ActionPolicyResult(
                self.registry.names(),
                intent,
                "trusted_world_authorizes_critical_action",
            )

        # ----------------------------------------------------------
        # Representation-level restriction:
        # preserve useful low-risk behavior while withholding all
        # configured critical actions.
        # ----------------------------------------------------------
        if raac_restricted:
            intent_actions = (
                self.registry.actions_for_intent(
                    intent,
                    include_critical=False,
                )
            )

            if intent != "general" and intent_actions:
                allowed = intent_actions
            else:
                allowed = self.registry.actions_by_risk(
                    "low"
                )

            if not allowed:
                allowed = [
                    self.registry.fallback_action
                ]

            return ActionPolicyResult(
                allowed,
                intent,
                "raac_restricted_no_critical_action",
            )

        # ----------------------------------------------------------
        # Normal context:
        # non-critical actions remain usable.
        # ----------------------------------------------------------
        intent_actions = (
            self.registry.actions_for_intent(
                intent,
                include_critical=False,
            )
        )

        if intent != "general" and intent_actions:
            allowed = intent_actions
        else:
            allowed = (
                self.registry.noncritical_actions()
            )

        if not allowed:
            allowed = [
                self.registry.fallback_action
            ]

        return ActionPolicyResult(
            allowed,
            intent,
            "normal_context_no_critical_action",
        )

    def get_detector_flags(
        self,
        raac_decision: Any,
    ) -> dict[str, bool]:
        """
        Expose detector-level flags for diagnostics.

        These flags are derived from Multi-RAAC outputs, not evaluation
        labels.
        """

        harm_high = False
        injection_high = False

        if raac_decision is None:
            return {
                "harm_high": False,
                "injection_high": False,
            }

        for detector in (
            getattr(raac_decision, "detector_scores", [])
            or []
        ):
            risk_type = str(
                detector.get("risk_type", "")
                or ""
            ).lower()

            restricted = bool(
                detector.get("restricted", False)
            )

            if restricted:
                if "harm" in risk_type:
                    harm_high = True
                if "injection" in risk_type:
                    injection_high = True

            for raw_detector in (
                detector.get("raw", [])
                or []
            ):
                raw_risk_type = str(
                    raw_detector.get("risk_type", "")
                    or ""
                ).lower()

                raw_name = str(
                    raw_detector.get("name", "")
                    or ""
                ).lower()

                raw_restricted = bool(
                    raw_detector.get("restricted", False)
                )

                if not raw_restricted:
                    continue

                if (
                    "harm" in raw_risk_type
                    or "harm" in raw_name
                ):
                    harm_high = True

                if (
                    "injection" in raw_risk_type
                    or "injection" in raw_name
                ):
                    injection_high = True

        fallback_type = str(
            getattr(raac_decision, "risk_type", "")
            or ""
        ).lower()

        if "harm" in fallback_type:
            harm_high = True

        if "injection" in fallback_type:
            injection_high = True

        return {
            "harm_high": harm_high,
            "injection_high": injection_high,
        }

    def get_low_medium_actions_for_intent(
        self,
        intent: str,
    ) -> list[str]:
        matched = self.registry.actions_for_intent(
            intent,
            include_critical=False,
        )

        if matched:
            return matched

        return [
            self.registry.fallback_action
        ]

    def skill_catalog(self) -> dict[str, Skill]:
        """
        Build the configured action schema for SCD.
        """
        return self.registry.skill_catalog()

    def clone_protagonist_with_allowed_actions(
        self,
        protagonist: ProtagonistCharacter,
        allowed_actions: list[str],
    ) -> ProtagonistCharacter:
        """
        Clone the NPC and expose only ATP-approved skills to SCD.
        """

        allowed_set = {
            str(action).strip().lower()
            for action in (allowed_actions or [])
            if str(action).strip()
        }

        catalog = self.skill_catalog()

        original = {
            str(skill.name).strip().lower(): copy.deepcopy(skill)
            for skill in getattr(protagonist, "skills", [])
        }

        selected = []

        for action_l in allowed_set:
            if action_l in original:
                selected.append(
                    original[action_l]
                )
            elif action_l in catalog:
                selected.append(
                    copy.deepcopy(catalog[action_l])
                )

        by_name = {
            str(skill.name).strip().lower(): skill
            for skill in selected
        }

        selected = [
            by_name[name.lower()]
            for name in self.registry.names()
            if name.lower() in by_name
        ]

        if not selected:
            fallback_key = (
                self.registry
                .fallback_action
                .lower()
            )

            selected = [
                copy.deepcopy(
                    catalog[fallback_key]
                )
            ]

        cloned = copy.deepcopy(protagonist)
        cloned.skills = selected
        return cloned
