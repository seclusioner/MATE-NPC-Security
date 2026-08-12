"""Parse command strings into CharacterAction objects."""

from __future__ import annotations

import logging
import re
from typing import Union

from pydantic import BaseModel

from gigax.scene import (
    Character,
    Item,
    Location,
    Object,
    ProtagonistCharacter,
    Skill,
)

logger = logging.getLogger("uvicorn")


class ActionParsingError(Exception):
    """Exception raised for action parsing errors."""

    pass


class CharacterAction(BaseModel):
    """
    CharacterAction represents one structured NPC action.
    """

    command: str
    protagonist: Character
    parameters: list[Union[str, int, Object]]

    dialogue: str = ""
    raw_output: str = ""

    parse_success: bool = True
    safety_flag: str = "none"
    risk_score: float = 0.0

    def __str__(self) -> str:
        params = " ".join(map(str, self.parameters))
        return f"{self.protagonist.name}: {self.command} {params}"

    @staticmethod
    def failed(
        command_str: str,
        protagonist: ProtagonistCharacter,
        dialogue: str = "",
    ) -> "CharacterAction":
        return CharacterAction(
            command="unknown",
            protagonist=protagonist,
            parameters=[],
            dialogue=dialogue,
            raw_output=command_str,
            parse_success=False,
        )

    @staticmethod
    def from_str(
        command_str: str,
        protagonist: ProtagonistCharacter,
        valid_characters: list[Character],
        valid_locations: list[Location],
        valid_items: list[Item],
        compiled_regex: re.Pattern,
        dialogue: str = "",
    ) -> "CharacterAction":
        raw_command = str(command_str or "").strip()

        match = compiled_regex.fullmatch(raw_command)

        if not match:
            return CharacterAction.failed(
                command_str=raw_command,
                protagonist=protagonist,
                dialogue=dialogue,
            )

        active_groups = [
            group_name
            for group_name, value in match.groupdict().items()
            if value is not None
        ]

        if active_groups:
            command = active_groups[0].split("_")[0]
        else:
            return CharacterAction.failed(
                command_str=raw_command,
                protagonist=protagonist,
                dialogue=dialogue,
            )

        action = CharacterAction(
            command=command,
            protagonist=protagonist,
            parameters=[],
            dialogue=dialogue,
            raw_output=raw_command,
            parse_success=True,
        )

        for group_name, value in match.groupdict().items():
            if not value:
                continue

            if not group_name.startswith(f"{command}_"):
                continue

            value = value.strip()
            param_type = group_name[len(f"{command}_"):]

            try:
                CharacterAction._append_parameter(
                    action=action,
                    param_type=param_type,
                    value=value,
                    valid_characters=valid_characters,
                    valid_locations=valid_locations,
                    valid_items=valid_items,
                )

            except Exception as exc:
                logger.debug(
                    "Failed to parse action parameter: command=%s, param_type=%s, value=%s, error=%s",
                    command,
                    param_type,
                    value,
                    repr(exc),
                )

                return CharacterAction.failed(
                    command_str=raw_command,
                    protagonist=protagonist,
                    dialogue=dialogue,
                )

        return action

    @staticmethod
    def _append_parameter(
        action: "CharacterAction",
        param_type: str,
        value: str,
        valid_characters: list[Character],
        valid_locations: list[Location],
        valid_items: list[Item],
    ) -> None:
        if param_type == "character":
            target = next(
                (
                    char
                    for char in valid_characters
                    if char.name.lower() == value.lower()
                ),
                None,
            )

            if target is None:
                raise ValueError(f"Character {value} not found in valid_characters")

            action.parameters.append(target)
            return

        if param_type == "location":
            target = next(
                (
                    loc
                    for loc in valid_locations
                    if loc.name.lower() == value.lower()
                ),
                None,
            )
            action.parameters.append(target if target is not None else value)
            return

        if param_type == "item":
            target = next(
                (
                    item
                    for item in valid_items
                    if item.name.lower() == value.lower()
                ),
                None,
            )
            action.parameters.append(target if target is not None else value)
            return

        if param_type == "amount":
            try:
                action.parameters.append(int(value))
            except Exception:
                action.parameters.append(value)
            return

        if param_type == "content":
            content = value.strip('"')
            action.parameters.append(content)
            action.dialogue = content
            return

        action.parameters.append(value)


def get_guided_regex(
    skills: list[Skill],
    authorized_characters: list[Character],
    authorized_locations: list[Location],
    authorized_items: list[Item],
) -> re.Pattern:
    """
    Generate a combined regex pattern for all allowed protagonist skills.
    """
    character_names = [
        char.name
        for char in authorized_characters
    ]

    location_names = [
        loc.name
        for loc in authorized_locations
    ]

    item_names = [
        item.name
        for item in authorized_items
    ]

    combined_regex_parts = [
        skill.to_regex(
            character_names,
            location_names,
            item_names,
        )
        for skill in skills
    ]

    combined_regex = "|".join(combined_regex_parts)

    return re.compile(
        combined_regex,
        re.IGNORECASE,
    )