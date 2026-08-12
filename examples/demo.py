from __future__ import annotations

import argparse
import json
from dataclasses import (
    asdict,
    is_dataclass,
)
from pathlib import Path
from typing import Any

from gigax.mate import load_mate
from gigax.runtime import (
    NPCSecurityInput,
    NPCSecurityOutput,
)


def to_jsonable(
    value: Any,
) -> Any:
    if value is None:
        return None

    if is_dataclass(value):
        return {
            key: to_jsonable(item)
            for key, item
            in asdict(value).items()
        }

    if hasattr(
        value,
        "model_dump",
    ):
        return to_jsonable(
            value.model_dump()
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): to_jsonable(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            to_jsonable(item)
            for item in value
        ]

    return value


def layer1_summary(
    runtime: Any,
) -> dict[str, Any]:
    config = getattr(
        runtime,
        "mate_config",
        None,
    )

    layer1 = getattr(
        config,
        "layer1",
        None,
    )

    layer1_type = getattr(
        layer1,
        "type",
        "unknown",
    )

    system = getattr(
        runtime,
        "layer1_system",
        {},
    )

    if not isinstance(
        system,
        dict,
    ):
        system = {}

    return {
        "type": str(
            layer1_type
        ),
        "name": system.get(
            "name",
            "",
        ),
        "method": system.get(
            "method",
            "",
        ),
        "fusion_threshold": (
            system.get(
                "fusion_threshold",
                0.0,
            )
        ),
        "layers": system.get(
            "layers",
            None,
        ),
    }


def build_sample(
    *,
    player_input: str,
    injection_active: bool = False,
    trusted_threat: bool = False,
    allow_attack: bool = False,
    sample_id: str = "",
) -> NPCSecurityInput:
    return NPCSecurityInput(
        player_input=player_input,
        injection_active=(
            injection_active
        ),
        trusted_threat=(
            trusted_threat
        ),
        allow_attack=(
            allow_attack
        ),
        sample_id=sample_id,
    )


def run_once(
    *,
    runtime: Any,
    config_path: str | Path,
    player_input: str,
    defense: str | None = None,
    injection_active: bool = False,
    trusted_threat: bool = False,
    allow_attack: bool = False,
    sample_id: str = "",
) -> dict[str, Any]:
    sample = build_sample(
        player_input=player_input,
        injection_active=(
            injection_active
        ),
        trusted_threat=(
            trusted_threat
        ),
        allow_attack=(
            allow_attack
        ),
        sample_id=sample_id,
    )

    selected_defense = (
        defense
        or getattr(
            runtime,
            "default_defense",
            "raac_multiaction",
        )
    )

    output: NPCSecurityOutput = (
        runtime.run(
            sample=sample,
            defense=(
                selected_defense
            ),
        )
    )

    return {
        "mate": {
            "config": str(
                config_path
            ),
            "layer1": (
                layer1_summary(
                    runtime
                )
            ),
            "defense": (
                selected_defense
            ),
        },
        "input": {
            "player_input": (
                player_input
            ),
            "injection_active": (
                injection_active
            ),
            "trusted_threat": (
                trusted_threat
            ),
            "allow_attack": (
                allow_attack
            ),
            "sample_id": (
                sample_id
            ),
        },
        "result": to_jsonable(
            output
        ),
    }


def print_result(
    payload: dict[str, Any],
) -> None:
    print(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


def run_interactive(
    *,
    runtime: Any,
    config_path: Path,
    defense: str | None,
    injection_active: bool,
    trusted_threat: bool,
    allow_attack: bool,
) -> None:
    print(
        "MATE interactive demo"
    )

    print(
        "Layer 1:",
        layer1_summary(
            runtime
        )["type"],
    )

    print(
        "Type 'exit' or 'quit' "
        "to stop."
    )

    turn = 0

    while True:
        text = input(
            "\nPlayer> "
        ).strip()

        if text.lower() in {
            "exit",
            "quit",
        }:
            break

        if not text:
            continue

        turn += 1

        payload = run_once(
            runtime=runtime,
            config_path=(
                config_path
            ),
            player_input=text,
            defense=defense,
            injection_active=(
                injection_active
            ),
            trusted_threat=(
                trusted_threat
            ),
            allow_attack=(
                allow_attack
            ),
            sample_id=(
                f"interactive_{turn}"
            ),
        )

        print_result(
            payload
        )


def build_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one complete MATE "
            "NPC-security pipeline."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "MATE YAML configuration."
        ),
    )

    parser.add_argument(
        "--prompt",
        default="",
        help=(
            "Single-turn player input."
        ),
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Run an interactive session."
        ),
    )

    parser.add_argument(
        "--defense",
        choices=[
            "regex",
            "raac_tiered",
            "raac_multiaction",
        ],
        default=None,
        help=(
            "Override the runtime "
            "defense mode."
        ),
    )

    parser.add_argument(
        "--injection-active",
        action="store_true",
        help=(
            "Wrap player input in the "
            "default environment as "
            "low-privilege injected text."
        ),
    )

    parser.add_argument(
        "--trusted-threat",
        action="store_true",
        help=(
            "Trusted world state confirms "
            "an immediate physical threat."
        ),
    )

    parser.add_argument(
        "--allow-attack",
        action="store_true",
        help=(
            "Trusted runtime policy "
            "explicitly authorizes Attack."
        ),
    )

    parser.add_argument(
        "--sample-id",
        default="",
    )

    return parser


def main(
) -> None:
    args = (
        build_parser()
        .parse_args()
    )

    config_path = Path(
        args.config
    ).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: "
            f"{config_path}"
        )

    if (
        not args.interactive
        and not args.prompt
    ):
        raise SystemExit(
            "Provide --prompt or "
            "use --interactive."
        )

    runtime = load_mate(
        config_path
    )

    if args.interactive:
        run_interactive(
            runtime=runtime,
            config_path=(
                config_path
            ),
            defense=args.defense,
            injection_active=(
                args.injection_active
            ),
            trusted_threat=(
                args.trusted_threat
            ),
            allow_attack=(
                args.allow_attack
            ),
        )
        return

    payload = run_once(
        runtime=runtime,
        config_path=config_path,
        player_input=args.prompt,
        defense=args.defense,
        injection_active=(
            args.injection_active
        ),
        trusted_threat=(
            args.trusted_threat
        ),
        allow_attack=(
            args.allow_attack
        ),
        sample_id=(
            args.sample_id
        ),
    )

    print_result(
        payload
    )

    error = (
        payload
        .get(
            "result",
            {},
        )
        .get(
            "error"
        )
    )

    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
