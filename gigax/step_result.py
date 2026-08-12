# gigax/step_result.py

from typing import Optional
from pydantic import BaseModel, Field

from gigax.parse import CharacterAction


class StepResult(BaseModel):
    """
    Structured result for one NPC generation step.

    This object is designed for experiments. It keeps the raw model output,
    parser result, timing information, and decoding constraints in one place.
    """

    raw_output: str = ""
    parsed_action: Optional[CharacterAction] = None
    parse_success: bool = False

    guided_regex: str = ""
    guided_actions: list[str] = Field(default_factory=list)

    input_tokens: int = 0
    output_tokens: int = 0
    generation_time: float = 0.0
    vram_peak_mb: float = 0.0

    error: Optional[str] = None