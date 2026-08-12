"""
Memory Module
"""
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

class Memory(BaseModel):
    event: str
    action: str
    parameters: List[str]
    dialogue: str
    speaker: str

    thought: Optional[str] = None # 當時的內心想法
    importance: int = Field(default=5, ge=1, le=10) # 1-10 權重
    created_at: datetime = Field(default_factory=datetime.now)
    # is_truncated: bool = False # 標記是否為被截斷的歷史
    
    tags: List[str] = Field(default_factory=list)
    risk_level: int = Field(default=0, ge=0, le=10)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def summary(self):
        return f"{self.speaker}: {self.dialogue} ({self.action})"
        
    def __str__(self):
        ts = self.created_at.strftime("%H:%M:%S")
        return f"[{ts}] {self.speaker}: {self.dialogue} (Action: {self.action})"

    def narrative(self) -> str:
        if self.thought:
            return f'<{self.speaker}\'s thinking：{self.thought}>\n{self.speaker}: "{self.dialogue}"'
        return f'{self.speaker}: "{self.dialogue}"'

    def to_dict(self):
        return self.model_dump()


