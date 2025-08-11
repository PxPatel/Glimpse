from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class LogEntry:
    entry_id: int
    level: str
    function: str
    args: str
    stage: str
    timestamp: datetime
    result: Optional[str] = None
    duration_ms: Optional[float] = None
    exception: Optional[str] = None

