import json
from pathlib import Path
from typing import Any, TextIO
from ..config import Config
from .base import BaseWriter


class JSONWriter(BaseWriter):
    def __init__(self, config: Config):
        self._config = config
        self._path = config.params.get("log_path", None) or "glimpse.jsonl"

        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = open(self._path, mode="a", encoding="utf-8")

    def write(self, entry: Any) -> None:
        # Convert to dict if it's a dataclass (like LogEntry)
        if hasattr(entry, "__dict__"):
            entry = entry.__dict__
        elif not isinstance(entry, dict):
            raise TypeError("Log entry must be a dict or dataclass")

        json_line = json.dumps(entry, ensure_ascii=False)
        self._file.write(json_line + "\n")

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()
