import dataclasses
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
        if dataclasses.is_dataclass(entry) and not isinstance(entry, type):
            d = dataclasses.asdict(entry)
            # Tag the record type so consumers can distinguish spans from log entries
            from ..span import Span  # local import to avoid circular at module level
            if isinstance(entry, Span):
                d["record_type"] = "span"
            else:
                d["record_type"] = "log_entry"
        elif isinstance(entry, dict):
            d = entry
        else:
            raise TypeError("Log entry must be a dict or dataclass")

        json_line = json.dumps(d, ensure_ascii=False, default=str)
        self._file.write(json_line + "\n")

    def write_span(self, span) -> None:
        self.write(span)

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()
