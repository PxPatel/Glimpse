import os
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()
env = os.getenv("env", "DEV").upper()
load_dotenv(dotenv_path = f".env.{env.lower()}")

class Config:
    _CORE_KEYS = {"DEST", "LEVEL", "TRACE_ID"}
    _ACCEPTABLE_DEST = {"json", "jsonl", "sqllite", "mongo"}

    def __init__(
        self,
        dest: str | list[str] = "jsonl",
        level: str = "INFO",
        enable_trace_id: bool = False,
        params: Optional[Dict[str, str]] = None,
        env_override: bool = True,
        env_prefix: str = "GLIMPSE_",
        max_field_length = 512
    ):
        self._env_prefix = env_prefix.upper()

        self._dest = []
        if isinstance(dest, list):
            self._dest = [candidate.strip() for candidate in dest if candidate.strip() in self._ACCEPTABLE_DEST]
        elif isinstance(dest, str):
            self._dest = [dest] if dest in self._ACCEPTABLE_DEST else None
                
        self._level = level.upper()
        self._enable_trace_id = enable_trace_id
        self._params = params or {}
        self._max_field_length = max_field_length

        if env_override:
            self._load_from_env()

        if not self._dest:
            raise ValueError(f"Invalid destination: '{self._dest}'")

    def _load_from_env(self):
        # Core config overrides
        dest_str = os.getenv(self.build_env_var("DEST"), None)
        if dest_str and dest_str.strip():
            self._dest = [x.strip() for x in dest_str.split(",") if x.strip() in self._ACCEPTABLE_DEST]
        elif dest_str is not None and not dest_str.strip():
            self._dest = ''

        self._level = os.getenv(self.build_env_var("LEVEL"), self._level).upper()

        trace_id_val = os.getenv(self.build_env_var("TRACE_ID"), None)
        if trace_id_val is not None:
            self._enable_trace_id = trace_id_val.lower() in {"1", "true", "yes"}

        # Load destination-specific parameters
        for key, val in os.environ.items():
            if key.startswith(self._env_prefix):
                suffix = key[len(self._env_prefix):]
                if suffix not in self._CORE_KEYS:
                    self._params[suffix.lower()] = val

    def build_env_var(self, suffix: str):
        return f"{self._env_prefix}{suffix}"

    def add_destination(self, dest: str) -> bool:
        self._dest.append(dest)
        return True

    def remove_destination(self, idx: int) -> bool:
        self._dest.pop(idx)
        return True

    @property
    def dest(self) -> str:
        return self._dest

    @property
    def level(self) -> str:
        return self._level

    @property
    def enable_trace_id(self) -> bool:
        return self._enable_trace_id

    @property
    def params(self) -> Dict[str, str]:
        return self._params.copy()  # return a copy to prevent accidental mutation

    @property
    def env_prefix(self) -> str:
        return self._env_prefix

    @property
    def max_field_length(self) -> int:
        return self._max_field_length or 512
