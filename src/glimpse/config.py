import os
from pathlib import Path
from typing import List, Optional, Dict
from dotenv import load_dotenv

def load_env_files():
    """
    Load environment files, trying multiple locations:
    1. Current working directory
    2. Directory where this config.py file is located
    3. Project root (parent directories)
    """
    # Try current working directory first
    cwd = Path.cwd()
    env_file = cwd / ".env"
    
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
        env_value = os.getenv("env", "DEV").upper()
        env_specific = cwd / f".env.{env_value.lower()}"
        if env_specific.exists():
            load_dotenv(dotenv_path=env_specific)
        return True
    
    # Try directory where this config.py file is located and walk up
    config_dir = Path(__file__).resolve().parent
    while config_dir != config_dir.parent:  # Stop at filesystem root
        env_file = config_dir / ".env"
        if env_file.exists():
            load_dotenv(dotenv_path=env_file)
            env_value = os.getenv("env", "DEV").upper()
            env_specific = config_dir / f".env.{env_value.lower()}"
            if env_specific.exists():
                load_dotenv(dotenv_path=env_specific)
            return True
        config_dir = config_dir.parent
    
    # Fallback to original behavior
    load_dotenv()
    env = os.getenv("env", "DEV").upper()
    load_dotenv(dotenv_path=f".env.{env.lower()}")
    return False

# Load environment files when module is imported
load_env_files()

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
            self._dest = [dest] if dest in self._ACCEPTABLE_DEST else []
                
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
        if dest not in self._ACCEPTABLE_DEST:
            return False
        
        if dest not in self._dest:
            self._dest.append(dest)

        return True 
    
    def remove_destination(self, idx: int) -> bool:
        if idx < len(self._dest):
            self._dest.pop(idx)
            return True

        raise IndexError(f"'idx' not in range of destination list")
        
    @property
    def dest(self) -> List[str]:
        return self._dest.copy()

    @property
    def dest_string(self) -> str:
        """Return destinations as comma-separated string for backwards compatibility."""
        return ','.join(self._dest)

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
