from abc import ABC, abstractmethod
from typing import Any

class BaseWriter(ABC):
    """
    Abstract base class for all log writers.
    All log writers must implement this interface.
    """

    @abstractmethod
    def write(self, entry: Any) -> None:
        """
        Write a single log entry to the backend.
        `entry` is expected to be a structured object (e.g., LogEntry or dict).
        """
        pass

    def flush(self) -> None:
        """
        Optional: flush any buffers or pending writes.
        Override only if needed by backend (e.g., file, batch writers).
        """
        pass

    def close(self) -> None:
        """
        Optional: release any resources or close connections.
        Override if needed.
        """
        pass
