import sys

from ..config import Config
from .logentry import LogEntry
from .json import JSONWriter
from .sqlite import SQLiteWriter

# Add support for multiple destinations

class LogWriter:

    def __init__(self, config: Config, writer_initiation = True):
        self._config = config
        self._initialized = writer_initiation
        self._writers = []
        if writer_initiation:
            self.initalize_writers()

    def initalize_writers(self):
        for dest in self._config._dest:
            self._writers.append(self._initialize_destination(dest))

    def _initialize_destination(self, dest):
        dest = dest.lower()

        if dest == "sqlite":
            return SQLiteWriter(self._config)
        elif dest == "jsonl" or dest == "json":
            return JSONWriter(self._config)
        elif dest == "mongo":
            pass
            # return MongoWriter(self._config)
        elif dest == "jaeger":
            from .jaeger import JaegerWriter
            endpoint = self._config.params.get("jaeger_endpoint", "http://localhost:4318/v1/traces")
            return JaegerWriter(endpoint=endpoint)
        else:
            raise ValueError(f"Unsupported log destination: {dest}")

    def write_span(self, span) -> None:
        for writer in self._writers:
            if hasattr(writer, "write_span"):
                try:
                    writer.write_span(span)
                except Exception as e:
                    print(f"Glimpse writer error ({writer.__class__.__name__}): {e}", file=sys.stderr)

    def write(self, entry: LogEntry):
        for writer in self._writers:
            try:
                writer.write(entry)
            except Exception as e:
                print(f"Glimpse writer error ({writer.__class__.__name__}): {e}", file=sys.stderr)

    def flush(self):
        for writer in self._writers:
            if hasattr(writer, "flush"):
                try:
                    writer.flush()
                except Exception as e:
                    print(f"Glimpse flush error ({writer.__class__.__name__}): {e}", file=sys.stderr)

    def close(self):
        for writer in self._writers:
            if hasattr(writer, "close"):
                writer.close()

    def add_destination(self, dest: str):
        if not isinstance(dest, str) or dest is None:
            raise ValueError(f"'dest' must be a non-empty string")
        
        success = self._config.add_destination(dest)
        if success:
            self._writers.append(self._initialize_destination(dest))

    def remove_destination(self, idx: int):
        if not isinstance(idx, int) or idx < 0:
            raise ValueError(f"'idx' must be be a positive integer")
        if idx >= len(self._writers):
            raise IndexError(f"'idx' must be within a valid index in the _writers array")

        success = self._config.remove_destination(idx)
        if success:
            self._writers.pop(idx)
            

    @property
    def get_destinations(self):
        return self._config.dest.copy()


