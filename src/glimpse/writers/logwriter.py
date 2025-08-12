from ..config import Config
from .logentry import LogEntry
from .json import JSONWriter

class LogWriter:

    def __init__(self, config: Config, writer_initiation = True):
        self._config = config
        self._backend_type = config.dest
        self._initialized = writer_initiation
        self._writer = None 
        if writer_initiation:
            self._writer = self._initialize_backend()

    def _initialize_backend(self):
        if self._writer:
            raise RuntimeError("Backend writer is already initialized.")

        dest = self._config.dest.lower()
        if dest == "sqllite":
            pass
            # return SQLiteWriter(self._config)
        elif dest == "jsonl" or dest == "json":
            return JSONWriter(self._config)
        elif dest == "mongo":
            pass
            # return MongoWriter(self._config)
        else:
            raise ValueError(f"Unsupported log destination: {dest}")
    
    def write(self, entry: LogEntry):
        self._writer.write(entry)

    def flush(self):
        if hasattr(self._writer, "flush"):
            self._writer.flush()

    def close(self):
        if hasattr(self._writer, "close"):
            self._writer.close()

    @property
    def backend_name(self):
        return self._backend_type


