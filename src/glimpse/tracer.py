import inspect
import sys
from functools import wraps
from typing import Optional
from src.glimpse.config import Config
from src.glimpse.logwriter import LogWriter
from src.glimpse.logentry import LogEntry
from datetime import datetime
import pprint

class Tracer:
    
    def __init__(self, config: Config, writer_initiation = True):
        self._config = config
        self._writer = LogWriter(config, writer_initiation)

    @staticmethod
    def get_function_arguments(func, *args, **kwargs):
        # Build a function call string with actual arguments
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        call_parts = []
        for name, value in bound.arguments.items():
            call_parts.append(f"{name}={value!r}")
        call_str = f"{func.__name__}({', '.join(call_parts)})"
        return call_str


    def trace_function(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            exception = None
            result = None
            level = self._config.level

            call_str = self.get_function_arguments(func, *args, **kwargs)

            try:
                result = func(*args, **kwargs)
            except Exception as e:
                exception = str(e)
                level = "ERROR"

            end_time = datetime.now()

            PRECISION = 3
            duration_ms = round((end_time - start_time).total_seconds() * 1000, PRECISION)

            log_entry = LogEntry(
                entry_id=-1,  # You can let DBs handle real IDs later
                level=level,
                function=func.__qualname__,
                args=call_str,
                result=pprint.pformat(result, indent=2, width=80),
                timestamp=start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                duration_ms=str(duration_ms),
                exception=exception
            )

            self._writer.write(log_entry)

            if exception:
                raise Exception(exception)  # Re-raise the caught exception

            return result

        return wrapper

    def multi_trace(self, func):
        def wrapper(*args, **kwargs):
            exception = None
            result = None
            level = self._config.level

            call_str = self.get_function_arguments(func, *args, **kwargs)

            try:
                start_time = datetime.now()
                start_log_entry = LogEntry(
                    entry_id=-1,
                    level=level,
                    function=func.__qualname__,
                    args=self._truncate(call_str),
                    stage="START",
                    timestamp=start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                )

                self._writer.write(start_log_entry)

                result = func(*args, **kwargs)

                PRECISION = 3
                end_time = datetime.now()
                end_log_entry = LogEntry(
                    entry_id=-1,
                    level=level,
                    function=func.__qualname__,
                    args=self._truncate(call_str),
                    stage="END",
                    result=self._truncate(str(result)),
                    timestamp=end_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    duration_ms=round((end_time - start_time).total_seconds() * 1000, PRECISION)
                )

                self._writer.write(end_log_entry)

            except Exception as e:
                exception = str(e)
                level = "ERROR"

                error_log_entry = LogEntry(
                    entry_id=-1,
                    level=level,
                    function=func.__qualname__,
                    args=self._truncate(call_str),
                    stage="EXCEPTION",
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    exception=exception
                )

                self._writer.write(error_log_entry)
            
            if exception:
                raise Exception(exception)

        return wrapper
    
    # Call multi-trace for every function call automatically
    # Need to inspect call stack

    def _truncate(self, value: str) -> str:
        max_len = self._config.max_field_length
        if len(value) > max_len:
            return value[:max_len] + "..."
        return value

    def run(self, package_prefix: Optional[str] = None):
        pass

