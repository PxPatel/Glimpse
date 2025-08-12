import sys
import inspect
import pprint
from pathlib import Path
from typing import Optional
from functools import wraps
from datetime import datetime
from .config import Config 
from .policy.policy import TracingPolicy
from .writers.logentry import LogEntry
from .writers.logwriter import LogWriter

class Tracer:
    
    def __init__(self, config: Config, writer_initiation = True, policy: TracingPolicy = None):
        self._config = config
        self.policy = policy
        self._writer = LogWriter(config, writer_initiation)

        # Capture where tracer was initialized
        caller_frame = inspect.currentframe().f_back
        self._init_file = Path(caller_frame.f_code.co_filename).resolve()
        self._init_module_name = self._get_module_name_from_file(self._init_file)

        # Think about logging an entry at initialization for record of instance

    def _get_module_name_from_file(self, file_path: Path) -> Optional[str]:
        """
        Convert a file path to its Python module name.
        Returns None if it can't be determined.
        """
        try:
            # Handle common cases
            if file_path.name == '__main__.py':
                # For files run as modules (python -m package)
                return '__main__'
            
            if file_path.suffix == '.py':
                # For regular .py files, use the filename without extension
                return file_path.stem
            
            return None
        except Exception:
            return None

    def should_trace_function(self, func) -> bool:
        """
        Determine if a function should be traced based on tracing policy.
        
        Args:
            func: The function to check
            
        Returns:
            True if function should be traced, False otherwise
        """
        if not hasattr(func, '__module__'):
            return False
        
        module_name = func.__module__
        
        if self._init_module_name and module_name == self._init_module_name:
            return True
        
        # Check if it's an explicitly included external package or internal subpackage 
        return self._policy.should_trace_package(module_name)

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
    
    def _truncate(self, value: str) -> str:
        max_len = self._config.max_field_length
        if len(value) > max_len:
            return value[:max_len] + "..."
        return value

    def run(self, package_prefix: Optional[str] = None):
        pass

