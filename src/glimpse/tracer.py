import sys
import inspect
import pprint
from pathlib import Path
from typing import Optional
from functools import wraps
from datetime import datetime
from .config import Config
from .policy import TracingPolicy
from .writers.logentry import LogEntry
from .writers.logwriter import LogWriter
from .common.ids import IDGenerator
from .span import Span, SpanEvent
from .context import get_active_span, set_active_span, reset_active_span
from .propagation import inject as _inject, extract as _extract


class _SpanContext:
    def __init__(self, tracer, name: str):
        self._tracer = tracer
        self._name = name
        self._span: Optional[Span] = None
        self._token = None

    def __enter__(self) -> Span:
        parent = get_active_span()
        now = datetime.utcnow().isoformat()
        self._span = Span(
            trace_id=self._tracer._id_generator.get_current_trace_id() or self._tracer._id_generator.new_trace_id(),
            span_id=self._tracer._id_generator.new_call_id(),
            name=self._name,
            start_time=now,
            parent_span_id=parent.span_id if parent else None,
        )
        self._token = set_active_span(self._span)
        return self._span

    def __exit__(self, exc_type, exc_val, exc_tb):
        span = self._span
        span.end_time = datetime.utcnow().isoformat()
        if exc_type is not None:
            span.status = "error"
            span.events.append(SpanEvent(
                name="exception",
                timestamp=span.end_time,
                attributes={
                    "exception.type": exc_type.__name__,
                    "exception.message": str(exc_val),
                },
            ))
        reset_active_span(self._token)
        self._tracer._on_span_end(span)
        return False  # never suppress exceptions

class _AsyncSpanContext:
    def __init__(self, tracer, name: str):
        self._tracer = tracer
        self._name = name
        self._span: Optional[Span] = None
        self._token = None

    async def __aenter__(self) -> Span:
        parent = get_active_span()
        now = datetime.utcnow().isoformat()
        self._span = Span(
            trace_id=self._tracer._id_generator.get_current_trace_id() or self._tracer._id_generator.new_trace_id(),
            span_id=self._tracer._id_generator.new_call_id(),
            name=self._name,
            start_time=now,
            parent_span_id=parent.span_id if parent else None,
        )
        self._token = set_active_span(self._span)
        return self._span

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        span = self._span
        span.end_time = datetime.utcnow().isoformat()
        if exc_type is not None:
            span.status = "error"
            span.events.append(SpanEvent(
                name="exception",
                timestamp=span.end_time,
                attributes={
                    "exception.type": exc_type.__name__,
                    "exception.message": str(exc_val),
                },
            ))
        reset_active_span(self._token)
        self._tracer._on_span_end(span)
        return False


class Tracer:
    
    def __init__(self, config: Config, writer_initiation = True, policy: TracingPolicy = None):
        self._config = config
        self._policy = policy
        self._writer = LogWriter(config, writer_initiation)
        self._id_generator = IDGenerator()

        # Capture where tracer was initialized
        caller_frame = inspect.currentframe().f_back
        self._init_module_name = caller_frame.f_globals.get('__name__')

        # Execution variables
        self._call_metadata = {}
        self._tracing_active = None

        # Think about logging an entry at initialization for record of instance

    # Removed from use
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
        trace_bool = self._policy.should_trace_package(module_name)
        return trace_bool

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
        def wrapper(*args, **kwargs):
            exception = None
            result = None
            level = self._config.level

            # Generate call_id ONCE for this function execution
            call_id = self._id_generator.new_call_id()
            call_str = self.get_function_arguments(func, *args, **kwargs)

            try:
                start_time = datetime.now()
                start_log_entry = LogEntry(
                    entry_id=self._id_generator.new_entry_id(),
                    call_id=call_id, 
                    trace_id=self._id_generator.get_current_trace_id() if self._config.enable_trace_id else None,
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
                    entry_id=self._id_generator.new_entry_id(),
                    call_id=call_id,
                    trace_id=self._id_generator.get_current_trace_id() if self._config.enable_trace_id else None,
                    level=level,
                    function=func.__qualname__,
                    args=self._truncate(call_str),
                    stage="END",
                    result=self._truncate(str(result)),
                    timestamp=end_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    duration_ms=round((end_time - start_time).total_seconds() * 1000, PRECISION)
                )

                self._writer.write(end_log_entry)
                return result

            except Exception as e:
                exception_str = str(e)
                level = self._config.level

                error_log_entry = LogEntry(
                    entry_id=self._id_generator.new_entry_id(),
                    call_id=call_id,
                    trace_id=self._id_generator.get_current_trace_id() if self._config.enable_trace_id else None,
                    level=level,
                    function=func.__qualname__,
                    args=self._truncate(call_str),
                    stage="EXCEPTION",
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    exception=exception_str
                )

                self._writer.write(error_log_entry)
                raise

        return wrapper

    def trace_async_function(self, func):
        """Decorator for async functions. Produces a span for each call."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            parent = get_active_span()
            now = datetime.utcnow().isoformat()
            span = Span(
                trace_id=self._id_generator.get_current_trace_id() or self._id_generator.new_trace_id(),
                span_id=self._id_generator.new_call_id(),
                name=func.__qualname__,
                start_time=now,
                parent_span_id=parent.span_id if parent else None,
            )
            token = set_active_span(span)
            try:
                result = await func(*args, **kwargs)
                span.end_time = datetime.utcnow().isoformat()
                return result
            except Exception as exc:
                span.end_time = datetime.utcnow().isoformat()
                span.status = "error"
                span.events.append(SpanEvent(
                    name="exception",
                    timestamp=span.end_time,
                    attributes={
                        "exception.type": type(exc).__name__,
                        "exception.message": str(exc),
                    },
                ))
                raise
            finally:
                reset_active_span(token)
                self._on_span_end(span)
        return wrapper

    def _truncate(self, value: str) -> str:
        max_len = self._config.max_field_length
        if len(value) > max_len:
            return value[:max_len] + "..."
        return value

    def run(self):
        """
        Start automatic tracing of function calls based on TracingPolicy.
        Uses sys.settrace() to intercept all function calls.
        """
        if not self._policy:
            raise RuntimeError("TracingPolicy is required for automatic tracing. Initialize Tracer with a policy.")
        
        # Initialize call stack tracking
        self._tracing_active = True
        
        # Set the trace function
        sys.settrace(self._trace_calls)
        
        print(f"Glimpse automatic tracing started. Trace depth: {self._policy.trace_depth}")


    def stop(self):
        """Stop automatic tracing and clean up resources."""
        sys.settrace(None)
        self._tracing_active = False
        self._call_metadata.clear()
        self._writer.flush()
        self._writer.close()
        print("Glimpse automatic tracing stopped.")

    def _trace_calls(self, frame, event, arg):
        """
        Trace function called by sys.settrace() for each function call/return.
        
        Args:
            frame: Current execution frame
            event: Type of event ('call', 'return', 'exception', etc.)
            arg: Event-specific argument
            
        Returns:
            This function or None to continue/stop tracing this frame
        """
        if not self._tracing_active:
            return None
        try:
            if event == 'call':
                return self._handle_function_call(frame)
            elif event == 'return':
                return self._handle_function_return(frame, arg)
            elif event == 'exception':
                return self._handle_function_exception(frame, arg)
                
        except Exception as e:
            # Tracer should never break user code
            print(f"Glimpse tracer error: {e}")
            
        return self._trace_calls

    def _handle_function_call(self, frame):
        """Handle a function call event."""
        # Check trace depth limit
        if len(self._call_metadata) >= self._policy.trace_depth:
            return None
 
        # Get function info
        func_name = frame.f_code.co_name
        module_name = frame.f_globals.get('__name__', '')
        filename = frame.f_code.co_filename
        
        # Create a mock function object for policy checking
        mock_func = type('MockFunction', (), {
            '__module__': module_name,
            '__name__': func_name,
            '__qualname__': func_name
        })()
        
        # Check if we should trace this function
        if not self.should_trace_function(mock_func):
            return None
        
        # Generate new call_id to identify execution
        call_id = self._id_generator.new_call_id()
        
        # Track call in stack
        active = get_active_span()
        call_info = {
            'call_id': call_id,
            'function_name': func_name,
            'module_name': module_name,
            'qualname': f"{module_name}.{func_name}" if module_name else func_name,
            'start_time': datetime.now(),
            'frame': frame,
            'parent_span_id': active.span_id if active else None,
            'trace_id': (active.trace_id if active else None) or self._id_generator.get_current_trace_id(),
        }
        
        self._call_metadata[frame] = call_info

        # Log function entry
        self._log_function_entry(call_info, frame)
        
        return self._trace_calls

    def _handle_function_return(self, frame, return_value):
        """Handle a function return event."""
        if not self._call_metadata:
            return self._trace_calls
        
        call_info = self._call_metadata.pop(frame, None)

        if call_info:
            self._log_function_exit(call_info, return_value)
            
        return self._trace_calls

    def _handle_function_exception(self, frame, exc_info):
        """Handle a function exception event."""
        if not self._call_metadata:
            return self._trace_calls

        call_info = self._call_metadata.pop(frame, None)

        if call_info:
            self._log_function_exception(call_info, exc_info)
            
        return self._trace_calls
    
    def _get_function_args_from_frame(self, frame) -> str:
        """Extract function arguments from frame for logging."""
        try:
            args = []
            code = frame.f_code
            
            # Get argument names and values
            for i, arg_name in enumerate(code.co_varnames[:code.co_argcount]):
                if arg_name in frame.f_locals:
                    value = frame.f_locals[arg_name]
                    # Truncate large values
                    value_str = repr(value)
                    if len(value_str) > 100:
                        value_str = value_str[:97] + "..."
                    args.append(f"{arg_name}={value_str}")
                    
            return f"({', '.join(args)})"
        except Exception:
            return "(args unavailable)"

    def _log_function_entry(self, call_info: dict, frame):
        """Log function entry."""
        args_str = self._get_function_args_from_frame(frame)
        
        log_entry = LogEntry(
            entry_id=self._id_generator.new_entry_id(),
            call_id=call_info['call_id'],
            trace_id=self._id_generator.get_current_trace_id() if self._config.enable_trace_id else None,

            level=self._config.level,
            function=call_info['qualname'],
            args=self._truncate(f"{call_info['function_name']}{args_str}"),
            stage="START",
            timestamp=call_info['start_time'].strftime("%Y-%m-%d %H:%M:%S.%f")
        )
        
        self._writer.write(log_entry)

    def _log_function_exit(self, call_info: dict, return_value):
        """Log function exit with return value."""
        end_time = datetime.now()
        duration_ms = round((end_time - call_info['start_time']).total_seconds() * 1000, 3)
        
        # Format return value
        result_str = self._truncate(pprint.pformat(return_value, indent=2, width=80))
        
        log_entry = LogEntry(
            entry_id=self._id_generator.new_entry_id(),
            call_id=call_info['call_id'],
            trace_id=self._id_generator.get_current_trace_id() if self._config.enable_trace_id else None,

            level=self._config.level,
            function=call_info['qualname'],
            args=call_info['function_name'],
            stage="END",
            result=result_str,
            timestamp=end_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
            duration_ms=str(duration_ms)
        )
        
        self._writer.write(log_entry)

    def _log_function_exception(self, call_info: dict, exc_info):
        """Log function exception."""
        exc_type, exc_value, exc_traceback = exc_info
        exception_str = f"{exc_type.__name__}: {exc_value}"
        
        log_entry = LogEntry(
            entry_id=self._id_generator.new_entry_id(),
            call_id=call_info['call_id'],
            trace_id=self._id_generator.get_current_trace_id() if self._config.enable_trace_id else None,

            level=self._config.level,
            function=call_info['qualname'],
            args=call_info['function_name'],
            stage="EXCEPTION",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            exception=self._truncate(exception_str)
        )
        
        self._writer.write(log_entry)

    def span(self, name: str) -> "_SpanContext":
        """Return a context manager that creates and activates a Span."""
        return _SpanContext(self, name)

    def async_span(self, name: str) -> "_AsyncSpanContext":
        """Return an async context manager that creates and activates a Span."""
        return _AsyncSpanContext(self, name)

    def _on_span_end(self, span: Span) -> None:
        """Write completed span to writer if it supports spans."""
        try:
            self._writer.write_span(span)
        except AttributeError:
            pass  # Writer doesn't support spans — silently skip

    def inject(self, headers: dict) -> None:
        """
        Inject the active span's W3C traceparent into `headers`.

        Mutates `headers` in-place. No-op when no span is active.
        Use before making an outbound HTTP call so the downstream
        service can continue the trace.
        """
        _inject(headers)

    def extract(self, headers: dict):
        """
        Extract W3C traceparent from `headers`.

        Returns a dict {"trace_id": str, "parent_span_id": str} if a
        valid traceparent is present, otherwise None.
        Use when receiving an inbound HTTP request to continue the
        upstream trace: pass the returned context into tracer.span()
        or use it to set trace_id / parent_span_id on a new Span.
        """
        return _extract(headers)

    def __enter__(self):
        """Context manager entry - start tracing."""
        self.run()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop tracing."""
        self.stop()
        return False  # Don't suppress exceptions