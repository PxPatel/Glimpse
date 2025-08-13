import uuid
import os
import threading
from typing import Optional
from ..tracer import Tracer

class IDGenerator:
    """Thread and multi-processing safe ID generator for tracing."""
    
    def __init__(self):
        self._lock = threading.Lock()

        self._process_id = os.getpid()
        self._instance_id = uuid.uuid4().hex[:8]
        self._entry_counter = 0
        
        self._current_trace_id: Optional[str] = None
    
    def new_entry_id(self) -> str:
        """Generate incremental entry ID (unique across restarts)."""
        with self._lock:
            self._entry_counter += 1

            # Base36 uses 0-9, a-z (36 characters)
            # counter_b36 = self._to_base36(self._entry_counter)

            # Format: pid-instance-counter (e.g., "1234-a1b2c3d4-000001") -- 1M entries per process
            return f"{self._process_id}-{self._instance_id}-{self._entry_counter:06d}"
    
    def _to_base36(self, num: int) -> str:
        """Convert number to base36 (0-9, a-z)."""
        if num == 0:
            return "0"
        chars = "0123456789abcdefghijklmnopqrstuvwxyz"
        result = ""
        while num:
            result = chars[num % 36] + result
            num //= 36
        return result.zfill(6)  # Pad to 6 chars: "000001" to "zzzzz5"


    def new_call_id(self) -> str:
        """Generate unique call ID for function correlation."""
        return f"call_{uuid.uuid4().hex[:12]}"  # Shorter for readability
    
    def new_trace_id(self) -> str:
        """Generate unique trace ID for distributed correlation."""
        return f"trace_{uuid.uuid4().hex[:12]}"
    
    def set_trace_id(self, trace_id: str):
        """Set trace ID for distributed tracing context."""
        self._current_trace_id = trace_id
    
    def get_current_trace_id(self) -> Optional[str]:
        """Get current trace ID (for distributed context)."""
        return self._current_trace_id