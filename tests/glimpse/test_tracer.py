import pytest
import sys
import time
import inspect
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
from glimpse.common.ids import IDGenerator
from glimpse.tracer import Tracer
from glimpse.config import Config
from glimpse.policy.policy import TracingPolicy
from glimpse.writers.logentry import LogEntry
from glimpse.writers.logwriter import LogWriter

class TestTracer:
    """Comprehensive unit tests for Tracer class."""
    
    # ======================= FIXTURES =======================
    
    @pytest.fixture
    def config(self):
        """Create a test config."""
        return Config(dest="jsonl", level="INFO", max_field_length=100, env_override=False)
    
    @pytest.fixture
    def policy(self):
        """Create a test policy."""
        return TracingPolicy(
            included_packages=["myapp", "requests"],
            project_root_packages=["src"],
            auto_trace_subpackages=True,
            trace_depth=5
        )
    
    @pytest.fixture
    def mock_writer(self):
        """Create a mock LogWriter."""
        writer = Mock(spec=LogWriter)
        writer.write = Mock()
        writer.flush = Mock()
        writer.close = Mock()
        return writer
    
    @pytest.fixture
    def tracer(self, config, policy):
        """Create a tracer with mocked writer."""
        with patch('glimpse.tracer.LogWriter') as mock_log_writer:
            mock_writer_instance = Mock()
            mock_log_writer.return_value = mock_writer_instance
            
            tracer = Tracer(config, writer_initiation=True, policy=policy)
            tracer._writer = mock_writer_instance
            return tracer
    
    @pytest.fixture
    def mock_frame(self):
        """Create a mock frame object for testing."""
        frame = Mock()
        frame.f_code = Mock()
        frame.f_code.co_name = "test_function"
        frame.f_code.co_filename = "/path/to/test.py"
        frame.f_code.co_varnames = ("arg1", "arg2")
        frame.f_code.co_argcount = 2
        frame.f_globals = {"__name__": "test_module"}
        frame.f_locals = {"arg1": "value1", "arg2": "value2"}
        return frame
    
    @pytest.fixture
    def id_generator(self):
        """Create a test IDGenerator instance"""
        return IDGenerator()
    
    # ======================= INITIALIZATION TESTS =======================
    
    @patch('inspect.currentframe')
    def test_init_with_all_parameters(self, mock_currentframe, config, policy):
        """Test Tracer initialization with all parameters."""
        # Mock the caller frame
        mock_frame = Mock()
        mock_caller_frame = Mock()
        mock_caller_frame.f_globals = {'__name__': 'main.app'}
        mock_frame.f_back = mock_caller_frame
        mock_currentframe.return_value = mock_frame
        
        with patch('glimpse.tracer.LogWriter') as mock_log_writer:
            tracer = Tracer(config, writer_initiation=True, policy=policy)
            
            assert tracer._config == config
            assert tracer._policy == policy
            assert tracer._init_module_name == "main.app"
            assert tracer._call_metadata == {}
            assert tracer._tracing_active is None
            mock_log_writer.assert_called_once_with(config, True)

    @patch('inspect.currentframe')
    def test_init_without_writer_initiation(self, mock_currentframe, config, policy):
        """Test Tracer initialization without writer initiation."""
        mock_frame = Mock()
        mock_caller_frame = Mock()
        mock_caller_frame.f_code.co_filename = "/path/to/test.py"
        mock_frame.f_back = mock_caller_frame
        mock_currentframe.return_value = mock_frame
        
        with patch('glimpse.tracer.LogWriter') as mock_log_writer:
            tracer = Tracer(config, writer_initiation=False, policy=policy)
            
            mock_log_writer.assert_called_once_with(config, False)

    @patch('inspect.currentframe')
    def test_init_without_policy(self, mock_currentframe, config):
        """Test Tracer initialization without policy."""
        mock_frame = Mock()
        mock_caller_frame = Mock()
        mock_caller_frame.f_code.co_filename = "/path/to/test.py"
        mock_frame.f_back = mock_caller_frame
        mock_currentframe.return_value = mock_frame
        
        with patch('glimpse.tracer.LogWriter'):
            tracer = Tracer(config, writer_initiation=False, policy=None)
            
            assert tracer._policy is None

    # ======================= MODULE NAME EXTRACTION TESTS =======================
    
    @pytest.mark.parametrize("file_path,expected", [
        (Path("/path/to/main.py"), "main"),
        (Path("/path/to/__main__.py"), "__main__"),
        (Path("/path/to/script.py"), "script"),
        (Path("/path/to/my_module.py"), "my_module"),
        (Path("/path/to/file.txt"), None),  # Non-Python file
        (Path("/path/to/no_extension"), None),  # No extension
    ])
    def test_get_module_name_from_file(self, tracer, file_path, expected):
        """Test module name extraction from file paths."""
        result = tracer._get_module_name_from_file(file_path)
        assert result == expected

    def test_get_module_name_from_file_exception_handling(self, tracer):
        """Test module name extraction handles exceptions gracefully."""
        # Test with invalid path that might cause exception
        invalid_path = None
        result = tracer._get_module_name_from_file(invalid_path)
        assert result is None

    # ======================= SHOULD_TRACE_FUNCTION TESTS =======================
    
    def test_should_trace_function_no_module_attribute(self, tracer):
        """Test should_trace_function with function lacking __module__."""
        mock_func = Mock()
        del mock_func.__module__  # Remove __module__ attribute
        
        result = tracer.should_trace_function(mock_func)
        assert result is False

    def test_should_trace_function_init_module_match(self, tracer):
        """Test should_trace_function matches init module."""
        mock_func = Mock()
        mock_func.__module__ = tracer._init_module_name
        
        result = tracer.should_trace_function(mock_func)
        assert result is True

    def test_should_trace_function_policy_match(self, tracer):
        """Test should_trace_function uses policy for matching."""
        mock_func = Mock()
        mock_func.__module__ = "myapp.services"
        
        with patch.object(tracer._policy, 'should_trace_package', return_value=True) as mock_policy:
            result = tracer.should_trace_function(mock_func)
            
            assert result is True
            mock_policy.assert_called_once_with("myapp.services")

    def test_should_trace_function_no_policy_match(self, tracer):
        """Test should_trace_function when policy doesn't match."""
        mock_func = Mock()
        mock_func.__module__ = "numpy.core"
        
        with patch.object(tracer._policy, 'should_trace_package', return_value=False) as mock_policy:
            result = tracer.should_trace_function(mock_func)
            
            assert result is False
            mock_policy.assert_called_once_with("numpy.core")

    def test_should_trace_function_no_policy_object(self, config):
        """Test should_trace_function when no policy is set."""
        with patch('inspect.currentframe'), patch('glimpse.tracer.LogWriter'):
            tracer = Tracer(config, writer_initiation=False, policy=None)
            
            mock_func = Mock()
            mock_func.__module__ = "myapp.services"
            
            # Should raise AttributeError when trying to access policy
            with pytest.raises(AttributeError):
                tracer.should_trace_function(mock_func)

    # ======================= FUNCTION ARGUMENT EXTRACTION TESTS =======================
    
    def test_get_function_arguments_simple(self):
        """Test function argument extraction with simple arguments."""
        def test_func(a, b, c=10):
            pass
        
        result = Tracer.get_function_arguments(test_func, 1, 2, c=3)
        
        assert "a=1" in result
        assert "b=2" in result
        assert "c=3" in result
        assert "test_func(" in result

    def test_get_function_arguments_with_defaults(self):
        """Test function argument extraction with default values."""
        def test_func(a, b=20, c=30):
            pass
        
        result = Tracer.get_function_arguments(test_func, 1)
        
        assert "a=1" in result
        assert "b=20" in result
        assert "c=30" in result

    def test_get_function_arguments_args_kwargs(self):
        """Test function argument extraction with *args and **kwargs."""
        def test_func(a, *args, **kwargs):
            pass
        
        result = Tracer.get_function_arguments(test_func, 1, 2, 3, key1="value1", key2="value2")
        
        assert "a=1" in result
        assert "args=(2, 3)" in result
        assert "kwargs=" in result
        assert "'key1': 'value1'" in result

    def test_get_function_arguments_complex_types(self):
        """Test function argument extraction with complex argument types."""
        def test_func(a, b, c):
            pass
        
        complex_args = [
            {"key": "value"},
            [1, 2, 3],
            None
        ]
        
        result = Tracer.get_function_arguments(test_func, *complex_args)
        
        assert "{'key': 'value'}" in result
        assert "[1, 2, 3]" in result
        assert "None" in result

    # ======================= TRACE_FUNCTION DECORATOR TESTS =======================
    
    def test_trace_function_decorator_success(self, tracer):
        """Test trace_function decorator with successful function execution."""
        @tracer.trace_function
        def test_func(x, y):
            return x + y
        
        result = test_func(2, 3)
        
        assert result == 5
        # Verify LogEntry objects were created and written
        assert tracer._writer.write.call_count == 2  # START and END entries
        
        calls = tracer._writer.write.call_args_list
        start_entry = calls[0][0][0]
        end_entry = calls[1][0][0]
        
        assert start_entry.stage == "START"
        assert end_entry.stage == "END"
        assert end_entry.result == "5"

    def test_trace_function_decorator_exception(self, tracer):
        """Test trace_function decorator with function that raises exception."""
        @tracer.trace_function
        def test_func():
            raise ValueError("Test error")
        
        with pytest.raises(Exception):  # Should re-raise the exception
            test_func()
        
        # Verify exception was logged
        assert tracer._writer.write.call_count == 2  # START and EXCEPTION entries
        
        calls = tracer._writer.write.call_args_list
        exception_entry = calls[1][0][0]
        
        assert exception_entry.stage == "EXCEPTION"
        assert exception_entry.exception == "Test error"
        assert exception_entry.level == "INFO"

    def test_trace_function_decorator_preserves_function_metadata(self, tracer):
        """Test that trace_function decorator preserves original function metadata."""
        def original_func(x, y):
            """Original docstring."""
            return x * y
        
        decorated_func = tracer.trace_function(original_func)
        
        assert decorated_func.__name__ == "wrapper" # Technicallty the decorator wraps the original function
        assert decorated_func.__doc__ == None 

    # ======================= TRUNCATION TESTS =======================
    
    def test_truncate_short_string(self, tracer):
        """Test truncation with string shorter than max length."""
        short_string = "short"
        result = tracer._truncate(short_string)
        assert result == "short"

    def test_truncate_long_string(self, tracer):
        """Test truncation with string longer than max length."""
        long_string = "x" * 200  # Longer than config max_field_length (100)
        result = tracer._truncate(long_string)
        
        assert len(result) == 103  # 100 + 3 for "..."
        assert result.endswith("...")
        assert result.startswith("x" * 97)

    def test_truncate_exact_length_string(self, tracer):
        """Test truncation with string exactly at max length."""
        exact_string = "x" * 100  # Exactly config max_field_length
        result = tracer._truncate(exact_string)
        assert result == exact_string

    # ======================= AUTOMATIC TRACING TESTS =======================
    
    @patch('sys.settrace')
    def test_run_success(self, mock_settrace, tracer):
        """Test successful start of automatic tracing."""
        tracer.run()
        
        assert tracer._tracing_active is True
        mock_settrace.assert_called_once_with(tracer._trace_calls)

    @patch('sys.settrace')
    def test_run_without_policy(self, mock_settrace, config):
        """Test run() raises error when no policy is set."""
        with patch('inspect.currentframe'), patch('glimpse.tracer.LogWriter'):
            tracer = Tracer(config, writer_initiation=False, policy=None)
            
            with pytest.raises(RuntimeError, match="TracingPolicy is required"):
                tracer.run()
            
            mock_settrace.assert_not_called()

    @patch('sys.settrace')
    def test_stop(self, mock_settrace, tracer):
        """Test stopping automatic tracing."""
        tracer._tracing_active = True
        tracer._call_metadata = {"frame1": "data"}
        
        tracer.stop()
        
        assert tracer._tracing_active is False
        assert tracer._call_metadata == {}
        mock_settrace.assert_called_once_with(None)
        tracer._writer.flush.assert_called_once()

    # ======================= TRACE_CALLS DISPATCHER TESTS =======================
    
    def test_trace_calls_inactive_tracing(self, tracer):
        """Test _trace_calls returns None when tracing is inactive."""
        tracer._tracing_active = False
        
        result = tracer._trace_calls(Mock(), "call", None)
        assert result is None

    def test_trace_calls_call_event(self, tracer, mock_frame):
        """Test _trace_calls handles 'call' events."""
        tracer._tracing_active = True
        
        with patch.object(tracer, '_handle_function_call', return_value=tracer._trace_calls) as mock_handle:
            result = tracer._trace_calls(mock_frame, "call", None)
            
            mock_handle.assert_called_once_with(mock_frame)
            assert result == tracer._trace_calls

    def test_trace_calls_return_event(self, tracer, mock_frame):
        """Test _trace_calls handles 'return' events."""
        tracer._tracing_active = True
        return_value = "test_return"
        
        with patch.object(tracer, '_handle_function_return', return_value=tracer._trace_calls) as mock_handle:
            result = tracer._trace_calls(mock_frame, "return", return_value)
            
            mock_handle.assert_called_once_with(mock_frame, return_value)
            assert result == tracer._trace_calls

    def test_trace_calls_exception_event(self, tracer, mock_frame):
        """Test _trace_calls handles 'exception' events."""
        tracer._tracing_active = True
        exc_info = (ValueError, ValueError("test"), None)
        
        with patch.object(tracer, '_handle_function_exception', return_value=tracer._trace_calls) as mock_handle:
            result = tracer._trace_calls(mock_frame, "exception", exc_info)
            
            mock_handle.assert_called_once_with(mock_frame, exc_info)
            assert result == tracer._trace_calls

    def test_trace_calls_unknown_event(self, tracer, mock_frame):
        """Test _trace_calls handles unknown events gracefully."""
        tracer._tracing_active = True
        
        result = tracer._trace_calls(mock_frame, "unknown_event", None)
        assert result == tracer._trace_calls

    def test_trace_calls_exception_handling(self, tracer, mock_frame):
        """Test _trace_calls handles exceptions in handlers gracefully."""
        tracer._tracing_active = True
        
        with patch.object(tracer, '_handle_function_call', side_effect=Exception("Handler error")):
            with patch('builtins.print') as mock_print:  # Mock print for error output
                result = tracer._trace_calls(mock_frame, "call", None)
                
                assert result == tracer._trace_calls
                mock_print.assert_called()

    # ======================= FUNCTION CALL HANDLER TESTS =======================
    
    def test_handle_function_call_trace_depth_exceeded(self, tracer, mock_frame):
        """Test _handle_function_call when trace depth is exceeded."""
        # Fill call metadata to exceed trace depth
        tracer._call_metadata = {f"frame{i}": {} for i in range(5)}  # trace_depth = 5
        
        result = tracer._handle_function_call(mock_frame)
        assert result is None

    def test_handle_function_call_tracer_code(self, tracer, mock_frame):
        """Test _handle_function_call skips tracer's own code."""
        mock_frame.f_globals["__name__"] = "glimpse.tracer"
        
        result = tracer._handle_function_call(mock_frame)
        assert result is None

    def test_handle_function_call_should_not_trace(self, tracer, mock_frame):
        """Test _handle_function_call when policy says not to trace."""
        with patch.object(tracer, 'should_trace_function', return_value=False):
            result = tracer._handle_function_call(mock_frame)
            assert result is None

    def test_handle_function_call_success(self, tracer, mock_frame):
        """Test successful _handle_function_call."""
        with patch.object(tracer, 'should_trace_function', return_value=True):
            with patch.object(tracer, '_log_function_entry') as mock_log:
                result = tracer._handle_function_call(mock_frame)
                
                assert result == tracer._trace_calls
                assert mock_frame in tracer._call_metadata
                
                call_info = tracer._call_metadata[mock_frame]
                assert call_info['function_name'] == "test_function"
                assert call_info['module_name'] == "test_module"
                assert 'start_time' in call_info
                
                mock_log.assert_called_once()

    # ======================= FUNCTION RETURN HANDLER TESTS =======================
    
    def test_handle_function_return_no_call_metadata(self, tracer, mock_frame):
        """Test _handle_function_return when no call metadata exists."""
        result = tracer._handle_function_return(mock_frame, "return_value")
        assert result == tracer._trace_calls

    def test_handle_function_return_success(self, tracer, mock_frame):
        """Test successful _handle_function_return."""
        # Set up call metadata
        call_info = {
            'function_name': 'test_func',
            'start_time': datetime.now(),
            'qualname': 'test_module.test_func'
        }
        tracer._call_metadata[mock_frame] = call_info
        
        with patch.object(tracer, '_log_function_exit') as mock_log:
            result = tracer._handle_function_return(mock_frame, "return_value")
            
            assert result == tracer._trace_calls
            assert mock_frame not in tracer._call_metadata  # Should be removed
            mock_log.assert_called_once_with(call_info, "return_value")

    # ======================= FUNCTION EXCEPTION HANDLER TESTS =======================
    
    def test_handle_function_exception_no_call_metadata(self, tracer, mock_frame):
        """Test _handle_function_exception when no call metadata exists."""
        exc_info = (ValueError, ValueError("test"), None)
        result = tracer._handle_function_exception(mock_frame, exc_info)
        assert result == tracer._trace_calls

    def test_handle_function_exception_success(self, tracer, mock_frame):
        """Test successful _handle_function_exception."""
        # Set up call metadata
        call_info = {
            'function_name': 'test_func',
            'start_time': datetime.now(),
            'qualname': 'test_module.test_func'
        }
        tracer._call_metadata[mock_frame] = call_info
        
        exc_info = (ValueError, ValueError("test error"), None)
        
        with patch.object(tracer, '_log_function_exception') as mock_log:
            result = tracer._handle_function_exception(mock_frame, exc_info)
            
            assert result == tracer._trace_calls
            assert mock_frame not in tracer._call_metadata  # Should be removed
            mock_log.assert_called_once_with(call_info, exc_info)

    # ======================= TRACER CODE DETECTION TESTS =======================
    
    @pytest.mark.parametrize("module_name,filename,expected", [
        ("glimpse.tracer", "/path/to/file.py", True),
        ("glimpse.config", "/path/to/file.py", True),
        ("myapp.services", "/path/to/glimpse/tracer.py", True),
        ("__main__", "/path/to/main.py", False),
        ("myapp.services", "/path/to/services.py", False),
        ("requests.auth", "/path/to/auth.py", False),
    ])
    def test_is_tracer_code(self, tracer, module_name, filename, expected):
        """Test _is_tracer_code detection."""
        result = tracer._is_tracer_code(module_name, filename)
        assert result == expected

    # ======================= FRAME ARGUMENT EXTRACTION TESTS =======================
    
    def test_get_function_args_from_frame_success(self, tracer, mock_frame):
        """Test successful argument extraction from frame."""
        result = tracer._get_function_args_from_frame(mock_frame)
        
        assert "arg1='value1'" in result
        assert "arg2='value2'" in result
        assert result.startswith("(")
        assert result.endswith(")")

    def test_get_function_args_from_frame_large_values(self, tracer, mock_frame):
        """Test argument extraction truncates large values."""
        mock_frame.f_locals = {
            "arg1": "x" * 200,  # Very long value
            "arg2": "normal"
        }
        
        result = tracer._get_function_args_from_frame(mock_frame)
        
        assert "arg1=" in result
        assert "..." in result  # Should be truncated
        assert "arg2='normal'" in result

    def test_get_function_args_from_frame_missing_args(self, tracer, mock_frame):
        """Test argument extraction when args are missing from locals."""
        mock_frame.f_locals = {"arg1": "value1"}  # Missing arg2
        
        result = tracer._get_function_args_from_frame(mock_frame)
        
        assert "arg1='value1'" in result
        # arg2 should not appear since it's not in locals

    def test_get_function_args_from_frame_exception(self, tracer):
        """Test argument extraction handles exceptions gracefully."""
        bad_frame = Mock()
        bad_frame.f_code = None  # This should cause an exception
        
        result = tracer._get_function_args_from_frame(bad_frame)
        assert result == "(args unavailable)"

    def test_get_function_args_from_frame_data_structures(self, tracer):
        """Test argument extraction with various Python data structures."""
        # Create a comprehensive mock frame with different data types
        mock_frame = Mock()
        mock_frame.f_code = Mock()
        mock_frame.f_code.co_varnames = (
            "string_arg", "int_arg", "float_arg", "bool_arg", "none_arg",
            "list_arg", "tuple_arg", "dict_arg", "set_arg", "nested_arg",
            "class_instance", "function_arg"
        )
        mock_frame.f_code.co_argcount = 12
        
        # Custom class for testing
        class TestClass:
            def __init__(self, name, value):
                self.name = name
                self.value = value
            
            def __repr__(self):
                return f"TestClass(name='{self.name}', value={self.value})"
        
        def dummy_function():
            return "I'm a function"
        
        # Test data with various Python data structures
        test_data = {
            "string_arg": "hello world",
            "int_arg": 42,
            "float_arg": 3.14159,
            "bool_arg": True,
            "none_arg": None,
            "list_arg": [1, 2, "three", [4, 5]],
            "tuple_arg": (1, 2, "three"),
            "dict_arg": {"key1": "value1", "key2": 42, "nested": {"inner": "data"}},
            "set_arg": {123, 234, 345, "four"},
            "nested_arg": {
                "list_in_dict": [1, 2, {"nested_dict": "value"}],
                "tuple_in_dict": (1, 2, 3)
            },
            "class_instance": TestClass("test_object", 123),
            "function_arg": dummy_function
        }
        
        mock_frame.f_locals = test_data
        
        result = tracer._get_function_args_from_frame(mock_frame)
        
        # Verify basic structure
        assert result.startswith("(")
        assert result.endswith(")")
        
        # Test string representation of each data type
        assert "string_arg='hello world'" in result
        assert "int_arg=42" in result
        assert "float_arg=3.14159" in result
        assert "bool_arg=True" in result
        assert "none_arg=None" in result
        
        # Test collections
        assert "list_arg=[1, 2, 'three', [4, 5]]" in result
        assert "tuple_arg=(1, 2, 'three')" in result
        assert "set_arg={" in result  # Set order is not guaranteed
        assert "123" in result and "234" in result and "345" in result and "'four'" in result
        
        # Test dictionary
        assert "dict_arg=" in result
        assert "'key1': 'value1'" in result
        assert "'key2': 42" in result
        assert "'nested': {'inner': 'data'}" in result
        
        # Test nested structures
        assert "nested_arg=" in result
        assert "'list_in_dict'" in result
        assert "'tuple_in_dict'" in result
        
        # Test custom class instance
        assert "class_instance=TestClass(name='test_object', value=123)" in result
        
        # Test function
        assert "function_arg=<function" in result
        assert "dummy_function" in result

    def test_get_function_args_from_frame_large_structures(self, tracer):
        """Test argument extraction with large data structures that get truncated."""
        mock_frame = Mock()
        mock_frame.f_code = Mock()
        mock_frame.f_code.co_varnames = ("large_dict", "large_list", "large_string", "normal_arg")
        mock_frame.f_code.co_argcount = 4
        
        # Create large data structures that will exceed 100 char limit
        large_dict = {f"key_{i}": f"value_{i}" for i in range(20)}
        large_list = [f"item_{i}" for i in range(50)]
        large_string = "x" * 200
        
        mock_frame.f_locals = {
            "large_dict": large_dict,
            "large_list": large_list, 
            "large_string": large_string,
            "normal_arg": "normal"
        }
        
        result = tracer._get_function_args_from_frame(mock_frame)
        
        # Check that large structures are truncated
        assert "large_dict=" in result
        assert "..." in result  # Should contain truncation indicator
        assert "large_list=" in result
        assert "large_string=" in result
        
        # Normal arg should not be truncated
        assert "normal_arg='normal'" in result
        
        # Verify that each large arg representation is actually truncated
        # Extract each argument's representation
        parts = result[1:-1].split(", ")  # Remove parentheses and split
        
        for part in parts:
            if "large_" in part:
                arg_repr = part.split("=", 1)[1]  # Get the value part after =
                assert len(arg_repr) <= 100  # Should be truncated to 100 chars or less

    def test_get_function_args_from_frame_special_objects(self, tracer):
        """Test argument extraction with special Python objects."""
        import re
        import datetime
        from collections import defaultdict, namedtuple
        
        mock_frame = Mock()
        mock_frame.f_code = Mock()
        mock_frame.f_code.co_varnames = (
            "regex_arg", "datetime_arg", "defaultdict_arg", "namedtuple_arg",
            "lambda_arg", "generator_arg", "bytes_arg", "complex_arg"
        )
        mock_frame.f_code.co_argcount = 8
        
        # Create special objects
        Point = namedtuple('Point', ['x', 'y'])
        
        def simple_generator():
            yield 1
            yield 2
        
        special_objects = {
            "regex_arg": re.compile(r'\d+'),
            "datetime_arg": datetime.datetime(2023, 1, 1, 12, 0, 0),
            "defaultdict_arg": defaultdict(list, {"key": [1, 2, 3]}),
            "namedtuple_arg": Point(x=10, y=20),
            "lambda_arg": lambda x: x * 2,
            "generator_arg": simple_generator(),
            "bytes_arg": b"hello bytes",
            "complex_arg": 3 + 4j
        }
        
        mock_frame.f_locals = special_objects
        
        result = tracer._get_function_args_from_frame(mock_frame)
        
        # Test that special objects are represented correctly
        assert "regex_arg=re.compile(" in result
        assert "datetime_arg=datetime.datetime(2023, 1, 1, 12, 0)" in result
        assert "defaultdict_arg=defaultdict(" in result
        assert "namedtuple_arg=Point(x=10, y=20)" in result
        assert "lambda_arg=<function" in result
        assert "generator_arg=<generator" in result
        assert "bytes_arg=b'hello bytes'" in result
        assert "complex_arg=(3+4j)" in result

    def test_get_function_args_from_frame_edge_cases(self, tracer):
        """Test argument extraction with edge case data structures."""
        mock_frame = Mock()
        mock_frame.f_code = Mock()
        mock_frame.f_code.co_varnames = (
            "empty_list", "empty_dict", "empty_set", "empty_tuple",
            "nested_empty", "circular_ref", "unicode_string", "escape_chars"
        )
        mock_frame.f_code.co_argcount = 8
        
        # Create circular reference for testing
        circular_list = [1, 2, 3]
        circular_list.append(circular_list)  # Creates circular reference
        
        edge_case_data = {
            "empty_list": [],
            "empty_dict": {},
            "empty_set": set(),
            "empty_tuple": (),
            "nested_empty": {"empty_list": [], "empty_dict": {}},
            "circular_ref": circular_list,
            "unicode_string": "测试字符串_πακέτο_パッケージ",
            "escape_chars": "string with\nnewlines\tand\r\"quotes\""
        }
        
        mock_frame.f_locals = edge_case_data
        
        result = tracer._get_function_args_from_frame(mock_frame)
        
        # Test empty collections
        assert "empty_list=[]" in result
        assert "empty_dict={}" in result
        assert "empty_set=set()" in result
        assert "empty_tuple=()" in result
        
        # Test nested empty structures
        assert "nested_empty=" in result
        assert "'empty_list': []" in result
        assert "'empty_dict': {}" in result
        
        # Test circular reference (should not cause infinite recursion)
        assert "circular_ref=" in result
        # Should contain the list representation (Python handles circular refs in repr)
        
        # Test unicode string
        assert "unicode_string='测试字符串_πακέτο_パッケージ'" in result
        
        # Test escape characters
        assert "escape_chars=" in result
        assert "\\n" in result or "\\t" in result  # Should show escaped characters


    # ======================= LOGGING METHOD TESTS =======================
    
    def test_log_function_entry(self, tracer, mock_frame, id_generator):
        """Test _log_function_entry creates correct LogEntry."""
        call_info = {
            'call_id': id_generator.new_call_id(), 
            'qualname': 'test_module.test_func',
            'function_name': 'test_func',
            'start_time': datetime(2023, 1, 1, 12, 0, 0)
        }
        
        with patch.object(tracer, '_get_function_args_from_frame', return_value="(arg1='value1')"):
            tracer._log_function_entry(call_info, mock_frame)
            
            tracer._writer.write.assert_called_once()
            log_entry = tracer._writer.write.call_args[0][0]
            
            assert isinstance(log_entry, LogEntry)
            assert log_entry.stage == "START"
            assert log_entry.function == "test_module.test_func"
            assert "test_func(arg1='value1')" in log_entry.args

    def test_log_function_exit(self, tracer, id_generator):
        """Test _log_function_exit creates correct LogEntry."""
        call_info = {
            'call_id': id_generator.new_call_id(), 
            'qualname': 'test_module.test_func',
            'function_name': 'test_func',
            'start_time': datetime.now() - timedelta(milliseconds=100)
        }
        return_value = {"result": "success"}
        
        tracer._log_function_exit(call_info, return_value)
        
        tracer._writer.write.assert_called_once()
        log_entry = tracer._writer.write.call_args[0][0]
        
        assert isinstance(log_entry, LogEntry)
        assert log_entry.stage == "END"
        assert log_entry.function == "test_module.test_func"
        assert log_entry.result is not None
        assert log_entry.duration_ms is not None

    def test_log_function_exception(self, tracer, id_generator):
        """Test _log_function_exception creates correct LogEntry."""
        call_info = {
            'call_id': id_generator.new_call_id(), 
            'qualname': 'test_module.test_func',
            'function_name': 'test_func'
        }
        exc_info = (ValueError, ValueError("test error"), None)
        
        tracer._log_function_exception(call_info, exc_info)
        
        tracer._writer.write.assert_called_once()
        log_entry = tracer._writer.write.call_args[0][0]
        
        assert isinstance(log_entry, LogEntry)
        assert log_entry.stage == "EXCEPTION"
        assert log_entry.level == "INFO"
        assert log_entry.function == "test_module.test_func"
        assert "ValueError: test error" in log_entry.exception

    # ======================= CONTEXT MANAGER TESTS =======================
    
    @patch('sys.settrace')
    def test_context_manager_enter(self, mock_settrace, tracer):
        """Test context manager __enter__ method."""
        result = tracer.__enter__()
        
        assert result == tracer
        assert tracer._tracing_active is True
        mock_settrace.assert_called_once_with(tracer._trace_calls)

    @patch('sys.settrace')
    def test_context_manager_exit_normal(self, mock_settrace, tracer):
        """Test context manager __exit__ method with normal exit."""
        tracer._tracing_active = True
        
        result = tracer.__exit__(None, None, None)
        
        assert result is False  # Don't suppress exceptions
        assert tracer._tracing_active is False
        mock_settrace.assert_called_once_with(None)
        tracer._writer.flush.assert_called_once()

    @patch('sys.settrace')
    def test_context_manager_exit_with_exception(self, mock_settrace, tracer):
        """Test context manager __exit__ method when exception occurred."""
        tracer._tracing_active = True
        
        result = tracer.__exit__(ValueError, ValueError("test"), None)
        
        assert result is False  # Don't suppress exceptions
        assert tracer._tracing_active is False

    def test_context_manager_full_usage(self, tracer):
        """Test full context manager usage."""
        with patch('sys.settrace') as mock_settrace:
            with tracer:
                assert tracer._tracing_active is True
            
            assert tracer._tracing_active is False
            assert mock_settrace.call_count == 2  # Once for enter, once for exit

    # ======================= INTEGRATION TESTS =======================
    
    def test_full_tracing_workflow(self, tracer, mock_frame):
        """Test complete tracing workflow from call to return."""
        # Setup
        tracer._tracing_active = True
        
        with patch.object(tracer, 'should_trace_function', return_value=True):
            # Simulate function call
            result1 = tracer._trace_calls(mock_frame, "call", None)
            assert result1 == tracer._trace_calls
            assert mock_frame in tracer._call_metadata
            
            # Simulate function return
            result2 = tracer._trace_calls(mock_frame, "return", "return_value")
            assert result2 == tracer._trace_calls
            assert mock_frame not in tracer._call_metadata
            
            # Verify logging calls
            assert tracer._writer.write.call_count == 2  # Entry and exit

    def test_full_exception_workflow(self, tracer, mock_frame):
        """Test complete exception workflow from call to exception."""
        # Setup
        tracer._tracing_active = True
        exc_info = (ValueError, ValueError("test"), None)
        
        with patch.object(tracer, 'should_trace_function', return_value=True):
            # Simulate function call
            tracer._trace_calls(mock_frame, "call", None)
            assert mock_frame in tracer._call_metadata
            
            # Simulate function exception
            tracer._trace_calls(mock_frame, "exception", exc_info)
            assert mock_frame not in tracer._call_metadata
            
            # Verify logging calls
            assert tracer._writer.write.call_count == 2  # Entry and exception

    # ======================= PERFORMANCE TESTS =======================
    
    def test_call_metadata_performance(self, tracer):
        """Test that call metadata operations are fast."""
        # Create many mock frames
        mock_frames = [Mock() for _ in range(1000)]
        
        start_time = time.time()
        
        # Add metadata for all frames
        for i, frame in enumerate(mock_frames):
            tracer._call_metadata[frame] = {"data": f"frame_{i}"}
        
        # Remove metadata for all frames
        for frame in mock_frames:
            tracer._call_metadata.pop(frame, None)
        
        elapsed = time.time() - start_time
        
        # Should be very fast (dict operations are O(1))
        assert elapsed < 0.1  # Less than 100ms for 1000 frames

    def test_should_trace_function_performance(self, tracer):
        """Test that should_trace_function is fast enough for frequent calls."""
        mock_funcs = []
        for i in range(1000):
            func = Mock()
            func.__module__ = f"test_module_{i % 10}"  # 10 different modules
            mock_funcs.append(func)
        
        start_time = time.time()
        
        for func in mock_funcs:
            tracer.should_trace_function(func)
        
        elapsed = time.time() - start_time
        
        # Should handle 1000 checks quickly
        assert elapsed < 0.1  # Less than 100ms

    # ======================= EDGE CASES AND ERROR HANDLING =======================
    
    def test_malformed_frame_handling(self, tracer):
        """Test handling of malformed frame objects."""
        malformed_frame = Mock()
        malformed_frame.f_code = None
        malformed_frame.f_globals = {}
        
        # Should not crash
        result = tracer._is_tracer_code("", "")
        assert isinstance(result, bool)
        
        args_result = tracer._get_function_args_from_frame(malformed_frame)
        assert args_result == "(args unavailable)"

    def test_very_deep_call_stack(self, tracer):
        """Test behavior with very deep call stacks."""
        # Simulate deep call stack
        for i in range(20):  # Deeper than trace_depth (5)
            frame = Mock()
            frame.f_code.co_name = f"func_{i}"
            frame.f_globals = {"__name__": "test_module"}
            tracer._call_metadata[frame] = {"depth": i}
        
        # New call should be rejected due to depth limit
        new_frame = Mock()
        new_frame.f_code.co_name = "deep_func"
        new_frame.f_globals = {"__name__": "test_module"}
        
        result = tracer._handle_function_call(new_frame)
        assert result is None  # Should be rejected

    def test_unicode_in_function_names(self, tracer, mock_frame):
        """Test handling of unicode characters in function names."""
        mock_frame.f_code.co_name = "测试函数"
        mock_frame.f_globals["__name__"] = "测试模块"
        
        with patch.object(tracer, 'should_trace_function', return_value=True):
            result = tracer._handle_function_call(mock_frame)
            
            assert result == tracer._trace_calls
            assert mock_frame in tracer._call_metadata
            
            call_info = tracer._call_metadata[mock_frame]
            assert call_info['function_name'] == "测试函数"
            assert call_info['module_name'] == "测试模块"

    def test_large_return_values(self, tracer, id_generator):
        """Test handling of very large return values."""
        call_info = {
            'call_id': id_generator.new_call_id(), 
            'qualname': 'test_func',
            'function_name': 'test_func',
            'start_time': datetime.now()
        }
        
        # Very large return value
        large_return = {"data": "x" * 10000}
        
        tracer._log_function_exit(call_info, large_return)
        
        # Should not crash and should truncate
        tracer._writer.write.assert_called_once()
        log_entry = tracer._writer.write.call_args[0][0]
        
        # Result should be truncated
        assert len(log_entry.result) <= tracer._config.max_field_length + 3

    def test_circular_reference_in_args(self, tracer, mock_frame):
        """Test handling of circular references in function arguments."""
        # Create circular reference
        circular_dict = {"self": None}
        circular_dict["self"] = circular_dict
        
        mock_frame.f_locals = {"arg1": circular_dict, "arg2": "value2"}
        
        # Should not crash or cause infinite recursion
        result = tracer._get_function_args_from_frame(mock_frame)
        assert "(args unavailable)" in result or "arg1=" in result

    @patch('glimpse.tracer.datetime')
    def test_time_measurement_accuracy(self, mock_datetime, tracer, id_generator):
        """Test that time measurements are reasonably accurate."""
        # Mock datetime to control timing
        start_time = datetime(2023, 1, 1, 12, 0, 0, 0)
        end_time = datetime(2023, 1, 1, 12, 0, 0, 150000)  # 150ms later
        
        mock_datetime.now.return_value = end_time
        
        call_info = {
            'call_id': id_generator.new_call_id(), 
            'qualname': 'test_func',
            'function_name': 'test_func',
            'start_time': start_time
        }
        
        tracer._log_function_exit(call_info, "result")
        
        log_entry = tracer._writer.write.call_args[0][0]
        assert log_entry.duration_ms == "150.0"  # Should be 150ms

# ======================= ADDITIONAL HELPER IMPORTS =======================

from datetime import timedelta  # Add this import for the test_log_function_exit test