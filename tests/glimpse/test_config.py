import pytest
import os
from unittest.mock import patch, MagicMock
from glimpse.config import Config

class TestConfig:
    """Comprehensive unit tests for Config class with multiple destinations support."""
    
    # ======================= INITIALIZATION TESTS =======================
    
    def test_init_with_defaults(self):
        """Test Config initialization with default values."""
        with patch.dict(os.environ, {}, clear=True):  # Clear environment
            config = Config(env_override=False)  # Disable env loading for pure default test
            
            assert config.dest == ["jsonl"]  # Now returns list
            assert config.level == "INFO"
            assert config.enable_trace_id is False
            assert config.params == {}
            assert config.env_prefix == "GLIMPSE_"
            assert config.max_field_length == 512

    def test_init_with_single_destination_string(self):
        """Test Config initialization with single destination as string."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(dest="json", env_override=False)
            
            assert config.dest == ["json"]  # Converted to list
            assert isinstance(config.dest, list)

    def test_init_with_multiple_destinations_list(self):
        """Test Config initialization with multiple destinations as list."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(dest=["json", "jsonl", "sqllite"], env_override=False)
            
            assert config.dest == ["json", "jsonl", "sqllite"]
            assert len(config.dest) == 3

    def test_init_with_custom_values(self):
        """Test Config initialization with custom values."""
        custom_params = {"log_path": "/custom/path", "buffer_size": "1024"}
        
        with patch.dict(os.environ, {}, clear=True):
            config = Config(
                dest=["json", "sqllite"],
                level="debug",
                enable_trace_id=True,
                params=custom_params,
                env_override=False,
                env_prefix="CUSTOM_",
                max_field_length=256
            )
            
            assert config.dest == ["json", "sqllite"]
            assert config.level == "DEBUG"  # Should be uppercased
            assert config.enable_trace_id is True
            assert config.params == custom_params
            assert config.env_prefix == "CUSTOM_"
            assert config.max_field_length == 256

    def test_init_filters_invalid_destinations_from_list(self):
        """Test that invalid destinations are filtered out from list."""
        with patch.dict(os.environ, {}, clear=True):
            # Mix of valid and invalid destinations
            config = Config(dest=["json", "invalid", "jsonl", "bad_dest", "sqllite"], env_override=False)
            
            # Should only keep valid destinations
            assert config.dest == ["json", "jsonl", "sqllite"]
            assert "invalid" not in config.dest
            assert "bad_dest" not in config.dest

    @pytest.mark.parametrize("invalid_dest", [
        "invalid", "txt", "csv", "mysql", "", "JSON", "JSONL"  # Case sensitive
    ])
    def test_init_invalid_single_destination(self, invalid_dest):
        """Test Config raises ValueError for invalid single destination."""
        with pytest.raises(ValueError, match=f"Invalid destination"):
            Config(dest=invalid_dest, env_override=False)

    def test_init_all_invalid_destinations_in_list(self):
        """Test Config raises ValueError when all destinations in list are invalid."""
        with pytest.raises(ValueError, match="Invalid destination"):
            Config(dest=["invalid", "bad", "wrong"], env_override=False)

    def test_init_empty_destination_list(self):
        """Test Config raises ValueError for empty destination list."""
        with pytest.raises(ValueError, match="Invalid destination"):
            Config(dest=[], env_override=False)

    @pytest.mark.parametrize("valid_dest", [
        "json", "jsonl", "sqllite", "mongo"
    ])
    def test_init_valid_single_destinations(self, valid_dest):
        """Test Config accepts all valid single destinations."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(dest=valid_dest, env_override=False)
            assert config.dest == [valid_dest]

    def test_init_valid_multiple_destinations(self):
        """Test Config accepts multiple valid destinations."""
        valid_combinations = [
            ["json", "jsonl"],
            ["jsonl", "sqllite"],
            ["json", "sqllite", "mongo"],
            ["json", "jsonl", "sqllite", "mongo"]
        ]
        
        for dest_list in valid_combinations:
            with patch.dict(os.environ, {}, clear=True):
                config = Config(dest=dest_list, env_override=False)
                assert config.dest == dest_list

    def test_init_duplicate_destinations_in_list(self):
        """Test handling of duplicate destinations in list."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(dest=["json", "jsonl", "json", "sqllite", "jsonl"], env_override=False)
            
            # Should preserve order but remove duplicates might be expected behavior
            # Based on current implementation, duplicates would remain
            assert "json" in config.dest
            assert "jsonl" in config.dest
            assert "sqllite" in config.dest

    def test_init_params_none_handling(self):
        """Test Config handles None params correctly."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(params=None, env_override=False)
            assert config.params == {}
            assert isinstance(config.params, dict)

    def test_init_level_case_handling(self):
        """Test Config converts level to uppercase."""
        test_cases = [
            ("info", "INFO"),
            ("debug", "DEBUG"),
            ("warning", "WARNING"),
            ("error", "ERROR"),
            ("INFO", "INFO"),
            ("Debug", "DEBUG"),
            ("WaRnInG", "WARNING"),
        ]
        
        for input_level, expected_level in test_cases:
            with patch.dict(os.environ, {}, clear=True):
                config = Config(level=input_level, env_override=False)
                assert config.level == expected_level

    def test_init_env_prefix_case_handling(self):
        """Test Config converts env_prefix to uppercase."""
        test_cases = [
            ("glimpse_", "GLIMPSE_"),
            ("GLIMPSE_", "GLIMPSE_"),
            ("MyApp_", "MYAPP_"),
            ("test", "TEST"),
        ]
        
        for input_prefix, expected_prefix in test_cases:
            with patch.dict(os.environ, {}, clear=True):
                config = Config(env_prefix=input_prefix, env_override=False)
                assert config.env_prefix == expected_prefix

    # ======================= ENVIRONMENT LOADING TESTS =======================
    
    def test_load_from_env_single_destination(self):
        """Test loading single destination from environment variables."""
        env_vars = {
            "GLIMPSE_DEST": "json",
            "GLIMPSE_LEVEL": "debug",
            "GLIMPSE_TRACE_ID": "true"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()  # env_override=True by default
            
            assert config.dest == ["json"]
            assert config.level == "DEBUG"  # Uppercased
            assert config.enable_trace_id is True

    def test_load_from_env_multiple_destinations(self):
        """Test loading multiple destinations from environment variables."""
        env_vars = {
            "GLIMPSE_DEST": "json,jsonl,sqllite",
            "GLIMPSE_LEVEL": "debug",
            "GLIMPSE_TRACE_ID": "true"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()
            
            assert config.dest == ["json", "jsonl", "sqllite"]
            assert len(config.dest) == 3

    def test_load_from_env_destinations_with_spaces(self):
        """Test loading destinations with spaces around commas."""
        env_vars = {
            "GLIMPSE_DEST": " json , jsonl , sqllite ",
            "GLIMPSE_LEVEL": "info"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()
            
            # Should handle spaces correctly
            assert config.dest == ["json", "jsonl", "sqllite"]

    def test_load_from_env_destinations_with_invalid_mixed(self):
        """Test loading destinations with mix of valid and invalid from environment."""
        env_vars = {
            "GLIMPSE_DEST": "json,invalid,jsonl,bad,sqllite"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()
            
            # Should filter out invalid destinations
            assert config.dest == ["json", "jsonl", "sqllite"]
            assert "invalid" not in config.dest
            assert "bad" not in config.dest

    def test_load_from_env_all_invalid_destinations(self):
        """Test error when all destinations from environment are invalid."""
        env_vars = {
            "GLIMPSE_DEST": "invalid,bad,wrong"
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValueError, match="Invalid destination"):
                Config()

    def test_load_from_env_trace_id_variations(self):
        """Test various trace_id values from environment."""
        true_values = ["1", "true", "yes", "True", "YES", "TRUE"]
        false_values = ["0", "false", "no", "False", "NO", "other", "", "maybe"]
        
        for true_val in true_values:
            with patch.dict(os.environ, {"GLIMPSE_TRACE_ID": true_val, "GLIMPSE_DEST": "jsonl"}, clear=True):
                config = Config()
                assert config.enable_trace_id is True, f"'{true_val}' should be True"
        
        for false_val in false_values:
            with patch.dict(os.environ, {"GLIMPSE_TRACE_ID": false_val, "GLIMPSE_DEST": "jsonl"}, clear=True):
                config = Config()
                assert config.enable_trace_id is False, f"'{false_val}' should be False"

    def test_load_from_env_custom_params(self):
        """Test loading custom parameters from environment variables."""
        env_vars = {
            "GLIMPSE_LOG_PATH": "/custom/logs",
            "GLIMPSE_BUFFER_SIZE": "2048",
            "GLIMPSE_TIMEOUT": "30",
            "GLIMPSE_DEST": "json",  # Core key - should not be in params
            "GLIMPSE_LEVEL": "DEBUG",  # Core key - should not be in params
            "GLIMPSE_TRACE_ID": "true",  # Core key - should not be in params
            "OTHER_VAR": "ignore",  # Different prefix - should be ignored
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()
            
            expected_params = {
                "log_path": "/custom/logs",
                "buffer_size": "2048", 
                "timeout": "30"
            }
            
            assert config.params == expected_params
            # Verify core keys are not in params
            assert "dest" not in config.params
            assert "level" not in config.params
            assert "trace_id" not in config.params

    def test_load_from_env_custom_prefix(self):
        """Test loading with custom environment prefix."""
        env_vars = {
            "MYAPP_DEST": "mongo",
            "MYAPP_LEVEL": "error",
            "MYAPP_CUSTOM_PARAM": "value",
            "GLIMPSE_DEST": "json",  # Wrong prefix - should be ignored
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config(env_prefix="MYAPP_")
            
            assert config.dest == ["mongo"]
            assert config.level == "ERROR"
            assert config.params == {"custom_param": "value"}

    def test_load_from_env_override_disabled(self):
        """Test that env_override=False prevents environment loading."""
        env_vars = {
            "GLIMPSE_DEST": "mongo,sqllite",
            "GLIMPSE_LEVEL": "error",
            "GLIMPSE_CUSTOM": "ignored",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config(
                dest="jsonl",
                level="info", 
                env_override=False
            )
            
            # Should use constructor values, not environment
            assert config.dest == ["jsonl"]
            assert config.level == "INFO"
            assert config.params == {}

    def test_load_from_env_partial_override(self):
        """Test environment variables override only provided values."""
        env_vars = {
            "GLIMPSE_DEST": "mongo,sqllite",
            # LEVEL not provided - should use constructor value
            "GLIMPSE_CUSTOM": "from_env",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config(level="debug")  # Constructor level
            
            assert config.dest == ["mongo", "sqllite"]  # From environment
            assert config.level == "DEBUG"  # From constructor (env didn't override)
            assert config.params == {"custom": "from_env"}

    def test_load_from_env_case_insensitive_params(self):
        """Test that parameter keys are converted to lowercase."""
        env_vars = {
            "GLIMPSE_DEST": "jsonl",
            "GLIMPSE_LOG_PATH": "value1",
            "GLIMPSE_BUFFER_SIZE": "value2",
            "GLIMPSE_Custom_Key": "value3",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()
            
            expected_params = {
                "log_path": "value1",
                "buffer_size": "value2",
                "custom_key": "value3",
            }
            
            assert config.params == expected_params

    # ======================= PROPERTY TESTS =======================
    
    def test_property_getters(self):
        """Test all property getters return correct values."""
        custom_params = {"key1": "value1", "key2": "value2"}
        
        with patch.dict(os.environ, {}, clear=True):
            config = Config(
                dest=["json", "sqllite"],
                level="warning",
                enable_trace_id=True,
                params=custom_params,
                env_prefix="TEST_",
                max_field_length=1024,
                env_override=False
            )
            
            assert config.dest == ["json", "sqllite"]
            assert config.level == "WARNING"
            assert config.enable_trace_id is True
            assert config.params == custom_params
            assert config.env_prefix == "TEST_"
            assert config.max_field_length == 1024

    def test_dest_property_returns_list(self):
        """Test that dest property always returns a list."""
        test_cases = [
            ("jsonl", ["jsonl"]),
            (["json"], ["json"]),
            (["json", "jsonl"], ["json", "jsonl"]),
        ]
        
        for input_dest, expected_dest in test_cases:
            with patch.dict(os.environ, {}, clear=True):
                config = Config(dest=input_dest, env_override=False)
                assert config.dest == expected_dest
                assert isinstance(config.dest, list)

    def test_params_returns_copy(self):
        """Test that params property returns a copy to prevent mutation."""
        original_params = {"key1": "value1", "key2": "value2"}
        
        with patch.dict(os.environ, {}, clear=True):
            config = Config(params=original_params, env_override=False)
            
            # Get params and modify the returned dict
            returned_params = config.params
            returned_params["key3"] = "value3"
            returned_params["key1"] = "modified"
            
            # Original config params should be unchanged
            assert config.params == original_params
            assert "key3" not in config.params
            assert config.params["key1"] == "value1"

    def test_max_field_length_fallback(self):
        """Test max_field_length property fallback behavior."""
        with patch.dict(os.environ, {}, clear=True):
            # Test with explicit value
            config1 = Config(max_field_length=256, env_override=False)
            assert config1.max_field_length == 256
            
            # Test with default (should be 512)
            config2 = Config(env_override=False)
            assert config2.max_field_length == 512
            
            # Test with None (should fallback to 512)
            config3 = Config(max_field_length=None, env_override=False)
            assert config3.max_field_length == 512

    # ======================= DESTINATION MANAGEMENT TESTS =======================
    
    def test_add_destination_method(self):
        """Test add_destination method."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(dest="jsonl", env_override=False)
            
            # Add valid destination
            result = config.add_destination("sqllite")
            assert result is True
            assert config.dest == ["jsonl", "sqllite"]
            
            # Add another destination
            config.add_destination("json")
            assert config.dest == ["jsonl", "sqllite", "json"]

    def test_add_destination_duplicate(self):
        """Test adding duplicate destination."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(dest=["jsonl", "sqllite"], env_override=False)
            
            # Add existing destination - behavior depends on implementation
            config.add_destination("jsonl")
            # Current implementation would add duplicate
            assert "jsonl" in config.dest

    def test_remove_destination_method(self):
        """Test remove_destination method."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(dest=["jsonl", "sqllite", "json"], env_override=False)
            
            # Remove by index
            result = config.remove_destination(1)  # Remove "sqllite"
            assert result is True
            assert config.dest == ["jsonl", "json"]
            
            # Remove another
            config.remove_destination(0)  # Remove "jsonl"
            assert config.dest == ["json"]

    def test_remove_destination_invalid_index(self):
        """Test removing destination with invalid index."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(dest=["jsonl"], env_override=False)
            
            # Try to remove invalid index
            with pytest.raises(IndexError):
                config.remove_destination(5)

    def test_remove_last_destination(self):
        """Test removing the last destination."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(dest=["jsonl"], env_override=False)
            
            # Remove the only destination
            config.remove_destination(0)
            assert config.dest == []
            
            # This might cause issues in the application, but testing current behavior

    # ======================= BUILD_ENV_VAR METHOD TESTS =======================
    
    def test_build_env_var(self):
        """Test build_env_var method constructs correct environment variable names."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(env_prefix="GLIMPSE_", env_override=False)
            
            assert config.build_env_var("DEST") == "GLIMPSE_DEST"
            assert config.build_env_var("LEVEL") == "GLIMPSE_LEVEL"
            assert config.build_env_var("TRACE_ID") == "GLIMPSE_TRACE_ID"
            assert config.build_env_var("CUSTOM") == "GLIMPSE_CUSTOM"

    def test_build_env_var_custom_prefix(self):
        """Test build_env_var with custom prefix."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(env_prefix="MYAPP_", env_override=False)
            
            assert config.build_env_var("DEST") == "MYAPP_DEST"
            assert config.build_env_var("CUSTOM_PARAM") == "MYAPP_CUSTOM_PARAM"

    def test_build_env_var_empty_suffix(self):
        """Test build_env_var with empty suffix."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(env_prefix="GLIMPSE_", env_override=False)
            
            assert config.build_env_var("") == "GLIMPSE_"

    # ======================= INTEGRATION TESTS =======================
    
    def test_constructor_and_environment_integration_multiple_dest(self):
        """Test integration between constructor parameters and environment variables with multiple destinations."""
        env_vars = {
            "GLIMPSE_DEST": "mongo,sqllite",  # Override constructor
            "GLIMPSE_CUSTOM_PARAM": "env_value",  # Additional param
        }
        
        constructor_params = {"existing_param": "constructor_value"}
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config(
                dest="jsonl",  # Should be overridden by env
                level="debug",  # Should remain from constructor
                params=constructor_params  # Should be merged with env params
            )
            
            assert config.dest == ["mongo", "sqllite"]  # From environment
            assert config.level == "DEBUG"  # From constructor
            
            expected_params = {
                "existing_param": "constructor_value",  # From constructor
                "custom_param": "env_value"  # From environment
            }
            assert config.params == expected_params

    def test_environment_precedence_over_constructor_multiple_dest(self):
        """Test that environment variables take precedence over constructor values for multiple destinations."""
        env_vars = {
            "GLIMPSE_DEST": "mongo,json",
            "GLIMPSE_LEVEL": "error",
            "GLIMPSE_TRACE_ID": "1",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config(
                dest=["jsonl", "sqllite"],  # Should be overridden
                level="info",  # Should be overridden
                enable_trace_id=False  # Should be overridden
            )
            
            assert config.dest == ["mongo", "json"]
            assert config.level == "ERROR"
            assert config.enable_trace_id is True

    # ======================= EDGE CASES AND ERROR HANDLING =======================
    
    def test_empty_string_destinations_env(self):
        """Test handling of empty string destination values from environment."""
        env_vars = {
            "GLIMPSE_DEST": "",  # Empty dest
            "GLIMPSE_LEVEL": "INFO",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            # Empty dest from environment should cause validation error
            with pytest.raises(ValueError, match="Invalid destination"):
                Config()

    def test_comma_only_destinations_env(self):
        """Test handling of comma-only destination values from environment."""
        env_vars = {
            "GLIMPSE_DEST": ",,,"  # Only commas
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValueError, match="Invalid destination"):
                Config()

    def test_mixed_empty_and_valid_destinations_env(self):
        """Test handling of mixed empty and valid destinations from environment."""
        env_vars = {
            "GLIMPSE_DEST": "json,,jsonl,"  # Mixed with empty strings
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()
            # Should filter out empty strings and keep valid ones
            assert config.dest == ["json", "jsonl"]

    def test_missing_environment_variables(self):
        """Test behavior when expected environment variables are missing."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config(dest=["jsonl", "sqllite"], level="info")
            
            # Should use constructor/default values
            assert config.dest == ["jsonl", "sqllite"]
            assert config.level == "INFO"
            assert config.enable_trace_id is False
            assert config.params == {}

    def test_very_long_environment_values(self):
        """Test handling of very long environment variable values."""
        long_value = "x" * 10000
        env_vars = {
            "GLIMPSE_DEST": "jsonl",
            "GLIMPSE_LONG_PARAM": long_value,
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()
            
            assert config.dest == ["jsonl"]
            assert config.params["long_param"] == long_value

    def test_special_characters_in_env_values(self):
        """Test handling of special characters in environment values."""
        special_values = {
            "GLIMPSE_DEST": "jsonl",
            "GLIMPSE_PATH": "/path/with spaces/and-symbols!@#$%",
            "GLIMPSE_JSON": '{"key": "value", "nested": {"array": [1,2,3]}}',
            "GLIMPSE_UNICODE": "测试值_пакет_パッケージ",
        }
        
        with patch.dict(os.environ, special_values, clear=True):
            config = Config()
            
            assert config.dest == ["jsonl"]
            assert config.params["path"] == "/path/with spaces/and-symbols!@#$%"
            assert config.params["json"] == '{"key": "value", "nested": {"array": [1,2,3]}}'
            assert config.params["unicode"] == "测试值_пакет_パッケージ"

    # ======================= CLASS CONSTANTS TESTS =======================
    
    def test_class_constants(self):
        """Test that class constants are correctly defined."""
        assert Config._CORE_KEYS == {"DEST", "LEVEL", "TRACE_ID"}
        assert Config._ACCEPTABLE_DEST == {"json", "jsonl", "sqllite", "mongo"}
        
        # Verify constants are used correctly for single destinations
        for dest in Config._ACCEPTABLE_DEST:
            with patch.dict(os.environ, {}, clear=True):
                config = Config(dest=dest, env_override=False)
                assert config.dest == [dest]

    def test_core_keys_not_in_params(self):
        """Test that core keys are never added to params."""
        env_vars = {
            "GLIMPSE_DEST": "json,jsonl",
            "GLIMPSE_LEVEL": "debug", 
            "GLIMPSE_TRACE_ID": "true",
            "GLIMPSE_CUSTOM": "should_be_in_params",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()
            
            # Core keys should not be in params
            for core_key in Config._CORE_KEYS:
                assert core_key.lower() not in config.params
            
            # Custom param should be in params
            assert "custom" in config.params

    # ======================= PERFORMANCE TESTS =======================
    
    def test_config_creation_performance_multiple_dest(self):
        """Test that Config creation is reasonably fast even with many env vars and multiple destinations."""
        # Create many environment variables
        many_env_vars = {f"GLIMPSE_PARAM_{i}": f"value_{i}" for i in range(100)}
        many_env_vars["GLIMPSE_DEST"] = "jsonl,sqllite,json"
        many_env_vars["GLIMPSE_LEVEL"] = "info"
        
        with patch.dict(os.environ, many_env_vars, clear=True):
            import time
            start = time.time()
            
            # Create multiple configs
            for _ in range(100):
                config = Config()
            
            elapsed = time.time() - start
            
            # Should complete quickly
            assert elapsed < 1.0  # Less than 1 second for 100 config creations
            
            # Verify last config loaded correctly
            assert len(config.params) == 100
            assert len(config.dest) == 3

    # ======================= BACKWARDS COMPATIBILITY TESTS =======================
    
    def test_backwards_compatibility_single_destination(self):
        """Test that existing single destination code still works."""
        # These should work exactly as before, just returning lists now
        with patch.dict(os.environ, {}, clear=True):
            config = Config(dest="jsonl", env_override=False)
            assert config.dest == ["jsonl"]
            
        with patch.dict(os.environ, {"GLIMPSE_DEST": "sqllite"}, clear=True):
            config = Config()
            assert config.dest == ["sqllite"]

    def test_backwards_compatibility_error_behavior(self):
        """Test that error conditions still behave the same way."""
        # Invalid single destination should still raise ValueError
        with pytest.raises(ValueError):
            Config(dest="invalid", env_override=False)
            
        # Invalid destination from environment should still raise ValueError
        with patch.dict(os.environ, {"GLIMPSE_DEST": "invalid"}, clear=True):
            with pytest.raises(ValueError):
                Config()

    # ======================= MOCK ISOLATION TESTS =======================
    
    def test_isolated_environment_tests_multiple_dest(self):
        """Test that environment mocking properly isolates tests with multiple destinations."""
        # Test 1: Set specific environment with multiple destinations
        with patch.dict(os.environ, {"GLIMPSE_DEST": "mongo,sqllite"}, clear=True):
            config1 = Config()
            assert config1.dest == ["mongo", "sqllite"]
        
        # Test 2: Different environment should not interfere
        with patch.dict(os.environ, {"GLIMPSE_DEST": "json"}, clear=True):
            config2 = Config()
            assert config2.dest == ["json"]
        
        # Test 3: Clean environment
        with patch.dict(os.environ, {}, clear=True):
            config3 = Config(env_override=False)
            assert config3.dest == ["jsonl"]  # Default