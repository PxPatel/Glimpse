import pytest
import json
import re
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from glimpse.policy.policy import TracingPolicy, ExactPatternTrie, ExactModuleSet

class TestExactPatternTrie:
    """Unit tests for the ExactPatternTrie data structure."""
    
    def test_trie_init(self):
        """Test trie initialization."""
        trie = ExactPatternTrie()
        assert trie.root == {}
    
    def test_add_single_pattern(self):
        """Test adding a single pattern to trie."""
        trie = ExactPatternTrie()
        trie.add_pattern("myapp")
        
        # Should create path: root -> myapp -> _is_pattern
        assert "myapp" in trie.root
        assert trie.root["myapp"]["_is_pattern"] is True
    
    def test_add_nested_pattern(self):
        """Test adding nested pattern to trie."""
        trie = ExactPatternTrie()
        trie.add_pattern("myapp.services.user")
        
        # Should create path: root -> myapp -> services -> user -> _is_pattern
        node = trie.root["myapp"]["services"]["user"]
        assert node["_is_pattern"] is True
    
    def test_add_multiple_patterns(self):
        """Test adding multiple patterns with shared prefixes."""
        trie = ExactPatternTrie()
        trie.add_pattern("myapp")
        trie.add_pattern("myapp.services")
        trie.add_pattern("myapp.models")
        trie.add_pattern("requests")
        
        # Both myapp and myapp.services should be marked as patterns
        assert trie.root["myapp"]["_is_pattern"] is True
        assert trie.root["myapp"]["services"]["_is_pattern"] is True
        assert trie.root["myapp"]["models"]["_is_pattern"] is True
        assert trie.root["requests"]["_is_pattern"] is True
    
    def test_add_empty_pattern(self):
        """Test adding empty or whitespace patterns."""
        trie = ExactPatternTrie()
        trie.add_pattern("")
        trie.add_pattern("   ")
        trie.add_pattern("\t\n")
        
        # Should not add anything to trie
        assert trie.root == {}
    
    def test_matches_exact_pattern(self):
        """Test exact pattern matching."""
        trie = ExactPatternTrie()
        trie.add_pattern("myapp")
        trie.add_pattern("requests")
        
        assert trie.matches("myapp") is True
        assert trie.matches("requests") is True
        assert trie.matches("numpy") is False
    
    def test_matches_submodule_pattern(self):
        """Test submodule pattern matching."""
        trie = ExactPatternTrie()
        trie.add_pattern("myapp")
        
        # Should match submodules
        assert trie.matches("myapp.services") is True
        assert trie.matches("myapp.services.user") is True
        assert trie.matches("myapp.models.account") is True
        
        # Should not match similar but different packages
        assert trie.matches("myapplication") is False
        assert trie.matches("app") is False
    
    def test_matches_multiple_levels(self):
        """Test matching with patterns at different levels."""
        trie = ExactPatternTrie()
        trie.add_pattern("myapp")
        trie.add_pattern("myapp.services")
        
        # Should match at first level
        assert trie.matches("myapp.models") is True
        
        # Should match at second level
        assert trie.matches("myapp.services.user") is True
        
        # Should not match deeper than any pattern
        assert trie.matches("other.module") is False
    
    def test_matches_empty_input(self):
        """Test matching with empty input."""
        trie = ExactPatternTrie()
        trie.add_pattern("myapp")
        
        assert trie.matches("") is False
        assert trie.matches("   ") is False
    
    def test_matches_empty_trie(self):
        """Test matching with empty trie."""
        trie = ExactPatternTrie()
        
        assert trie.matches("any.module") is False
        assert trie.matches("") is False


class TestExactModuleSet:
    """Unit tests for the ExactModuleSet data structure."""
    
    def test_init(self):
        """Test ExactModuleSet initialization."""
        module_set = ExactModuleSet()
        assert module_set.modules == set()
    
    def test_add_module(self):
        """Test adding modules to set."""
        module_set = ExactModuleSet()
        module_set.add_module("myapp.payment")
        module_set.add_module("requests.sessions")
        
        assert "myapp.payment" in module_set.modules
        assert "requests.sessions" in module_set.modules
    
    def test_add_empty_module(self):
        """Test adding empty/whitespace modules."""
        module_set = ExactModuleSet()
        module_set.add_module("")
        module_set.add_module("   ")
        module_set.add_module("\t\n")
        
        # Should not add empty modules
        assert len(module_set.modules) == 0
    
    def test_matches_exact_only(self):
        """Test that ExactModuleSet only matches exactly."""
        module_set = ExactModuleSet()
        module_set.add_module("myapp.payment")
        
        # Should match exactly
        assert module_set.matches("myapp.payment") is True
        
        # Should NOT match submodules
        assert module_set.matches("myapp.payment.stripe") is False
        assert module_set.matches("myapp.payment.processor") is False
        
        # Should NOT match partial matches
        assert module_set.matches("myapp") is False
        assert module_set.matches("myapp.payment.processor.utils") is False
    
    def test_matches_multiple_modules(self):
        """Test matching with multiple modules."""
        module_set = ExactModuleSet()
        module_set.add_module("myapp.payment")
        module_set.add_module("myapp.auth.login")
        module_set.add_module("requests.sessions")
        
        assert module_set.matches("myapp.payment") is True
        assert module_set.matches("myapp.auth.login") is True
        assert module_set.matches("requests.sessions") is True
        
        # Submodules should not match
        assert module_set.matches("myapp.payment.stripe") is False
        assert module_set.matches("myapp.auth.login.oauth") is False
        assert module_set.matches("requests.sessions.poolmanager") is False


class TestTracingPolicy:
    """Comprehensive unit tests for TracingPolicy class with new exact_modules/package_trees design."""
    
    # ======================= INITIALIZATION TESTS =======================
    
    def test_init_with_defaults(self):
        """Test TracingPolicy initialization with default values."""
        policy = TracingPolicy()
        
        # Check exact modules structures
        assert isinstance(policy.exact_modules_set, ExactModuleSet)
        assert len(policy.exact_modules_set.modules) == 0
        assert isinstance(policy.exact_modules_wildcards, set)
        assert len(policy.exact_modules_wildcards) == 0
        
        # Check package trees structures
        assert isinstance(policy.package_trees_trie, ExactPatternTrie)
        assert policy.package_trees_trie.root == {}
        assert isinstance(policy.package_trees_wildcards, set)
        assert len(policy.package_trees_wildcards) == 0
        
        # Check other attributes
        assert policy.trace_depth == 5

    def test_init_exact_modules_only(self):
        """Test initialization with only exact modules."""
        policy = TracingPolicy(
            exact_modules=["myapp.payment.processor", "requests.sessions"],
            trace_depth=10
        )
        
        # Verify exact modules
        assert policy.exact_modules_set.matches("myapp.payment.processor") is True
        assert policy.exact_modules_set.matches("requests.sessions") is True
        assert len(policy.exact_modules_wildcards) == 0
        
        # Verify no package trees
        assert policy.package_trees_trie.root == {}
        assert len(policy.package_trees_wildcards) == 0
        
        assert policy.trace_depth == 10

    def test_init_package_trees_only(self):
        """Test initialization with only package trees."""
        policy = TracingPolicy(
            package_trees=["myapp.auth", "django.contrib"],
            trace_depth=8
        )
        
        # Verify no exact modules
        assert len(policy.exact_modules_set.modules) == 0
        assert len(policy.exact_modules_wildcards) == 0
        
        # Verify package trees
        assert policy.package_trees_trie.matches("myapp.auth") is True
        assert policy.package_trees_trie.matches("django.contrib") is True
        assert len(policy.package_trees_wildcards) == 0
        
        assert policy.trace_depth == 8

    def test_init_mixed_exact_and_trees(self):
        """Test initialization with both exact modules and package trees."""
        policy = TracingPolicy(
            exact_modules=["requests.sessions", "urllib3.poolmanager"],
            package_trees=["myapp.auth", "myapp.payment"]
        )
        
        # Verify exact modules
        assert policy.exact_modules_set.matches("requests.sessions") is True
        assert policy.exact_modules_set.matches("urllib3.poolmanager") is True
        
        # Verify package trees
        assert policy.package_trees_trie.matches("myapp.auth") is True
        assert policy.package_trees_trie.matches("myapp.payment") is True

    def test_init_with_wildcards_exact_modules(self):
        """Test initialization with wildcard exact modules."""
        policy = TracingPolicy(
            exact_modules=["test_*", "api_v?", "config[12]"]
        )
        
        # Should have wildcard patterns compiled
        assert len(policy.exact_modules_set.modules) == 0
        assert len(policy.exact_modules_wildcards) == 3
        
        # Verify patterns are compiled regex objects
        for pattern in policy.exact_modules_wildcards:
            assert hasattr(pattern, 'match')

    def test_init_with_wildcards_package_trees(self):
        """Test initialization with wildcard package trees."""
        policy = TracingPolicy(
            package_trees=["app*", "test_*", "*.models"]
        )
        
        # Should have wildcard patterns compiled
        assert policy.package_trees_trie.root == {}
        assert len(policy.package_trees_wildcards) == 3
        
        # Verify patterns are compiled regex objects
        for pattern in policy.package_trees_wildcards:
            assert hasattr(pattern, 'match')

    def test_init_mixed_exact_and_wildcards(self):
        """Test initialization with mix of exact and wildcard patterns."""
        policy = TracingPolicy(
            exact_modules=["myapp.payment", "test_*"],
            package_trees=["myapp.auth", "api_*"]
        )
        
        # Exact modules: one exact, one wildcard
        assert policy.exact_modules_set.matches("myapp.payment") is True
        assert len(policy.exact_modules_wildcards) == 1
        
        # Package trees: one exact, one wildcard
        assert policy.package_trees_trie.matches("myapp.auth") is True
        assert len(policy.package_trees_wildcards) == 1

    def test_init_filters_empty_patterns(self):
        """Test that initialization filters out empty/whitespace patterns."""
        policy = TracingPolicy(
            exact_modules=["myapp.payment", "", "  ", "\t\n", "requests.sessions"],
            package_trees=["myapp.auth", "", "   ", "myapp.models"]
        )
        
        # Only non-empty patterns should be added
        assert policy.exact_modules_set.matches("myapp.payment") is True
        assert policy.exact_modules_set.matches("requests.sessions") is True
        assert policy.package_trees_trie.matches("myapp.auth") is True
        assert policy.package_trees_trie.matches("myapp.models") is True

    def test_init_none_parameters(self):
        """Test initialization with None parameters."""
        policy = TracingPolicy(
            exact_modules=None,
            package_trees=None
        )
        
        assert len(policy.exact_modules_set.modules) == 0
        assert len(policy.exact_modules_wildcards) == 0
        assert policy.package_trees_trie.root == {}
        assert len(policy.package_trees_wildcards) == 0

    # ======================= WILDCARD HELPER METHODS TESTS =======================
    
    @pytest.mark.parametrize("pattern,expected", [
        ("myapp", False),
        ("requests", False),
        ("normal.package.name", False),
        ("", False),
        ("app*", True),
        ("test_*", True),
        ("*.models", True),
        ("api_v?", True),
        ("config[12]", True),
        ("config[abc]", True),
        ("app[abc]*test", True),
        ("complex*pattern?with[123]", True),
    ])
    def test_has_wildcards(self, pattern, expected):
        """Test _has_wildcards method with various patterns."""
        policy = TracingPolicy()
        assert policy._has_wildcards(pattern) == expected

    def test_compile_exact_pattern(self):
        """Test _compile_exact_pattern method for exact module wildcards."""
        policy = TracingPolicy()
        
        # Test exact pattern compilation (should NOT match submodules)
        pattern = policy._compile_exact_pattern("test_*")
        
        # Should match the pattern exactly
        assert pattern.match("test_unit") is not None
        assert pattern.match("test_integration") is not None
        
        # Should NOT match submodules
        assert pattern.match("test_unit.helpers") is None
        assert pattern.match("test_integration.db") is None
        
        # Should NOT match non-matching patterns
        assert pattern.match("mytest_unit") is None

    def test_compile_tree_pattern(self):
        """Test _compile_tree_pattern method for package tree wildcards."""
        policy = TracingPolicy()
        
        # Test tree pattern compilation (should match submodules)
        pattern = policy._compile_tree_pattern("test_*")
        
        # Should match the pattern exactly
        assert pattern.match("test_unit") is not None
        assert pattern.match("test_integration") is not None
        
        # Should ALSO match submodules
        assert pattern.match("test_unit.helpers") is not None
        assert pattern.match("test_integration.db.models") is not None
        
        # Should NOT match non-matching patterns
        assert pattern.match("mytest_unit") is None

    def test_wildcard_compilation_comparison(self):
        """Test that exact vs tree pattern compilation behaves differently."""
        policy = TracingPolicy()
        
        exact_pattern = policy._compile_exact_pattern("api_*")
        tree_pattern = policy._compile_tree_pattern("api_*")
        
        test_cases = [
            ("api_v1", True, True),           # Both should match base
            ("api_v2", True, True),           # Both should match base
            ("api_v1.auth", False, True),     # Only tree should match submodule
            ("api_v1.auth.oauth", False, True), # Only tree should match deep submodule
            ("myapi_v1", False, False),       # Neither should match
        ]
        
        for module_name, exact_expected, tree_expected in test_cases:
            exact_result = exact_pattern.match(module_name) is not None
            tree_result = tree_pattern.match(module_name) is not None
            
            assert exact_result == exact_expected, f"Exact pattern failed for {module_name}"
            assert tree_result == tree_expected, f"Tree pattern failed for {module_name}"

    # ======================= EXACT MODULES MATCHING TESTS =======================
    
    def test_matches_exact_modules_simple(self):
        """Test _matches_exact_modules with simple cases."""
        policy = TracingPolicy(exact_modules=["myapp.payment", "requests.sessions"])
        
        # Should match exactly
        assert policy._matches_exact_modules("myapp.payment") is True
        assert policy._matches_exact_modules("requests.sessions") is True
        
        # Should NOT match submodules
        assert policy._matches_exact_modules("myapp.payment.stripe") is False
        assert policy._matches_exact_modules("requests.sessions.poolmanager") is False
        
        # Should NOT match partial matches
        assert policy._matches_exact_modules("myapp") is False
        assert policy._matches_exact_modules("requests") is False

    def test_matches_exact_modules_wildcards(self):
        """Test _matches_exact_modules with wildcard patterns."""
        policy = TracingPolicy(exact_modules=["test_*", "api_v?"])
        
        # Should match exact wildcard patterns
        assert policy._matches_exact_modules("test_unit") is True
        assert policy._matches_exact_modules("test_integration") is True
        assert policy._matches_exact_modules("api_v1") is True
        assert policy._matches_exact_modules("api_v2") is True
        
        # Should NOT match submodules of wildcard patterns
        assert policy._matches_exact_modules("test_unit.helpers") is False
        assert policy._matches_exact_modules("api_v1.auth") is False
        
        # Should NOT match non-matching patterns
        assert policy._matches_exact_modules("mytest_unit") is False
        assert policy._matches_exact_modules("api_v12") is False

    def test_matches_exact_modules_mixed(self):
        """Test _matches_exact_modules with mixed exact and wildcard patterns."""
        policy = TracingPolicy(exact_modules=["myapp.payment", "test_*"])
        
        # Exact pattern
        assert policy._matches_exact_modules("myapp.payment") is True
        assert policy._matches_exact_modules("myapp.payment.stripe") is False
        
        # Wildcard pattern
        assert policy._matches_exact_modules("test_unit") is True
        assert policy._matches_exact_modules("test_unit.helpers") is False

    # ======================= PACKAGE TREES MATCHING TESTS =======================
    
    def test_matches_package_trees_simple(self):
        """Test _matches_package_trees with simple cases."""
        policy = TracingPolicy(package_trees=["myapp.auth", "django.contrib"])
        
        # Should match exactly
        assert policy._matches_package_trees("myapp.auth") is True
        assert policy._matches_package_trees("django.contrib") is True
        
        # Should ALSO match submodules
        assert policy._matches_package_trees("myapp.auth.login") is True
        assert policy._matches_package_trees("myapp.auth.oauth.google") is True
        assert policy._matches_package_trees("django.contrib.admin") is True
        assert policy._matches_package_trees("django.contrib.auth.models") is True
        
        # Should NOT match non-matching patterns
        assert policy._matches_package_trees("myapp.payment") is False
        assert policy._matches_package_trees("flask.contrib") is False

    def test_matches_package_trees_wildcards(self):
        """Test _matches_package_trees with wildcard patterns."""
        policy = TracingPolicy(package_trees=["test_*", "api_*"])
        
        # Should match wildcard patterns
        assert policy._matches_package_trees("test_unit") is True
        assert policy._matches_package_trees("test_integration") is True
        assert policy._matches_package_trees("api_v1") is True
        assert policy._matches_package_trees("api_v2") is True
        
        # Should ALSO match submodules of wildcard patterns
        assert policy._matches_package_trees("test_unit.helpers") is True
        assert policy._matches_package_trees("test_integration.db.models") is True
        assert policy._matches_package_trees("api_v1.auth.oauth") is True
        
        # Should NOT match non-matching patterns
        assert policy._matches_package_trees("mytest_unit") is False
        assert policy._matches_package_trees("myapi_v1") is False

    def test_matches_package_trees_mixed(self):
        """Test _matches_package_trees with mixed exact and wildcard patterns."""
        policy = TracingPolicy(package_trees=["myapp.auth", "test_*"])
        
        # Exact pattern + submodules
        assert policy._matches_package_trees("myapp.auth") is True
        assert policy._matches_package_trees("myapp.auth.login") is True
        
        # Wildcard pattern + submodules
        assert policy._matches_package_trees("test_unit") is True
        assert policy._matches_package_trees("test_unit.helpers") is True

    # ======================= SHOULD_TRACE_PACKAGE INTEGRATION TESTS =======================
    
    def test_should_trace_package_exact_modules_only(self):
        """Test should_trace_package with only exact modules."""
        policy = TracingPolicy(exact_modules=["myapp.payment", "requests.sessions"])
        
        # Should match exact modules only
        assert policy.should_trace_package("myapp.payment") is True
        assert policy.should_trace_package("requests.sessions") is True
        
        # Should NOT match submodules
        assert policy.should_trace_package("myapp.payment.stripe") is False
        assert policy.should_trace_package("requests.sessions.poolmanager") is False
        
        # Should NOT match other modules
        assert policy.should_trace_package("myapp.auth") is False
        assert policy.should_trace_package("django.contrib") is False

    def test_should_trace_package_package_trees_only(self):
        """Test should_trace_package with only package trees."""
        policy = TracingPolicy(package_trees=["myapp.auth", "django.contrib"])
        
        # Should match package trees
        assert policy.should_trace_package("myapp.auth") is True
        assert policy.should_trace_package("django.contrib") is True
        
        # Should ALSO match submodules
        assert policy.should_trace_package("myapp.auth.login") is True
        assert policy.should_trace_package("myapp.auth.oauth.google") is True
        assert policy.should_trace_package("django.contrib.admin.models") is True
        
        # Should NOT match other packages
        assert policy.should_trace_package("myapp.payment") is False
        assert policy.should_trace_package("flask.contrib") is False

    def test_should_trace_package_mixed_exact_and_trees(self):
        """Test should_trace_package with both exact modules and package trees."""
        policy = TracingPolicy(
            exact_modules=["requests.sessions", "urllib3.poolmanager"],
            package_trees=["myapp.auth", "myapp.payment"]
        )
        
        # Should match exact modules (no submodules)
        assert policy.should_trace_package("requests.sessions") is True
        assert policy.should_trace_package("urllib3.poolmanager") is True
        assert policy.should_trace_package("requests.sessions.adapters") is False
        
        # Should match package trees (including submodules)
        assert policy.should_trace_package("myapp.auth") is True
        assert policy.should_trace_package("myapp.auth.login") is True
        assert policy.should_trace_package("myapp.payment.stripe") is True
        
        # Should NOT match other modules
        assert policy.should_trace_package("django.contrib") is False

    def test_should_trace_package_priority_package_trees_wins(self):
        """Test that package_trees has priority over exact_modules when overlapping."""
        policy = TracingPolicy(
            exact_modules=["myapp.auth.login"],  # Would normally only match exactly
            package_trees=["myapp.auth"]         # This covers the exact module + submodules
        )
        
        # The overlap: myapp.auth.login should be handled by package_trees
        assert policy.should_trace_package("myapp.auth.login") is True
        
        # This proves package_trees wins: submodules should also match
        assert policy.should_trace_package("myapp.auth.login.utils") is True
        
        # Other myapp.auth submodules should also match via package_trees
        assert policy.should_trace_package("myapp.auth.oauth") is True

    def test_should_trace_package_wildcards_exact_modules(self):
        """Test should_trace_package with wildcard exact modules."""
        policy = TracingPolicy(exact_modules=["test_*", "api_v?"])
        
        # Should match wildcard patterns exactly
        assert policy.should_trace_package("test_unit") is True
        assert policy.should_trace_package("test_integration") is True
        assert policy.should_trace_package("api_v1") is True
        assert policy.should_trace_package("api_v2") is True
        
        # Should NOT match submodules
        assert policy.should_trace_package("test_unit.helpers") is False
        assert policy.should_trace_package("api_v1.auth") is False

    def test_should_trace_package_wildcards_package_trees(self):
        """Test should_trace_package with wildcard package trees."""
        policy = TracingPolicy(package_trees=["test_*", "api_*"])
        
        # Should match wildcard patterns
        assert policy.should_trace_package("test_unit") is True
        assert policy.should_trace_package("api_v1") is True
        
        # Should ALSO match submodules
        assert policy.should_trace_package("test_unit.helpers") is True
        assert policy.should_trace_package("test_integration.db.models") is True
        assert policy.should_trace_package("api_v1.auth.oauth") is True

    def test_should_trace_package_empty_configuration(self):
        """Test should_trace_package with no patterns configured."""
        policy = TracingPolicy()
        
        assert policy.should_trace_package("any.module") is False
        assert policy.should_trace_package("myapp.payment") is False
        assert policy.should_trace_package("") is False

    def test_should_trace_package_empty_inputs(self):
        """Test should_trace_package with empty/invalid inputs."""
        policy = TracingPolicy(exact_modules=["myapp.payment"])
        
        assert policy.should_trace_package("") is False
        assert policy.should_trace_package("   ") is False
        assert policy.should_trace_package("\t\n") is False

    # ======================= LOAD METHOD TESTS =======================
    
    def test_load_new_format_success(self, tmp_path):
        """Test load method with new JSON format."""
        policy_data = {
            "exact_modules": ["myapp.payment.processor", "requests.sessions"],
            "package_trees": ["myapp.auth", "django.contrib"],
            "trace_depth": 8
        }
        
        policy_file = tmp_path / "glimpse-policy.json"
        policy_file.write_text(json.dumps(policy_data))
        
        policy = TracingPolicy.load(str(policy_file))
        
        # Verify exact modules
        assert policy.exact_modules_set.matches("myapp.payment.processor") is True
        assert policy.exact_modules_set.matches("requests.sessions") is True
        
        # Verify package trees
        assert policy.package_trees_trie.matches("myapp.auth") is True
        assert policy.package_trees_trie.matches("django.contrib") is True
        
        assert policy.trace_depth == 8

    def test_load_mixed_exact_and_wildcard_patterns(self, tmp_path):
        """Test load method with mixed exact and wildcard patterns."""
        policy_data = {
            "exact_modules": ["myapp.payment", "test_*"],
            "package_trees": ["myapp.auth", "api_*"],
            "trace_depth": 5
        }
        
        policy_file = tmp_path / "glimpse-policy.json"
        policy_file.write_text(json.dumps(policy_data))
        
        policy = TracingPolicy.load(str(policy_file))
        
        # Verify exact modules: one exact, one wildcard
        assert policy.exact_modules_set.matches("myapp.payment") is True
        assert len(policy.exact_modules_wildcards) == 1
        
        # Verify package trees: one exact, one wildcard
        assert policy.package_trees_trie.matches("myapp.auth") is True
        assert len(policy.package_trees_wildcards) == 1

    def test_load_explicit_path_invalid_filename(self, tmp_path):
        """Test load method with invalid filename."""
        policy_file = tmp_path / "wrong-name.json"
        policy_file.write_text('{"exact_modules": []}')
        
        with pytest.raises(ValueError, match="Policy file must be named 'glimpse-policy.json'"):
            TracingPolicy.load(str(policy_file))

    def test_load_explicit_path_not_found(self, tmp_path):
        """Test load method with non-existent file."""
        policy_file = tmp_path / "glimpse-policy.json"
        
        with pytest.raises(FileNotFoundError, match="Policy file not found"):
            TracingPolicy.load(str(policy_file))

    def test_load_invalid_json(self, tmp_path):
        """Test load method with invalid JSON."""
        policy_file = tmp_path / "glimpse-policy.json"
        policy_file.write_text('{"invalid": json content}')
        
        with pytest.raises(json.JSONDecodeError, match="Invalid JSON"):
            TracingPolicy.load(str(policy_file))

    def test_load_discovery_success(self, tmp_path):
        """Test load method with automatic discovery."""
        # Create nested directory structure
        nested_dir = tmp_path / "app" / "deep" / "nested"
        nested_dir.mkdir(parents=True)
        
        # Create policy file in parent directory
        policy_data = {
            "exact_modules": ["discovered.module"],
            "trace_depth": 3
        }
        policy_file = tmp_path / "glimpse-policy.json"
        policy_file.write_text(json.dumps(policy_data))
        
        # Mock the caller frame to simulate calling from nested directory
        with patch('inspect.currentframe') as mock_frame:
            mock_caller = MagicMock()
            mock_caller.f_code.co_filename = str(nested_dir / "caller.py")
            mock_frame.return_value.f_back = mock_caller
            
            policy = TracingPolicy.load()
            
            assert policy.exact_modules_set.matches("discovered.module") is True
            assert policy.trace_depth == 3

    def test_load_discovery_not_found(self, tmp_path):
        """Test load method when discovery fails."""
        nested_dir = tmp_path / "app" / "deep" / "nested"
        nested_dir.mkdir(parents=True)
        
        # No policy file anywhere in the tree
        with patch('inspect.currentframe') as mock_frame:
            mock_caller = MagicMock()
            mock_caller.f_code.co_filename = str(nested_dir / "caller.py")
            mock_frame.return_value.f_back = mock_caller
            
            with pytest.raises(FileNotFoundError, match="No 'glimpse-policy.json' found"):
                TracingPolicy.load()

    def test_load_with_defaults(self, tmp_path):
        """Test load method uses proper defaults for missing keys."""
        policy_data = {
            "exact_modules": ["myapp.payment"]
            # Missing other keys should use defaults
        }
        
        policy_file = tmp_path / "glimpse-policy.json"
        policy_file.write_text(json.dumps(policy_data))
        
        policy = TracingPolicy.load(str(policy_file))
        
        assert policy.exact_modules_set.matches("myapp.payment") is True
        assert policy.trace_depth == 5  # Default
        assert policy.package_trees_trie.root == {}  # Default empty

    def test_load_empty_lists(self, tmp_path):
        """Test load method with empty lists."""
        policy_data = {
            "exact_modules": [],
            "package_trees": [],
            "trace_depth": 10
        }
        
        policy_file = tmp_path / "glimpse-policy.json"
        policy_file.write_text(json.dumps(policy_data))
        
        policy = TracingPolicy.load(str(policy_file))
        
        assert len(policy.exact_modules_set.modules) == 0
        assert policy.package_trees_trie.root == {}
        assert policy.trace_depth == 10

    # ======================= PERFORMANCE TESTS =======================
    
    def test_performance_exact_modules_large_set(self):
        """Test performance with large number of exact modules."""
        # Create many exact modules
        modules = [f"module_{i}.component" for i in range(1000)]
        policy = TracingPolicy(exact_modules=modules)
        
        test_modules = [
            "module_500.component",     # Should match
            "module_750.component",     # Should match
            "module_500.component.sub", # Should NOT match (submodule)
            "nomatch.module",           # Should NOT match
        ]
        
        # Should be fast due to set lookup for exact matches
        start = time.time()
        for _ in range(1000):
            for module in test_modules:
                policy.should_trace_package(module)
        elapsed = time.time() - start
        
        # Should complete quickly even with 1000 modules
        assert elapsed < 0.1  # Less than 100ms for 4000 checks

    def test_performance_package_trees_large_trie(self):
        """Test performance with large package trees trie."""
        # Create many package trees with shared prefixes
        packages = []
        for i in range(100):
            base = f"package_{i}"
            packages.append(base)
            for j in range(10):
                packages.append(f"{base}.module_{j}")
        
        policy = TracingPolicy(package_trees=packages)
        
        test_modules = [
            "package_50.module_5.deep.path",   # Should match
            "package_75.module_8",             # Should match
            "nomatch.module",                  # Should NOT match
        ]
        
        # Should be fast due to trie optimization
        start = time.time()
        for _ in range(1000):
            for module in test_modules:
                policy.should_trace_package(module)
        elapsed = time.time() - start
        
        # Should complete quickly even with 1100 packages
        assert elapsed < 0.2  # Less than 200ms for 3000 checks

    def test_performance_mixed_large_configuration(self):
        """Test performance with large mixed configuration."""
        # Large exact modules set
        exact_modules = [f"exact_{i}.module" for i in range(500)]
        
        # Large package trees set
        package_trees = [f"tree_{i}" for i in range(500)]
        
        policy = TracingPolicy(
            exact_modules=exact_modules,
            package_trees=package_trees
        )
        
        test_modules = [
            "exact_250.module",         # Should match via exact modules
            "tree_300.any.deep.path",   # Should match via package trees
            "nomatch.module",           # Should NOT match
        ]
        
        # Should handle large mixed configuration efficiently
        start = time.time()
        for _ in range(1000):
            for module in test_modules:
                policy.should_trace_package(module)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 0.3  # Less than 300ms for 3000 checks

    # ======================= EDGE CASES AND ERROR HANDLING =======================
    
    def test_unicode_patterns_and_modules(self):
        """Test handling of unicode characters."""
        policy = TracingPolicy(
            exact_modules=["测试模块.支付"],
            package_trees=["пакет*", "パッケージ.モジュール"]
        )
        
        # Exact unicode module
        assert policy.should_trace_package("测试模块.支付") is True
        assert policy.should_trace_package("测试模块.支付.子模块") is False  # No submodules for exact
        
        # Wildcard unicode package tree
        assert policy.should_trace_package("пакет_тест") is True
        assert policy.should_trace_package("пакет_тест.подмодуль") is True  # Submodules for trees
        
        # Exact unicode package tree
        assert policy.should_trace_package("パッケージ.モジュール") is True
        assert policy.should_trace_package("パッケージ.モジュール.サブ") is True  # Submodules for trees

    def test_very_long_module_names(self):
        """Test with very long module names."""
        policy = TracingPolicy(
            exact_modules=["very.long.exact.module.name"],
            package_trees=["very.long.tree*"]
        )
        
        # Create very long module names
        long_exact = "very.long.exact.module.name"
        long_tree_match = "very.long.tree." + ".".join([f"part_{i}" for i in range(50)])
        
        # Should handle long names efficiently
        start = time.time()
        
        result1 = policy.should_trace_package(long_exact)
        result2 = policy.should_trace_package(long_tree_match)
        
        elapsed = time.time() - start
        
        assert result1 is True  # Should match exact
        assert result2 is True  # Should match tree
        assert elapsed < 0.001  # Should be very fast

    def test_complex_wildcard_patterns(self):
        """Test complex wildcard patterns work correctly."""
        policy = TracingPolicy(
            exact_modules=[
                "test_*_processor",       # Multi-part wildcard exact
                "api_v[12]",             # Character class exact
            ],
            package_trees=[
                "app*_service",          # Multi-part wildcard tree
                "module_[abc]*",         # Character class + wildcard tree
            ]
        )
        
        # Test exact modules with complex wildcards
        assert policy.should_trace_package("test_unit_processor") is True
        assert policy.should_trace_package("test_integration_processor") is True
        assert policy.should_trace_package("test_unit_processor.helper") is False  # No submodules
        assert policy.should_trace_package("api_v1") is True
        assert policy.should_trace_package("api_v2") is True
        assert policy.should_trace_package("api_v1.auth") is False  # No submodules
        
        # Test package trees with complex wildcards
        assert policy.should_trace_package("app_main_service") is True
        assert policy.should_trace_package("app_auth_service.login") is True  # Submodules allowed
        assert policy.should_trace_package("module_a_extended") is True
        assert policy.should_trace_package("module_b_system.deep.path") is True  # Submodules allowed

    def test_edge_case_module_names(self):
        """Test edge cases in module naming."""
        policy = TracingPolicy(
            exact_modules=["test"],
            package_trees=["special*"]
        )
        
        # Test various edge cases
        test_cases = [
            ("test", True, False),           # Should match exact only
            ("test.", False, False),         # Trailing dot
            (".test", False, False),         # Leading dot
            ("test..module", False, False),  # Double dots
            ("special_chars", False, True),  # Should match tree wildcard
            ("special_chars.sub", False, True), # Submodule of tree
            ("UPPERCASE", False, False),     # Case sensitivity
        ]
        
        for module_name, expected_exact, expected_tree in test_cases:
            result = policy.should_trace_package(module_name)
            expected = expected_exact or expected_tree
            assert result == expected, f"Module '{module_name}': expected {expected}, got {result}"

    def test_stress_test_comprehensive(self):
        """Comprehensive stress test with many patterns and lookups."""
        # Create large configuration
        exact_modules = [f"exact_{i}.module_{j}" for i in range(20) for j in range(10)]
        package_trees = [f"tree_{i}" for i in range(50)]
        
        policy = TracingPolicy(
            exact_modules=exact_modules,
            package_trees=package_trees
        )
        
        test_modules = [
            "exact_10.module_5",           # Should match exact
            "tree_25.any.deep.path",       # Should match tree
            "exact_15.module_8.submodule", # Should NOT match (exact + submodule)
            "nomatch.module",              # Should NOT match
        ]
        
        # Should handle stress test efficiently
        start = time.time()
        for _ in range(100):
            for module in test_modules:
                policy.should_trace_package(module)
        elapsed = time.time() - start
        
        # Should complete quickly even with 250 patterns total
        assert elapsed < 0.1  # Less than 100ms for 400 checks