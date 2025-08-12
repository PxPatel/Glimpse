import pytest
import json
import re
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from glimpse.policy.policy import TracingPolicy

class TestTracingPolicy:
    """Comprehensive unit tests for TracingPolicy class."""
    
    # ======================= INITIALIZATION TESTS =======================
    
    def test_init_with_defaults(self):
        """Test TracingPolicy initialization with default values."""
        policy = TracingPolicy()
        
        assert isinstance(policy.exact_included, set)
        assert len(policy.exact_included) == 0
        assert isinstance(policy.wildcard_included, set)
        assert len(policy.wildcard_included) == 0
        assert isinstance(policy.exact_project_roots, set)
        assert len(policy.exact_project_roots) == 0
        assert isinstance(policy.wildcard_project_roots, set)
        assert len(policy.wildcard_project_roots) == 0
        assert policy.trace_depth == 5
        assert policy.auto_trace_subpackages is False

    def test_init_with_exact_patterns_only(self):
        """Test initialization with only exact patterns (no wildcards)."""
        policy = TracingPolicy(
            included_packages=["myapp", "requests", "numpy"],
            project_root_packages=["src", "lib"],
            trace_depth=10,
            auto_trace_subpackages=True
        )
        
        # All patterns should be in exact sets
        assert policy.exact_included == {"myapp", "requests", "numpy"}
        assert len(policy.wildcard_included) == 0
        assert policy.exact_project_roots == {"src", "lib"}
        assert len(policy.wildcard_project_roots) == 0
        assert policy.trace_depth == 10
        assert policy.auto_trace_subpackages is True

    def test_init_with_wildcard_patterns_only(self):
        """Test initialization with only wildcard patterns."""
        policy = TracingPolicy(
            included_packages=["app*", "test_*", "*.models"],
            project_root_packages=["src*", "lib?"],
            auto_trace_subpackages=True
        )
        
        # All patterns should be in wildcard lists
        assert len(policy.exact_included) == 0
        assert len(policy.wildcard_included) == 3
        assert len(policy.exact_project_roots) == 0
        assert len(policy.wildcard_project_roots) == 2
        assert policy.auto_trace_subpackages is True
        
        # Verify patterns are compiled regex objects
        for pattern in policy.wildcard_included:
            assert hasattr(pattern, 'match')  # Compiled regex has match method
        for pattern in policy.wildcard_project_roots:
            assert hasattr(pattern, 'match')

    def test_init_with_mixed_patterns(self):
        """Test initialization with both exact and wildcard patterns."""
        policy = TracingPolicy(
            included_packages=["myapp", "requests", "test_*", "app*"],
            project_root_packages=["src", "lib*"],
            auto_trace_subpackages=True
        )
        
        # Exact patterns
        assert policy.exact_included == {"myapp", "requests"}
        assert policy.exact_project_roots == {"src"}
        
        # Wildcard patterns (should be compiled)
        assert len(policy.wildcard_included) == 2  # test_*, app*
        assert len(policy.wildcard_project_roots) == 1  # lib*
        
        # Verify compiled patterns work
        test_pattern = None
        app_pattern = None
        for pattern in policy.wildcard_included:
            if pattern.match("test_unit"):
                test_pattern = pattern
            if pattern.match("application"):
                app_pattern = pattern
        
        assert test_pattern is not None
        assert app_pattern is not None

    def test_init_with_duplicates(self):
        """Test initialization removes duplicates correctly."""
        policy = TracingPolicy(
            included_packages=["myapp", "myapp", "test_*", "test_*"],
            project_root_packages=["src", "src", "lib*", "lib*"]
        )
        
        # Exact patterns should be deduplicated by set
        assert policy.exact_included == {"myapp"}
        assert policy.exact_project_roots == {"src"}
        
        # Wildcard patterns should be deduplicated (though stored as list)
        assert len(policy.wildcard_included) == 1
        assert len(policy.wildcard_project_roots) == 1

    def test_init_auto_trace_logic(self):
        """Test auto_trace_subpackages logic based on project root packages."""
        # Should be False when no project roots
        policy1 = TracingPolicy(auto_trace_subpackages=True)
        assert policy1.auto_trace_subpackages is False
        
        # Should be True when project roots exist
        policy2 = TracingPolicy(
            project_root_packages=["src"],
            auto_trace_subpackages=True
        )
        assert policy2.auto_trace_subpackages is True
        
        # Should be False when explicitly disabled
        policy3 = TracingPolicy(
            project_root_packages=["src"],
            auto_trace_subpackages=False
        )
        assert policy3.auto_trace_subpackages is False

    # ======================= WILDCARD DETECTION TESTS =======================
    
    @pytest.mark.parametrize("pattern,expected", [
        ("myapp", False),
        ("requests", False),
        ("app*", True),
        ("test_*", True),
        ("*.models", True),
        ("api_v?", True),
        ("config[12]", True),
        ("app[abc]*test", True),
        ("normal.package.name", False),
        ("", False),
    ])
    def test_has_wildcards(self, pattern, expected):
        """Test wildcard detection in patterns."""
        policy = TracingPolicy()
        assert policy._has_wildcards(pattern) == expected

    # ======================= PATTERN COMPILATION TESTS =======================
    
    def test_compile_pattern(self):
        """Test pattern compilation to regex."""
        policy = TracingPolicy()
        
        # Test various wildcard patterns
        patterns_and_tests = [
            ("app*", ["app", "application", "app_test"], ["myapp", "webapp"]),
            ("test_*", ["test_unit", "test_integration"], ["test", "mytest_unit"]),
            ("*.models", ["myapp.models", "core.user.models"], ["models", "myapp.model"]),
            ("api_v?", ["api_v1", "api_v2", "api_vx"], ["api_v", "api_v12"]),
            ("config[12]", ["config1", "config2"], ["config3", "config12"]),
        ]
        
        for pattern, should_match, should_not_match in patterns_and_tests:
            compiled = policy._compile_pattern(pattern)
            assert hasattr(compiled, 'match')  # Is a compiled regex
            
            for test_string in should_match:
                assert compiled.match(test_string), f"Pattern '{pattern}' should match '{test_string}'"
            
            for test_string in should_not_match:
                assert not compiled.match(test_string), f"Pattern '{pattern}' should not match '{test_string}'"

    # ======================= EXACT PATTERN MATCHING TESTS =======================
    
    @pytest.mark.parametrize("module_name,exact_patterns,expected", [
        # Direct exact matches
        ("myapp", {"myapp", "requests"}, True),
        ("requests", {"myapp", "requests"}, True),
        ("numpy", {"myapp", "requests"}, False),
        
        # Submodule matches
        ("myapp.services", {"myapp", "requests"}, True),
        ("myapp.models.user", {"myapp", "requests"}, True),
        ("requests.auth", {"myapp", "requests"}, True),
        ("requests.adapters.https", {"myapp", "requests"}, True),
        
        # Should NOT match (not proper submodules)
        ("myapplication", {"myapp", "requests"}, False),
        ("app_extended", {"myapp", "requests"}, False),
        ("requests_auth", {"myapp", "requests"}, False),
        
        # Edge cases
        ("", {"myapp", "requests"}, False),
        ("app", {"myapp", "requests"}, False),
        
        # Empty pattern set
        ("myapp", set(), False),
        ("myapp.services", set(), False),
    ])
    def test_matches_exact_patterns(self, module_name, exact_patterns, expected):
        """Test exact pattern matching logic."""
        policy = TracingPolicy()
        assert policy._matches_exact_patterns(module_name, exact_patterns) == expected

    def test_matches_exact_patterns_performance(self):
        """Test that exact pattern matching is fast (O(1) for direct matches)."""
        large_pattern_set = {f"package_{i}" for i in range(1000)}
        policy = TracingPolicy()
        
        # This should be fast even with large pattern set
        start = time.time()
        for _ in range(1000):
            # Direct match should be O(1)
            policy._matches_exact_patterns("package_500", large_pattern_set)
            # Submodule match should be O(P) but still reasonable
            policy._matches_exact_patterns("package_500.submodule", large_pattern_set)
        end = time.time()
        
        # Should complete quickly (this is a smoke test)
        assert (end - start) < 0.5  # Should be much faster than 0.5 seconds

    # ======================= WILDCARD PATTERN MATCHING TESTS =======================
    
    def test_matches_wildcard_patterns(self):
        """Test wildcard pattern matching with compiled regex."""
        policy = TracingPolicy()
        
        # Create compiled patterns
        wildcard_patterns = [
            policy._compile_pattern("app*"),
            policy._compile_pattern("test_*"),
            policy._compile_pattern("*.models"),
        ]
        
        # Test matches
        assert policy._matches_wildcard_patterns("application", wildcard_patterns) is True
        assert policy._matches_wildcard_patterns("app_extended", wildcard_patterns) is True
        assert policy._matches_wildcard_patterns("test_unit", wildcard_patterns) is True
        assert policy._matches_wildcard_patterns("myapp.models", wildcard_patterns) is True
        assert policy._matches_wildcard_patterns("core.user.models", wildcard_patterns) is True
        
        # Test non-matches
        assert policy._matches_wildcard_patterns("myapp", wildcard_patterns) is False
        assert policy._matches_wildcard_patterns("webapp", wildcard_patterns) is False
        assert policy._matches_wildcard_patterns("test", wildcard_patterns) is False
        assert policy._matches_wildcard_patterns("models", wildcard_patterns) is False
        assert policy._matches_wildcard_patterns("requests", wildcard_patterns) is False

    def test_matches_wildcard_patterns_empty_list(self):
        """Test wildcard matching with empty pattern list."""
        policy = TracingPolicy()
        assert policy._matches_wildcard_patterns("any.module", []) is False

    # ======================= SHOULD_TRACE_PACKAGE INTEGRATION TESTS =======================
    
    def test_should_trace_package_exact_patterns_only(self):
        """Test should_trace_package with only exact patterns."""
        policy = TracingPolicy(
            included_packages=["myapp", "requests"],
            auto_trace_subpackages=False
        )
        
        # Exact matches
        assert policy.should_trace_package("myapp") is True
        assert policy.should_trace_package("requests") is True
        
        # Submodule matches
        assert policy.should_trace_package("myapp.services") is True
        assert policy.should_trace_package("requests.auth") is True
        
        # Non-matches
        assert policy.should_trace_package("application") is False
        assert policy.should_trace_package("myapplication") is False
        assert policy.should_trace_package("numpy") is False

    def test_should_trace_package_wildcard_patterns_only(self):
        """Test should_trace_package with only wildcard patterns."""
        policy = TracingPolicy(
            included_packages=["app*", "test_*", "*.models"],
            auto_trace_subpackages=False
        )
        
        # Wildcard matches
        assert policy.should_trace_package("app") is True
        assert policy.should_trace_package("application") is True
        assert policy.should_trace_package("app_extended") is True
        assert policy.should_trace_package("test_unit") is True
        assert policy.should_trace_package("test_integration") is True
        assert policy.should_trace_package("myapp.models") is True
        assert policy.should_trace_package("core.user.models") is True
        
        # Non-matches
        assert policy.should_trace_package("myapp") is False
        assert policy.should_trace_package("webapp") is False
        assert policy.should_trace_package("test") is False
        assert policy.should_trace_package("models") is False

    def test_should_trace_package_mixed_patterns(self):
        """Test should_trace_package with both exact and wildcard patterns."""
        policy = TracingPolicy(
            included_packages=["myapp", "requests", "test_*", "api*"],
            auto_trace_subpackages=False
        )
        
        # Exact pattern matches (should be faster)
        assert policy.should_trace_package("myapp") is True
        assert policy.should_trace_package("myapp.services") is True
        assert policy.should_trace_package("requests.auth") is True
        
        # Wildcard pattern matches
        assert policy.should_trace_package("test_unit") is True
        assert policy.should_trace_package("api_v1") is True
        assert policy.should_trace_package("application") is False
        
        # Non-matches
        assert policy.should_trace_package("numpy") is False
        assert policy.should_trace_package("webapp") is False

    def test_should_trace_package_auto_trace_subpackages(self):
        """Test should_trace_package with auto_trace_subpackages enabled."""
        policy = TracingPolicy(
            project_root_packages=["src", "lib*"],
            included_packages=["requests"],
            auto_trace_subpackages=True
        )
        
        # Auto-trace project roots (exact)
        assert policy.should_trace_package("src") is True
        assert policy.should_trace_package("src.main") is True
        assert policy.should_trace_package("src.myapp.services") is True
        
        # Auto-trace project roots (wildcard)
        assert policy.should_trace_package("lib1") is True
        assert policy.should_trace_package("library") is True
        assert policy.should_trace_package("lib_test.utils") is True
        
        # Included packages still work
        assert policy.should_trace_package("requests.auth") is True
        
        # Non-matches
        assert policy.should_trace_package("mysrc") is False
        assert policy.should_trace_package("other") is False

    def test_should_trace_package_priority_order(self):
        """Test that exact patterns are checked before wildcard patterns."""
        policy = TracingPolicy(
            included_packages=["app", "app*"],  # Both exact and wildcard for same prefix
            auto_trace_subpackages=False
        )
        
        # These should match via exact pattern (faster path)
        assert policy.should_trace_package("app") is True
        assert policy.should_trace_package("app.services") is True
        
        # This should match via wildcard pattern
        assert policy.should_trace_package("application") is True

    # ======================= PERFORMANCE TESTS =======================
    
    def test_performance_large_pattern_sets(self):
        """Test performance with large numbers of patterns."""
        # Create policy with many patterns
        exact_patterns = [f"package_{i}" for i in range(100)]
        wildcard_patterns = [f"wild_{i}*" for i in range(10)]  # Fewer wildcards
        
        policy = TracingPolicy(
            included_packages=exact_patterns + wildcard_patterns
        )
        
        test_modules = [
            "package_50.module",  # Should match exact
            "wild_5_extended",    # Should match wildcard
            "nomatch.module",     # Should not match
        ]
        
        # Should handle large pattern sets efficiently
        start = time.time()
        for _ in range(1000):
            for module in test_modules:
                policy.should_trace_package(module)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 1.0  # Less than 1 second for 3000 checks

    # ======================= BACKWARDS COMPATIBILITY TESTS =======================
    
    def test_backwards_compatibility_with_original(self):
        """Test that TracingPolicy behaves identically to original for exact patterns."""
        patterns = ["myapp", "requests", "numpy"]
        
        optimized_policy = TracingPolicy(included_packages=patterns)
        
        test_cases = [
            ("myapp", True),
            ("myapp.services", True),
            ("requests.auth", True),
            ("numpy.array", True),
            ("myapplication", False),  # Key fix - should be False
            ("app_extended", False),   # Key fix - should be False
            ("requests_auth", False),  # Key fix - should be False
            ("pandas", False),
            ("", False),
        ]
        
        for module_name, expected in test_cases:
            assert optimized_policy.should_trace_package(module_name) == expected

    # ======================= EDGE CASES AND ERROR HANDLING =======================
    
    def test_empty_patterns(self):
        """Test behavior with empty pattern lists."""
        policy = TracingPolicy(
            included_packages=[],
            project_root_packages=[],
            auto_trace_subpackages=False
        )
        
        assert policy.should_trace_package("any.module") is False
        assert policy.should_trace_package("") is False

    def test_none_patterns(self):
        """Test behavior with None pattern lists."""
        policy = TracingPolicy(
            included_packages=None,
            project_root_packages=None,
            auto_trace_subpackages=False
        )
        
        assert policy.should_trace_package("any.module") is False

    def test_complex_wildcard_patterns(self):
        """Test complex wildcard patterns work correctly."""
        policy = TracingPolicy(
            included_packages=[
                "test_*_utils",           # Multi-part wildcard
                "api_v[12]",              # Character class
                "*.models",               # Suffix wildcard
                "django*.*admin*",        # Multiple wildcards
            ]
        )
        
        # Test multi-part wildcards
        assert policy.should_trace_package("test_unit_utils") is True
        assert policy.should_trace_package("test_integration_utils") is True
        assert policy.should_trace_package("test_utils") is False
        
        # Test character classes
        assert policy.should_trace_package("api_v1") is True
        assert policy.should_trace_package("api_v2") is True
        assert policy.should_trace_package("api_v3") is False
        
        # Test suffix wildcards
        assert policy.should_trace_package("myapp.models") is True
        assert policy.should_trace_package("core.user.models") is True
        assert policy.should_trace_package("models") is False
        
        # Test multiple wildcards
        assert policy.should_trace_package("django.contrib.admin") is True
        assert policy.should_trace_package("django_rest.admin_custom") is True

    def test_unicode_module_names(self):
        """Test handling of unicode characters in module names and patterns."""
        policy = TracingPolicy(
            included_packages=["测试包", "пакет*", "パッケージ.モジュール"]
        )
        
        assert policy.should_trace_package("测试包") is True
        assert policy.should_trace_package("测试包.模块") is True
        assert policy.should_trace_package("пакет_тест") is True
        assert policy.should_trace_package("パッケージ.モジュール") is True

    def test_very_long_module_names(self):
        """Test performance with very long module names."""
        policy = TracingPolicy(
            included_packages=["myapp", "very*"]
        )
        
        # Create very long module name
        long_module = "very." + ".".join([f"long_part_{i}" for i in range(100)])
        
        # Should handle long names efficiently
        start = time.time()
        result = policy.should_trace_package(long_module)
        elapsed = time.time() - start
        
        assert result is True  # Should match "very*"
        assert elapsed < 0.001  # Should be very fast

    # ======================= INTEGRATION WITH LOAD METHOD =======================
    
    def test_load_creates_optimized_policy(self, tmp_path):
        """Test that load method creates properly optimized policy."""
        policy_data = {
            "included_packages": ["myapp", "test_*", "api*"],
            "project_root_packages": ["src", "lib?"],
            "trace_depth": 8,
            "auto_trace_subpackages": True
        }
        
        policy_file = tmp_path / "glimpse-policy.json"
        policy_file.write_text(json.dumps(policy_data))
        
        # This test assumes load method is updated to use TracingPolicy
        # You'll need to update the load method accordingly
        policy = TracingPolicy.load(str(policy_file))
        
        # Verify optimization worked
        assert policy.exact_included == {"myapp"}
        assert len(policy.wildcard_included) == 2  # test_*, api*
        assert policy.exact_project_roots == {"src"}
        assert len(policy.wildcard_project_roots) == 1  # lib?
        assert policy.trace_depth == 8
        assert policy.auto_trace_subpackages is True