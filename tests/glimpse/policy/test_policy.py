import pytest
import json
import re
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from glimpse.policy.policy import TracingPolicy, ExactPatternTrie

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



class TestTracingPolicy:
    """Comprehensive unit tests for TracingPolicy class with trie optimization."""
    
    # ======================= INITIALIZATION TESTS =======================
    
    def test_init_with_defaults(self):
        """Test TracingPolicy initialization with default values."""
        policy = TracingPolicy()
        
        assert isinstance(policy.exact_included_trie, ExactPatternTrie)
        assert policy.exact_included_trie.root == {}
        assert isinstance(policy.wildcard_included, set)
        assert len(policy.wildcard_included) == 0
        assert isinstance(policy.exact_project_roots_trie, ExactPatternTrie)
        assert policy.exact_project_roots_trie.root == {}
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
        
        # All patterns should be in exact tries
        assert policy.exact_included_trie.matches("myapp") is True
        assert policy.exact_included_trie.matches("requests") is True
        assert policy.exact_included_trie.matches("numpy") is True
        assert len(policy.wildcard_included) == 0
        
        assert policy.exact_project_roots_trie.matches("src") is True
        assert policy.exact_project_roots_trie.matches("lib") is True
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
        assert policy.exact_included_trie.root == {}
        assert len(policy.wildcard_included) == 3
        assert policy.exact_project_roots_trie.root == {}
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
        
        # Exact patterns should be in tries
        assert policy.exact_included_trie.matches("myapp") is True
        assert policy.exact_included_trie.matches("requests") is True
        assert policy.exact_project_roots_trie.matches("src") is True
        
        # Wildcard patterns should be compiled
        assert len(policy.wildcard_included) == 2  # test_*, app*
        assert len(policy.wildcard_project_roots) == 1  # lib*
        
        # Verify compiled patterns work
        found_test_pattern = False
        found_app_pattern = False
        for pattern in policy.wildcard_included:
            if pattern.match("test_unit"):
                found_test_pattern = True
            if pattern.match("application"):
                found_app_pattern = True
        
        assert found_test_pattern is True
        assert found_app_pattern is True

    def test_init_with_empty_and_whitespace_patterns(self):
        """Test initialization filters out empty and whitespace patterns."""
        policy = TracingPolicy(
            included_packages=["myapp", "", "  ", "\t\n", "requests"],
            project_root_packages=["src", "", "   ", "lib"]
        )
        
        # Only non-empty patterns should be added
        assert policy.exact_included_trie.matches("myapp") is True
        assert policy.exact_included_trie.matches("requests") is True
        assert policy.exact_project_roots_trie.matches("src") is True
        assert policy.exact_project_roots_trie.matches("lib") is True
        
        # Empty patterns should not be in trie
        assert policy.exact_included_trie.matches("") is False

    def test_init_auto_trace_logic(self):
        """Test auto_trace_subpackages logic based on project root packages."""
        # Should be False when no project roots
        policy1 = TracingPolicy(auto_trace_subpackages=True)
        assert policy1.auto_trace_subpackages is False
        
        # Should be True when exact project roots exist
        policy2 = TracingPolicy(
            project_root_packages=["src"],
            auto_trace_subpackages=True
        )
        assert policy2.auto_trace_subpackages is True
        
        # Should be True when wildcard project roots exist
        policy3 = TracingPolicy(
            project_root_packages=["src*"],
            auto_trace_subpackages=True
        )
        assert policy3.auto_trace_subpackages is True
        
        # Should be False when explicitly disabled
        policy4 = TracingPolicy(
            project_root_packages=["src"],
            auto_trace_subpackages=False
        )
        assert policy4.auto_trace_subpackages is False

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

    # ======================= TRIE-BASED EXACT PATTERN MATCHING TESTS =======================
    
    @pytest.mark.parametrize("module_name,patterns,expected", [
        # Direct exact matches
        ("myapp", ["myapp", "requests"], True),
        ("requests", ["myapp", "requests"], True),
        ("numpy", ["myapp", "requests"], False),
        
        # Submodule matches
        ("myapp.services", ["myapp", "requests"], True),
        ("myapp.models.user", ["myapp", "requests"], True),
        ("requests.auth", ["myapp", "requests"], True),
        ("requests.adapters.https", ["myapp", "requests"], True),
        
        # Should NOT match (not proper submodules)
        ("myapplication", ["myapp", "requests"], False),
        ("app_extended", ["myapp", "requests"], False),
        ("requests_auth", ["myapp", "requests"], False),
        
        # Edge cases
        ("", ["myapp", "requests"], False),
        ("app", ["myapp", "requests"], False),
        
        # Empty pattern set
        ("myapp", [], False),
        ("myapp.services", [], False),
    ])
    def test_trie_exact_pattern_matching(self, module_name, patterns, expected):
        """Test trie-based exact pattern matching logic."""
        trie = ExactPatternTrie()
        for pattern in patterns:
            trie.add_pattern(pattern)
        
        assert TracingPolicy._op_matches_exact_patterns_(module_name, trie) == expected

    def test_trie_exact_patterns_performance(self):
        """Test that trie exact pattern matching is fast (O(k) where k = segments)."""
        # Create trie with many patterns
        trie = ExactPatternTrie()
        for i in range(1000):
            trie.add_pattern(f"package_{i}")
        
        # This should be fast even with large pattern set because trie depth is limited
        start = time.time()
        for _ in range(1000):
            # Should be O(k) where k = number of segments, not O(n) where n = patterns
            TracingPolicy._op_matches_exact_patterns_("package_500.submodule.deep.path", trie)
        end = time.time()
        
        # Should complete quickly (this is a smoke test for O(k) vs O(n) performance)
        assert (end - start) < 0.1  # Should be much faster than old O(n*m) implementation

    def test_trie_memory_efficiency(self):
        """Test that trie shares common prefixes efficiently."""
        trie = ExactPatternTrie()
        
        # Add many patterns with shared prefixes
        patterns = [
            "myapp.services.user",
            "myapp.services.auth", 
            "myapp.services.admin",
            "myapp.models.user",
            "myapp.models.account",
            "myapp.utils.helpers",
            "myapp.utils.validators"
        ]
        
        for pattern in patterns:
            trie.add_pattern(pattern)
        
        # All should share the "myapp" node
        myapp_node = trie.root["myapp"]
        assert "services" in myapp_node
        assert "models" in myapp_node
        assert "utils" in myapp_node
        
        # Services should share the services node
        services_node = myapp_node["services"]
        assert "user" in services_node
        assert "auth" in services_node
        assert "admin" in services_node
        
        # All endpoints should be marked as patterns
        assert services_node["user"]["_is_pattern"] is True
        assert services_node["auth"]["_is_pattern"] is True
        assert services_node["admin"]["_is_pattern"] is True

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
        """Test should_trace_package with only exact patterns using trie."""
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
        
        # Exact pattern matches (should be faster due to trie)
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
        
        # Auto-trace project roots (exact via trie)
        assert policy.should_trace_package("src") is True
        assert policy.should_trace_package("src.main") is True
        assert policy.should_trace_package("src.myapp.services") is True
        
        # Auto-trace project roots (wildcard)
        assert policy.should_trace_package("lib1") is True
        assert policy.should_trace_package("library") is True
        assert policy.should_trace_package("lib_test.utils") is True
        
        # Included packages still work (via trie)
        assert policy.should_trace_package("requests.auth") is True
        
        # Non-matches
        assert policy.should_trace_package("mysrc") is False
        assert policy.should_trace_package("other") is False

    def test_should_trace_package_trie_priority(self):
        """Test that trie-based exact patterns are checked before wildcard patterns."""
        policy = TracingPolicy(
            included_packages=["app", "app*"],  # Both exact and wildcard for same prefix
            auto_trace_subpackages=False
        )
        
        # These should match via exact pattern in trie (faster path)
        assert policy.should_trace_package("app") is True
        assert policy.should_trace_package("app.services") is True
        
        # This should match via wildcard pattern
        assert policy.should_trace_package("application") is True

    # ======================= PERFORMANCE TESTS =======================
    
    def test_performance_comparison_trie_vs_old_implementation(self):
        """Test that trie implementation is faster than O(n*m) set-based approach."""
        # Create policy with many exact patterns
        exact_patterns = [f"package_{i}" for i in range(100)]
        
        policy = TracingPolicy(included_packages=exact_patterns)
        
        test_modules = [
            "package_50.module",     # Should match exact
            "package_75.deep.path",  # Should match exact 
            "nomatch.module",        # Should not match
        ]
        
        # With trie, performance should be O(k) regardless of pattern count
        start = time.time()
        for _ in range(1000):
            for module in test_modules:
                policy.should_trace_package(module)
        elapsed = time.time() - start
        
        # Should complete in reasonable time even with many patterns
        assert elapsed < 0.1  # Less than 100ms for 3000 checks with 100 patterns

    def test_performance_large_pattern_sets(self):
        """Test performance with large numbers of patterns."""
        # Create policy with many patterns
        exact_patterns = [f"package_{i}" for i in range(200)]
        wildcard_patterns = [f"wild_{i}*" for i in range(10)]  # Fewer wildcards
        
        policy = TracingPolicy(
            included_packages=exact_patterns + wildcard_patterns
        )
        
        test_modules = [
            "package_100.module",  # Should match exact via trie
            "wild_5_extended",     # Should match wildcard
            "nomatch.module",      # Should not match
        ]
        
        # Should handle large pattern sets efficiently
        start = time.time()
        for _ in range(1000):
            for module in test_modules:
                policy.should_trace_package(module)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 0.5  # Less than 0.5 seconds for 3000 checks

    def test_trie_depth_performance(self):
        """Test that trie performance depends on module depth, not pattern count."""
        # Create trie with many patterns of varying depths
        patterns = []
        for i in range(100):
            patterns.append(f"level1_{i}")
            patterns.append(f"level1_{i}.level2_{i}")
            patterns.append(f"level1_{i}.level2_{i}.level3_{i}")
        
        policy = TracingPolicy(included_packages=patterns)
        
        # Test modules at different depths
        test_cases = [
            ("level1_50", 1),           # 1 segment
            ("level1_50.level2_50", 2), # 2 segments  
            ("level1_50.level2_50.level3_50", 3), # 3 segments
            ("level1_50.level2_50.level3_50.level4_50", 4), # 4 segments
        ]
        
        for module_name, depth in test_cases:
            start = time.time()
            for _ in range(1000):
                policy.should_trace_package(module_name)
            elapsed = time.time() - start
            
            # Time should correlate with depth, not pattern count (300 patterns)
            # All should be fast since max depth is small
            assert elapsed < 0.05, f"Depth {depth} took too long: {elapsed}s"

    # ======================= BACKWARDS COMPATIBILITY TESTS =======================
    
    def test_backwards_compatibility_with_original(self):
        """Test that TracingPolicy behaves identically to original for exact patterns."""
        patterns = ["myapp", "requests", "numpy"]
        
        trie_policy = TracingPolicy(included_packages=patterns)
        
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
            actual = trie_policy.should_trace_package(module_name)
            assert actual == expected, f"Module '{module_name}': expected {expected}, got {actual}"

    def test_interface_compatibility(self):
        """Test that TracingPolicy maintains the same public interface."""
        policy = TracingPolicy(
            included_packages=["myapp", "test_*"],
            project_root_packages=["src"],
            trace_depth=8,
            auto_trace_subpackages=True
        )
        
        # Should have same public interface
        assert hasattr(policy, 'should_trace_package')
        assert hasattr(policy, 'trace_depth')
        assert hasattr(policy, 'auto_trace_subpackages')
        assert hasattr(policy, '_has_wildcards')
        assert hasattr(policy, '_compile_pattern')
        
        # Methods should work the same way
        assert callable(policy.should_trace_package)
        assert isinstance(policy.trace_depth, int)
        assert isinstance(policy.auto_trace_subpackages, bool)

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
        
        # Exact patterns via trie
        assert policy.should_trace_package("测试包") is True
        assert policy.should_trace_package("测试包.模块") is True
        assert policy.should_trace_package("パッケージ.モジュール") is True
        
        # Wildcard patterns
        assert policy.should_trace_package("пакет_тест") is True

    def test_very_long_module_names(self):
        """Test performance with very long module names."""
        policy = TracingPolicy(
            included_packages=["myapp", "very*"]
        )
        
        # Create very long module name
        long_module = "very." + ".".join([f"long_part_{i}" for i in range(50)])
        
        # Should handle long names efficiently
        start = time.time()
        result = policy.should_trace_package(long_module)
        elapsed = time.time() - start
        
        assert result is True  # Should match "very*"
        assert elapsed < 0.001  # Should be very fast

    def test_trie_stress_test(self):
        """Stress test the trie with many patterns and lookups."""
        # Create many patterns with shared prefixes
        patterns = []
        for i in range(50):
            base = f"package_{i}"
            patterns.append(base)
            for j in range(10):
                patterns.append(f"{base}.module_{j}")
                for k in range(5):
                    patterns.append(f"{base}.module_{j}.submodule_{k}")
        
        policy = TracingPolicy(included_packages=patterns)
        
        # Test many lookups
        test_modules = [
            "package_25.module_5.submodule_2",  # Should match
            "package_30.module_8",              # Should match
            "package_40",                       # Should match
            "other_package.module",             # Should not match
        ]
        
        # Should handle stress test efficiently
        start = time.time()
        for _ in range(100):
            for module in test_modules:
                policy.should_trace_package(module)
        elapsed = time.time() - start
        
        # Should complete quickly even with 2750 patterns
        assert elapsed < 0.1  # Less than 100ms for 400 checks

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
        
        policy = TracingPolicy.load(str(policy_file))
        
        # Verify trie optimization worked
        assert policy.exact_included_trie.matches("myapp") is True
        assert len(policy.wildcard_included) == 2  # test_*, api*
        assert policy.exact_project_roots_trie.matches("src") is True
        assert len(policy.wildcard_project_roots) == 1  # lib?
        assert policy.trace_depth == 8
        assert policy.auto_trace_subpackages is True

    # ======================= MEMORY USAGE TESTS =======================
    
    def test_trie_memory_vs_set_efficiency(self):
        """Test that trie uses memory efficiently compared to set approach."""
        patterns_with_shared_prefixes = [
            "myapp.services.user.auth",
            "myapp.services.user.profile", 
            "myapp.services.user.settings",
            "myapp.services.admin.users",
            "myapp.services.admin.groups",
            "myapp.models.user",
            "myapp.models.account",
            "myapp.utils.helpers",
            "myapp.utils.validators",
        ]
        
        policy = TracingPolicy(included_packages=patterns_with_shared_prefixes)
        
        # Verify all patterns work
        for pattern in patterns_with_shared_prefixes:
            assert policy.should_trace_package(pattern) is True
            assert policy.should_trace_package(pattern + ".extra") is True
        
        # Verify trie structure shares prefixes
        root = policy.exact_included_trie.root
        assert "myapp" in root
        myapp_node = root["myapp"]
        assert "services" in myapp_node
        assert "models" in myapp_node
        assert "utils" in myapp_node
        
        # Services node should have user and admin
        services_node = myapp_node["services"]
        assert "user" in services_node
        assert "admin" in services_node
        
        # Each endpoint should be properly marked
        user_node = services_node["user"]
        assert "auth" in user_node
        assert "profile" in user_node
        assert "settings" in user_node
        assert user_node["auth"]["_is_pattern"] is True