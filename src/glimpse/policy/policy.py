import json
import inspect
from pathlib import Path
from typing import Optional, Set, List, Pattern
import fnmatch
import re

class OldTracingPolicy:
    def __init__(self, included_packages=None, trace_depth=5, project_root_packages=None, auto_trace_subpackages=False):
        self.included_packages = set(included_packages or [])
        self.project_root_packages = set(project_root_packages or [])
        self.trace_depth = trace_depth 
        self.auto_trace_subpackages = auto_trace_subpackages if self.project_root_packages is not None and isinstance(self.project_root_packages, set) and len(self.project_root_packages) else False

    @classmethod
    def load(cls, policy_path: Optional[str] = None):
        """
        Load TracingPolicy from JSON file.
        
        Args:
            policy_path: Explicit path to policy file. Must be named 'glimpse-policy.json'
                        If None, searches for closest 'glimpse-policy.json' from caller location
        
        Returns:
            TracingPolicy instance
            
        Raises:
            FileNotFoundError: If policy file not found
            ValueError: If provided path doesn't end with 'glimpse-policy.json'
            JSONDecodeError: If policy file has invalid JSON
        """
        
        if policy_path is not None:
            # Case 1: Explicit path provided
            policy_file = Path(policy_path)
            
            # Validate filename
            if policy_file.name != "glimpse-policy.json":
                raise ValueError(
                    f"Policy file must be named 'glimpse-policy.json', got '{policy_file.name}'"
                )
            
            # Check if file exists
            if not policy_file.exists():
                raise FileNotFoundError(f"Policy file not found: {policy_path}")
                
            target_file = policy_file
            
        else:
            # Case 2: Discover closest policy file from caller location
            caller_frame = inspect.currentframe().f_back
            caller_file = Path(caller_frame.f_code.co_filename).resolve()
            
            # Walk up directory tree from caller's location
            search_dir = caller_file.parent
            target_file = None
            
            while search_dir != search_dir.parent:  # Stop at filesystem root
                candidate = search_dir / "glimpse-policy.json"
                if candidate.exists():
                    target_file = candidate
                    break
                search_dir = search_dir.parent
            
            if target_file is None:
                raise FileNotFoundError(
                    f"No 'glimpse-policy.json' found in directory tree starting from {caller_file.parent}"
                )
        
        # Load and parse JSON
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                policy_data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in policy file {target_file}: {e.msg}",
                e.doc,
                e.pos
            )
        
        # Extract configuration with defaults
        included_packages = policy_data.get("included_packages", [])
        trace_depth = policy_data.get("trace_depth", 5)
        auto_trace_subpackages = policy_data.get("auto_trace_subpackages", False)
        project_root_packages = policy_data.get("project_root_packages", None)
        
        return cls(
            included_packages=included_packages,
            trace_depth=trace_depth,
            auto_trace_subpackages=auto_trace_subpackages,
            project_root_packages=project_root_packages
        )

    def should_trace_package(self, module_name: str) -> bool:
        """
        Check if a package should be traced based on policy rules.
        
        Supports two matching modes:
        1. Exact module hierarchy matching: "app" matches "app" and "app.module" but not "application"
        2. Wildcard matching: "app*" matches "app", "application", "app_extended", etc.
        
        Args:
            module_name: The fully qualified module name (e.g., 'myapp.services')
            
        Returns:
            True if module should be traced, False otherwise
        """
        if self.auto_trace_subpackages:
            for root in self.project_root_packages:
                if self._matches_package_pattern(module_name, root):
                    return True
        
        if not self.included_packages:
            return False
            
        for package in self.included_packages:
            if self._matches_package_pattern(module_name, package):
                return True

        return False

    def _matches_package_pattern(self, module_name: str, package_pattern: str) -> bool:
        # Handle wildcard patterns
        if '*' in package_pattern or '?' in package_pattern or '[' in package_pattern:
            return fnmatch.fnmatch(module_name, package_pattern)
        
        # Handle exact module hierarchy matching
        return (
            module_name == package_pattern or  # Exact match: "app" == "app"
            module_name.startswith(package_pattern + ".")  # Submodule: "app.services"
        )


class TracingPolicy:
    def __init__(self, included_packages=None, trace_depth=5, project_root_packages=None, auto_trace_subpackages=False):
        # Store original patterns
        included = included_packages or []
        project_roots = project_root_packages or []
        
        # Separate exact patterns from wildcard patterns for optimization
        self.exact_included = set()
        self.wildcard_included = set()
        self.exact_project_roots = set()
        self.wildcard_project_roots = set()
        
        # Pre-compile patterns for better performance
        for pattern in included:
            if self._has_wildcards(pattern):
                self.wildcard_included.add(self._compile_pattern(pattern))
            else:
                self.exact_included.add(pattern)
                
        for pattern in project_roots:
            if self._has_wildcards(pattern):
                self.wildcard_project_roots.add(self._compile_pattern(pattern))
            else:
                self.exact_project_roots.add(pattern)
        
        self.trace_depth = trace_depth
        self.auto_trace_subpackages = auto_trace_subpackages and (len(self.exact_project_roots) > 0 or len(self.wildcard_project_roots) > 0)

    def _has_wildcards(self, pattern: str) -> bool:
        """Check if pattern contains wildcard characters."""
        return any(char in pattern for char in '*?[]')

    def _compile_pattern(self, pattern: str) -> Pattern:
        """Pre-compile wildcard pattern to regex for faster matching."""
        # Convert fnmatch pattern to regex for better performance
        regex_pattern = fnmatch.translate(pattern)
        return re.compile(regex_pattern)

    @classmethod
    def load(cls, policy_path: Optional[str] = None):
        """
        Load TracingPolicy from JSON file.
        
        Args:
            policy_path: Explicit path to policy file. Must be named 'glimpse-policy.json'
                        If None, searches for closest 'glimpse-policy.json' from caller location
        
        Returns:
            TracingPolicy instance
            
        Raises:
            FileNotFoundError: If policy file not found
            ValueError: If provided path doesn't end with 'glimpse-policy.json'
            JSONDecodeError: If policy file has invalid JSON
        """
        
        if policy_path is not None:
            # Case 1: Explicit path provided
            policy_file = Path(policy_path)
            
            # Validate filename
            if policy_file.name != "glimpse-policy.json":
                raise ValueError(
                    f"Policy file must be named 'glimpse-policy.json', got '{policy_file.name}'"
                )
            
            # Check if file exists
            if not policy_file.exists():
                raise FileNotFoundError(f"Policy file not found: {policy_path}")
                
            target_file = policy_file
            
        else:
            # Case 2: Discover closest policy file from caller location
            caller_frame = inspect.currentframe().f_back
            caller_file = Path(caller_frame.f_code.co_filename).resolve()
            
            # Walk up directory tree from caller's location
            search_dir = caller_file.parent
            target_file = None
            
            while search_dir != search_dir.parent:  # Stop at filesystem root
                candidate = search_dir / "glimpse-policy.json"
                if candidate.exists():
                    target_file = candidate
                    break
                search_dir = search_dir.parent
            
            if target_file is None:
                raise FileNotFoundError(
                    f"No 'glimpse-policy.json' found in directory tree starting from {caller_file.parent}"
                )
        
        # Load and parse JSON
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                policy_data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in policy file {target_file}: {e.msg}",
                e.doc,
                e.pos
            )
        
        # Extract configuration with defaults
        included_packages = policy_data.get("included_packages", [])
        trace_depth = policy_data.get("trace_depth", 5)
        auto_trace_subpackages = policy_data.get("auto_trace_subpackages", False)
        project_root_packages = policy_data.get("project_root_packages", None)
        
        return cls(
            included_packages=included_packages,
            trace_depth=trace_depth,
            auto_trace_subpackages=auto_trace_subpackages,
            project_root_packages=project_root_packages
        )


    def should_trace_package(self, module_name: str) -> bool:
        # Fast path: Check exact patterns first (O(1) set lookup)
        if self.auto_trace_subpackages:
            if self._matches_exact_patterns(module_name, self.exact_project_roots):
                return True
            if self._matches_wildcard_patterns(module_name, self.wildcard_project_roots):
                return True
        
        # Check included packages
        if self._matches_exact_patterns(module_name, self.exact_included):
            return True
            
        if self._matches_wildcard_patterns(module_name, self.wildcard_included):
            return True
            
        return False

    def _matches_exact_patterns(self, module_name: str, exact_patterns: Set[str]) -> bool:
        if not exact_patterns:
            return False
            
        # Direct exact match - O(1) average case
        if module_name in exact_patterns:
            return True
            
        # Check if it's a submodule of any exact pattern
        # Optimization: Only check patterns that could be prefixes
        for pattern in exact_patterns:
            if len(pattern) < len(module_name) and module_name.startswith(pattern + "."):
                return True
                
        return False

    def _matches_wildcard_patterns(self, module_name: str, wildcard_patterns: List[Pattern]) -> bool:
        """
        Check wildcard pattern matching with pre-compiled regex.
        
        Time: O(W * M) where W=wildcard patterns, M=module name length
        Space: O(1)
        """
        for compiled_pattern in wildcard_patterns:
            if compiled_pattern.match(module_name):
                return True
        return False


def generate_test_data():
    """Generate realistic test data for benchmarking."""
    # Realistic module names
    module_names = [
        "myapp.services.user", "myapp.models.account", "myapp.utils.helpers",
        "requests.auth.basic", "requests.adapters.http", "requests.sessions",
        "numpy.core.array", "pandas.io.common", "django.contrib.admin",
        "application.core.models", "app_extended.views", "test_utils.mock"
    ]
    
    # Mix of exact and wildcard patterns
    patterns = [
        "myapp", "requests", "numpy",  # Exact patterns
        "app*", "test_*", "django*",   # Wildcard patterns
        "*.models", "api_v[12]"        # Complex wildcards
    ]
    
    return module_names, patterns

def benchmark_original_vs_optimized():
    import time
    import random
    """Compare performance of original vs optimized implementation."""
    module_names, patterns = generate_test_data()
    
    # Original implementation
    original_policy = OldTracingPolicy(included_packages=patterns)
    
    # Optimized implementation  
    optimized_policy = TracingPolicy(included_packages=patterns)
    
    num_iterations = 100000
    
    # Benchmark original
    start = time.time()
    for _ in range(num_iterations):
        module_name = random.choice(module_names)
        original_policy.should_trace_package(module_name)
    original_time = time.time() - start
    
    # Benchmark optimized
    start = time.time()
    for _ in range(num_iterations):
        module_name = random.choice(module_names)
        optimized_policy.should_trace_package(module_name)
    optimized_time = time.time() - start
    
    print(f"Original: {original_time:.4f}s ({num_iterations} calls)")
    print(f"Optimized: {optimized_time:.4f}s ({num_iterations} calls)")
    print(f"Speedup: {original_time / optimized_time:.2f}x")

if __name__ == "__main__":
    benchmark_original_vs_optimized()