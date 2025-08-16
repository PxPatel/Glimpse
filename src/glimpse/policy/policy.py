import json
import inspect
from pathlib import Path
from typing import Optional, Set, List, Pattern
import fnmatch
import re

class ExactPatternTrie:
    """Trie data structure for efficient exact pattern matching."""
    
    def __init__(self):
        self.root = {}
    
    def add_pattern(self, pattern: str):
        """Add a pattern to the trie."""
        if not pattern.strip():
            return
            
        node = self.root
        segments = pattern.split('.')
        
        for segment in segments:
            if segment not in node:
                node[segment] = {}
            node = node[segment]
        
        # Mark this node as a complete pattern
        node['_is_pattern'] = True
    
    def matches(self, module_name: str) -> bool:
        """Check if module_name matches any pattern in the trie."""
        if not module_name.strip():
            return False
            
        node = self.root
        segments = module_name.split('.')
        
        for segment in segments:
            if segment not in node:
                return False
            
            node = node[segment]
            
            # Check if we've found a complete pattern at this level
            if node.get('_is_pattern', False):
                return True
        
        return False


# Modified TracingPolicy class with trie optimization
class TracingPolicy:
    """TracingPolicy with trie-optimized exact pattern matching."""
    
    def __init__(self, included_packages=None, trace_depth=5, project_root_packages=None, auto_trace_subpackages=False):
        # Store original patterns
        included = included_packages or []
        project_roots = project_root_packages or []
        
        # Separate exact patterns from wildcard patterns for optimization
        self.exact_included_trie = ExactPatternTrie()
        self.wildcard_included = set()
        self.exact_project_roots_trie = ExactPatternTrie()
        self.wildcard_project_roots = set()
        
        # Pre-compile patterns for better performance
        for pattern in included:
            if not pattern.strip():
                continue
            if self._has_wildcards(pattern):
                self.wildcard_included.add(self._compile_pattern(pattern))
            else:
                self.exact_included_trie.add_pattern(pattern)
                
        for pattern in project_roots:
            if not pattern.strip():
                continue
            if self._has_wildcards(pattern):
                self.wildcard_project_roots.add(self._compile_pattern(pattern))
            else:
                self.exact_project_roots_trie.add_pattern(pattern)
        
        self.trace_depth = trace_depth
        self.auto_trace_subpackages = auto_trace_subpackages and (
            bool(self.exact_project_roots_trie.root) or len(self.wildcard_project_roots) > 0
        )

    def _has_wildcards(self, pattern: str) -> bool:
        """Check if pattern contains wildcard characters."""
        return any(char in pattern for char in '*?[]')

    def _compile_pattern(self, pattern: str) -> Pattern: 
        """Pre-compile wildcard pattern to regex for faster matching."""
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
        """Check if a package should be traced based on policy rules."""
        if self.auto_trace_subpackages:
            if self._op_matches_exact_patterns_(module_name, self.exact_project_roots_trie):
                return True
            if self._matches_wildcard_patterns(module_name, self.wildcard_project_roots):
                return True
        
        # Check included packages
        if self._op_matches_exact_patterns_(module_name, self.exact_included_trie):
            return True
            
        if self._matches_wildcard_patterns(module_name, self.wildcard_included):
            return True
            
        return False

    @staticmethod
    def _op_matches_exact_patterns_(module_name: str, exact_patterns_trie: ExactPatternTrie) -> bool:
        """ Optimized O(k) exact pattern matching using trie data structure."""
        return exact_patterns_trie.matches(module_name)

    @staticmethod
    def _matches_wildcard_patterns(module_name: str, wildcard_patterns: List[Pattern]):
        """Check wildcard pattern matching with pre-compiled regex."""
        for compiled_pattern in wildcard_patterns:
            if compiled_pattern.match(module_name):
                return True
        return False
