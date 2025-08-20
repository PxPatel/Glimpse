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
        """Check if module_name matches any pattern in the trie (including submodules)."""
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


class ExactModuleSet:
    """Simple set-based storage for exact module matching (no submodules)."""
    
    def __init__(self):
        self.modules = set()
    
    def add_module(self, module: str):
        """Add a module to the exact set."""
        if module.strip():
            self.modules.add(module)
    
    def matches(self, module_name: str) -> bool:
        """Check if module_name exactly matches any stored module."""
        return module_name in self.modules


class TracingPolicy:
    """TracingPolicy with clean exact_modules vs package_trees distinction."""
    
    def __init__(self, exact_modules=None, package_trees=None, trace_depth=5):
        # Initialize collections
        exact_mods = exact_modules or []
        package_trees_list = package_trees or []
        
        # Exact modules - only match exactly, no submodules
        self.exact_modules_set = ExactModuleSet()
        self.exact_modules_wildcards = set()
        
        # Package trees - match package and all submodules  
        self.package_trees_trie = ExactPatternTrie()
        self.package_trees_wildcards = set()
        
        # Process exact modules
        for module in exact_mods:
            if not module.strip():
                continue
            if self._has_wildcards(module):
                self.exact_modules_wildcards.add(self._compile_exact_pattern(module))
            else:
                self.exact_modules_set.add_module(module)
        
        # Process package trees
        for package in package_trees_list:
            if not package.strip():
                continue
            if self._has_wildcards(package):
                self.package_trees_wildcards.add(self._compile_tree_pattern(package))
            else:
                self.package_trees_trie.add_pattern(package)
        
        self.trace_depth = trace_depth

    def _has_wildcards(self, pattern: str) -> bool:
        """Check if pattern contains wildcard characters."""
        return any(char in pattern for char in '*?[]')

    def _compile_exact_pattern(self, pattern: str) -> Pattern:
        """Compile wildcard pattern for exact module matching (no submodules)."""
        regex_pattern = fnmatch.translate(pattern)
        regex_pattern = regex_pattern.replace('.*', '[^.]*')
        return re.compile(regex_pattern)

    def _compile_tree_pattern(self, pattern: str) -> Pattern:
        """Compile wildcard pattern for package tree matching (including submodules)."""
        # For package trees, we want to match the pattern and any submodules
        # Convert fnmatch pattern to regex, but allow additional dot-separated segments
        base_regex = fnmatch.translate(pattern)
        # Remove the $ anchor and add optional submodule matching
        if base_regex.endswith('\\Z'):
            base_regex = base_regex[:-2]  # Remove \Z
        elif base_regex.endswith('$'):
            base_regex = base_regex[:-1]  # Remove $
        
        # Add pattern to match submodules: either end here or continue with .anything
        tree_regex = f"({base_regex})(\\..*)?"
        return re.compile(f"^{tree_regex}$")

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
        exact_modules = policy_data.get("exact_modules", [])
        package_trees = policy_data.get("package_trees", [])
        trace_depth = policy_data.get("trace_depth", 5)
        
        return cls(
            exact_modules=exact_modules,
            package_trees=package_trees,
            trace_depth=trace_depth
        )

    def should_trace_package(self, module_name: str) -> bool:
        """
        Check if a package should be traced based on policy rules.
        
        Priority order:
        1. Package trees (more permissive, wins over exact modules in overlap)
        2. Exact modules (only if not covered by package trees)
        """
        
        # First check package trees (wins in overlap scenarios)
        if self._matches_package_trees(module_name):
            return True
        
        # Then check exact modules  
        if self._matches_exact_modules(module_name):
            return True
            
        return False
    
    def _matches_package_trees(self, module_name: str) -> bool:
        """Check if module matches any package tree (including submodules)."""
        # Check exact package trees via trie (matches package + submodules)
        if self.package_trees_trie.matches(module_name):
            return True
        
        # Check wildcard package trees (should match package + submodules)
        for compiled_pattern in self.package_trees_wildcards:
            if compiled_pattern.match(module_name):
                return True
        
        return False
    
    def _matches_exact_modules(self, module_name: str) -> bool:
        """Check if module matches any exact module (no submodules)."""
        # Check exact string matches
        if self.exact_modules_set.matches(module_name):
            return True
        
        # Check wildcard exact matches (should NOT match submodules)
        for compiled_pattern in self.exact_modules_wildcards:
            if compiled_pattern.match(module_name):
                return True
        
        return False